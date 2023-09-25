from accelerate.utils import DistributedDataParallelKwargs
from fastcore.script import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.text.all import *
import wandb; from fastai.callback.wandb import *
from xcube.text.all import *
from fastai.metrics import accuracy # there's an 'accuracy' metric in xcube as well

# extra imports <remove later>
import warnings; warnings.filterwarnings(action='ignore')
# end extra imports

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80
def pr(s):
    if rank_distrib()==0: print(s)

@patch
def after_batch(self: ProgressCallback):
        self.pbar.update(self.iter+1)
        mets = ('_valid_mets', '_train_mets')[self.training]
        self.pbar.comment = ' '.join([f'{met.name} = {met.value.item():.4f}' for met in getattr(self.recorder, mets)])

def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].to_list()
    return train, valid

def get_dls(source, data, bs=384, sl=80, workers=None):
    workers = ifnone(workers, min(8, num_cpus()))
    data = join_path_file(data, source, ext='.csv')
    df = pd.read_csv(data,
                 header=0,
                 usecols=['text', 'labels'],
                 dtype={'text': str, 'labels': str})
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)
    dls_lm = DataBlock(
          blocks=TextBlock.from_df('text', is_lm=True),
          get_x=ColReader('text'),
          splitter=RandomSplitter(0.1)
    ).dataloaders(df, bs=bs, seq_len=sl)
    return dls_lm, dls_lm.vocab

@call_parse
def main(
    data:  Param("Filename of the raw data", str)="mimic4_icd10_full",
    bs:    Param("Batch size", int)=16,
    epochs:Param("Number of epochs", str)="[1, 15]",
    lrs:   Param("learning rates for gradual unfreezing", str)="[1e-3, 1e-6]",
    fp16:  Param("Use mixed precision training", store_true)=False,
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
    track_train: Param("Record training metrics", store_true)=False,
    wandblog: Param("Experiment tracking in wandb.ai", store_true)=False,
    log: Param("Log loss and metrics after each epoch", store_true)=False,
    workers:   Param("Number of workers", int)=None,
    save_model: Param("Save model on improvement after each epoch", store_true)=False,
    root_dir: Param("Root dir for saving models", str)="..",
    fname: Param("Save model file", str)="mimic4",
    infer: Param("Don't train, just validate", int)=0,
    train_from_ckpt: Param("Load the most recent model and train", store_true)=False
):
    "Training of AWD-LSTM langauge model."

    source = rank0_first(untar_xxx, XURLs.MIMIC4)
    # make tmp directory to save and load models and dataloaders
    tmp = Path(root_dir)/'tmp/models'
    tmp.mkdir(exist_ok=True, parents=True)
    tmp = tmp.parent

    #loading dataloaders
    dls_file = join_path_file(data+'_dls_lm', tmp, ext='.pkl')
    vocab_file = join_path_file(data+'_dls_lm_vocab', tmp, ext='.pkl')
    if dls_file.exists(): 
        dls_lm = torch.load(dls_file, map_location=torch.device('cpu'))
    else:
        dls_lm, vocab = get_dls(source, data, bs, workers=workers)
        torch.save(dls_lm, dls_file)
        torch.save(vocab, vocab_file)

    epochs = json.loads(epochs)
    lrs = json.loads(lrs)
    
    cbs = SaveModelCallback(monitor='accuracy', fname=fname, with_opt=True, reset_on_fit=False) if save_model else None
    if log: 
        logfname = join_path_file(fname, tmp, ext='.csv')
        if logfname.exists(): logfname.unlink()
        cbs += L(CSVLogger(fname=logfname, append=True))
    if wandblog: cbs += L(WandbCallback(log_preds=False, log_model=True, model_name=fname))
    learn = language_model_learner(
        dls_lm, AWD_LSTM, drop_mult=0.3,
        metrics=[accuracy, Perplexity()], cbs=cbs, path=tmp)
    if track_train: 
        assert learn.cbs[1].__class__ is Recorder
        setattr(learn.cbs[1], 'train_metrics', true)
    if dump: pr(learn.model); exit()
    if fp16: learn = learn.to_fp16()
    if infer:
        try: 
            learn = learn.load(learn.save_model.fname)
            validate(learn)
        except FileNotFoundError as e:
            print("Exception:", e)
            print("Trained model not found!")
        finally: exit()

    # Workaround: In PyTorch 2.0.1 need to set DistributedDataParallel() with find_unused_parameters=True,
    # to avoid a crash that only happens in distributed mode of xmltext_clasifier_learner.fit()
    ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
    cms = learn.distrib_ctx(kwargs_handlers=[ddp_scaler])
    if wandblog: cms += L(wandb.init())
    with ContextManagers(cms):
        if not train_from_ckpt:
            lr_min, lr_steep, lr_valley, lr_slide = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
            print(lr_min, lr_steep, lr_valley, lr_slide)
            learn.fit_one_cycle(epochs[0], lr_min)
            learn.unfreeze()
            learn.fit(epochs[1], lrs[1])
        else:
            learn = learn.load(learn.save_model.fname)
            validate(learn)
            print(f"We are monitoring {learn.save_model.monitor}")
            best = input(f"Input the best {learn.save_model.monitor} so far: ")
            learn.save_model.best = float(best)
            epochs = int(input("how many epochs do you you want to train: "))
            learn.unfreeze()
            learn.fit(epochs, lrs[1])
    #saving the encoder of the LM
    learn.save_encoder(fname+'_finetuned')
    #saving the decoder of the LM
    learn.save_decoder(fname+'_decoder')
