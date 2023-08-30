from accelerate.utils import DistributedDataParallelKwargs
from fastcore.script import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.text.all import *
import wandb; from fastai.callback.wandb import *
from xcube.text.all import *
from fastai.metrics import accuracy # there's an 'accuracy' metric in xcube as well

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

def get_dls(source, data, bs, sl=16, workers=None):
    workers = ifnone(workers,min(8,num_cpus()))
    data = join_path_file(data, source, ext='.csv')
    df = pd.read_csv(data,
                 header=0,
                 names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid'],
                 dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool})
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)
    lbl_freqs = Counter()
    for labels in df.labels: lbl_freqs.update(labels.split(';'))
    lbls = list(lbl_freqs.keys())
    splits = splitter(df)
    lm_vocab = torch.load(source/'mimic3-9k_dls_lm_vocab.pkl')
    x_tfms = [Tokenizer.from_df('text', n_workers=workers), attrgetter("text"), Numericalize(vocab=lm_vocab)]
    y_tfms = [ColReader('labels', label_delim=';'), MultiCategorize(vocab=lbls), OneHotEncode()]
    tfms = [x_tfms, y_tfms]
    dsets = Datasets(df, tfms, splits=splits)
    # dsets = torch.load('tmp/mimic3-9k_dsets.pkl')
    dl_type = partial(SortedDL, shuffle=True)
    dls_clas = dsets.dataloaders(bs=bs, seq_len=sl,
                             dl_type=dl_type,
                             before_batch=pad_input_chunk, num_workers=workers)
    return dls_clas

@call_parse
def main(
    data: Param("Filename of the raw data", str)="mimic3-9k",
    lr:    Param("base Learning rate", float)=1e-2,
    bs:    Param("Batch size", int)=16,
    epochs:Param("Number of epochs", str)="[10, 5, 5, 5, 10]",
    fp16:  Param("Use mixed precision training", store_true)=False,
    lm:    Param("Use Pretrained LM", store_true)=False,
    plant: Param("PLANT attention", store_true)=True,
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
    track_train: Param("Record training metrics", store_true)=False,
    wandblog: Param("Experiment tracking in wandb.ai", store_true)=False,
    log: Param("Log loss and metrics after each epoch", store_true)=False,
    workers:   Param("Number of workers", int)=None,
    save_model: Param("Save model on improvement after each epoch", store_true)=False,
    fname: Param("Save model file", str)="mimic-9k"
):
    "Training of mimic classifier."

    source = rank0_first(untar_xxx, XURLs.MIMIC3)
    source_l2r = rank0_first(untar_xxx, XURLs.MIMIC3_L2R)

    # make tmp directory to save and load models and dataloaders
    tmp = Path.cwd()/'tmp/models'
    tmp.mkdir(exist_ok=True, parents=True)
    tmp = tmp.parent
    files = 'mimic3-9k_lm_finetuned.pth mimic3-9k_tok_lbl_info.pkl p_L.pkl lin_lambdarank_full.pth mimic3-9k_lm_decoder.pth'.split(' ')
    for f in files:
        if not (tmp/'models'/f).exists():
            (tmp/'models'/f).symlink_to(source/f) 

    # loading dataloaders
    dls_file = join_path_file(data+'_dls_clas_'+str(bs), tmp, ext='.pkl')
    if dls_file.exists(): 
        dls_clas = torch.load(dls_file, map_location=torch.device('cpu'))
    else:
        dls_clas = get_dls(source, data, bs, workers=workers)
        torch.save(dls_clas, dls_file)

    epochs = json.loads(epochs)
    for run in range(runs):
        set_seed(1, reproducible=True)
        pr(f'Rank[{rank_distrib()}] Run: {run}; epochs: {sum(epochs)}; lr: {lr}; bs: {bs}')

        cbs = SaveModelCallback(monitor='valid_precision_at_k', fname=fname, with_opt=True, reset_on_fit=False) if save_model else None
        if log: 
            logfname = join_path_file(fname, tmp, ext='.csv')
            if logfname.exists(): logfname.unlink()
            cbs += L(CSVLogger(fname=logfname, append=True))
        if wandblog: cbs += L(WandbCallback(log_preds=False, log_model=True, model_name=fname))
        learn = rank0_first(xmltext_classifier_learner, dls_clas, AWD_LSTM, drop_mult=0.1, max_len=72*40,
                                   metrics=partial(precision_at_k, k=15), path=tmp, cbs=cbs,
                                   pretrained=False,
                                   splitter=None,
                                   running_decoder=True,
                                   )
        if track_train: 
            assert learn.cbs[1].__class__ is Recorder
            setattr(learn.cbs[1], 'train_metrics', True)

        if dump: pr(learn.model); exit()
        if fp16: learn = learn.to_fp16()
        if lm: learn = rank0_first(learn.load_encoder, 'mimic3-9k_lm_finetuned')
        if plant: 
            learn = rank0_first(learn.load_both, 'mimic3-9k_tok_lbl_info', 'p_L', 'lin_lambdarank_full', 'mimic3-9k_lm_decoder')
            setattr(learn, 'splitter', awd_lstm_xclas_split)
            learn.create_opt()

        # Workaround: In PyTorch 2.0.1 need to set DistributedDataParallel() with find_unused_parameters=True,
        # to avoid a crash that only happens in distributed mode of xmltext_clasifier_learner.fit()
        ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
        cms = learn.distrib_ctx(kwargs_handlers=[ddp_scaler])
        if wandblog: cms += L(wandb.init())
        with ContextManagers(cms):
            print("unfreezing the last layer and pretrained l2r...")
            learn.freeze_to(-2) # unfreeze the clas decoder and the l2r
            learn.fit(epochs[0], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-3, 3e-2], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])

            print("unfreezing the LM decoder...")
            learn.freeze_to(-3) # unfreeze the lm decoder
            learn.fit(epochs[1], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-3, 1e-2], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])

            print("unfreezing one LSTM...")
            learn.freeze_to(-4) # unfreeze one LSTM
            learn.fit(epochs[2], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-3, 1e-2], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])

            print("unfreezing one more LSTM...")
            learn.freeze_to(-5) # unfreeze one more LSTM
            learn.fit(epochs[3], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-3, 1e-2], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])

            print("unfreezing the entire model...")
            learn.unfreeze() # unfreeze the rest
            learn.fit(epochs[4], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6], wd=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])



