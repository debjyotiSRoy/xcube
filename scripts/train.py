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

def get_dls(source, data, bs, sl=16, workers=None, lm_vocab_file='mimic3-9k_dls_lm_vocab.pkl'):
    workers = ifnone(workers,min(8,num_cpus()))
    data = join_path_file(data, source, ext='.csv')
    # mimic3
    # df = pd.read_csv(data,
    #              header=0,
    #              names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid', 'split'],
    #              dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool, 'split': str})
    # mimic4
    df = pd.read_csv(data,
                 header=0,
                 usecols=['subject_id', '_id', 'text', 'labels', 'num_targets', 'is_valid', 'split'],
                 dtype={'subject_id': str, '_id': str, 'text': str, 'labels': str, 'num_targets': np.int64, 'is_valid': bool, 'split': str})
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)
    lbl_freqs = Counter()
    for labels in df.labels: lbl_freqs.update(labels.split(';'))
    lbls = list(lbl_freqs.keys())
    splits = splitter(df)
    # lm_vocab = torch.load(source/'mimic3-9k_dls_lm_vocab.pkl')
    # import pdb; pdb.set_trace()
    lm_vocab = torch.load(source/lm_vocab_file)
    x_tfms = [Tokenizer.from_df('text', n_workers=workers), attrgetter("text"), Numericalize(vocab=lm_vocab)]
    y_tfms = [ColReader('labels', label_delim=';'), MultiCategorize(vocab=lbls), OneHotEncode()]
    tfms = [x_tfms, y_tfms]
    dsets = Datasets(df, tfms, splits=splits)
    dl_type = partial(SortedDL, shuffle=True)
    dls_clas = dsets.dataloaders(bs=bs, seq_len=sl,
                             dl_type=dl_type,
                             before_batch=pad_input_chunk, num_workers=workers)
    return dls_clas

def get_dev_dl(source, data, bs, sl=16, workers=None):
    workers = ifnone(workers,min(8,num_cpus()))
    data = join_path_file(data, source, ext='.csv')
    df = pd.read_csv(data,
                 header=0,
                 names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid', 'split'],
                 dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool, 'split': str})
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)
    # pdb.set_trace()
    lbl_freqs = Counter()
    for labels in df.labels: lbl_freqs.update(labels.split(';'))
    lbls = list(lbl_freqs.keys())
    splits = splitter(df)
    lm_vocab = torch.load(source/'mimic3-9k_dls_lm_vocab.pkl')
    x_tfms = [Tokenizer.from_df('text', n_workers=workers), attrgetter("text"), Numericalize(vocab=lm_vocab)]
    y_tfms = [ColReader('labels', label_delim=';'), MultiCategorize(vocab=lbls), OneHotEncode()]
    tfms = [x_tfms, y_tfms]
    dev_dset = Datasets(df[df['split']=='dev'], tfms)
    dl_type = partial(SortedDL, shuffle=True)
    dev_dl = TfmdDL(dev_dset, bs=bs, seq_len=sl,
                             dl_type=dl_type,
                             before_batch=pad_input_chunk, num_workers=workers, device=default_device())
    return dev_dl

def train_linear_attn(learn, epochs, lrs, lrs_sgdr, wd_linattn, fit_sgdr=False):
    print("unfreezing the last layer...")
    if fit_sgdr: learn.fit_sgdr(4, 1, lr_max=lrs_sgdr[0][0], wd=wd_linattn[0])
    else:  learn.fit(epochs[0]+epochs[1], lr=lrs[0][0])

    print("unfreezing one LSTM...")
    learn.freeze_to(-2)
    learn.fit(epochs[2], lr=lrs[2][0], wd=wd_linattn[1])

    print("unfreezing one more LSTM...")
    learn.freeze_to(-3)
    learn.fit(epochs[3], lr=lrs[3][0], wd=wd_linattn[2])

    print("unfreezing the entire model...")
    learn.unfreeze()
    learn.fit(epochs[4], lr=lrs[4][0], wd=wd_linattn[3])

    print("Done!!!")
    print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

def train_plant(learn, epochs, lrs, lrs_sgdr, fit_sgdr=False):
    print("unfreezing the last layer and pretrained l2r...")
    learn.freeze_to(-2) # unfreeze the clas decoder and the l2r
    # learn.fit_sgdr(4, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-3, 0.2], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01]) #top
    wd=5
    if fit_sgdr: learn.fit_sgdr(4, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs_sgdr[0][1], lrs_sgdr[0][0]], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01]) #rare
    else: learn.fit(epochs[0], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[0][1], lrs[0][0]], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])
    # learn.fit_sgdr(4, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-2, 0.6], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01]) #tiny
    print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

    print("unfreezing the LM decoder...")
    learn.freeze_to(-3) # unfreeze the lm decoder
    # learn.fit(epochs[1], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[1][1], lrs[1][0]], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])
    learn.fit_sgdr(4, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[1][1], 0.15], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])
    print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

    print("unfreezing one LSTM...")
    learn.freeze_to(-4) # unfreeze one LSTM
    # learn.fit(epochs[2], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[2][1], lrs[2][0]], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])
    learn.fit_sgdr(4, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[2][1], 0.15], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])
    print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

    print("unfreezing one more LSTM...")
    learn.freeze_to(-5) # unfreeze one more LSTM
    # learn.fit(epochs[3], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[3][1], lrs[3][0]], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])
    learn.fit_sgdr(4, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[3][1], 0.15], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01])
    print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

    print("unfreezing the entire model...")
    learn.unfreeze() # unfreeze the rest
    learn.fit(epochs[4], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[4][1], lrs[4][0]], wd=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

    print("Done!!!")
    print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

@delegates()
class TstLearner(Learner):
    def __init__(self, dls=None, model=None, **kwargs): self.pred, self.xb, self.yb = None, None, None

def compute_val(met, pred, targ, bs=16):
    met.reset()
    learn = TstLearner()
    for learn.pred,learn.yb in zip(torch.split(pred, bs), torch.split(targ, bs)): met.accumulate(learn)
    return met.value

def compute_val2(met, dl, learn, pred, targ):
    learn.model.eval()
    met.reset()
    _tst_learn = TstLearner()
    pdb.set_trace()
    for xb,yb in dl:
        _tst_learn.yb = yb
        _tst_learn.pred, *_ = learn.model(xb)
        met.accumulate(_tst_learn)
    return met.value

class CancelValid(Callback):
    order = 100
    def before_validate(self): raise CancelValidException()

def _print_metrics(vals, learn):
    print(f"test_loss = {vals[0]}")
    for m,v in zip(learn.metrics[:3], vals[1:4]):
        if not isinstance(m.func, partial): raise AssertionError 
        func = m.func.func
        name = '_'.join(L(m.name.split('_')[:-1] + [m.func.keywords['k']]).map(str))
        print(f"{name} = {v}")
    for m,v in zip(learn.metrics[3:], vals[4:]):
        name = m.name + '_' + m.kwargs['average']
        print(f"{name} = {v}")
        

@call_parse
def main(
    source_url: Param("Source url", str)="XURLs.MIMIC3",
    source_url_l2r: Param("Source url", str)="XURLs.MIMIC3_L2R",
    data:  Param("Filename of the raw data", str)="mimic3-9k",
    lr:    Param("base Learning rate", float)=1e-2,
    bs:    Param("Batch size", int)=16,
    epochs:Param("Number of epochs", str)="[10, 5, 5, 5, 10]",
    lrs:   Param("lr of the last layer and lm decoder for gradual unfreezing", str)="[(3e-2,1e-3), (1e-2,1e-3), (1e-2, 1e-3), (1e-2,1e-3), (1e-6,1e-6)]",
    lrs_sgdr:   Param("lr of the last layer and lm decoder for gradual unfreezing", str)="[(3e-2,1e-3), (1e-2,1e-3), (1e-2, 1e-3), (1e-2,1e-3), (1e-6,1e-6)]",
    wd_linattn:Param("Weight decays for the gradual unfreezing", str)="[0.01, 0.01, 0.01, 0.3]",
    fp16:  Param("Use mixed precision training", store_true)=False,
    lm:    Param("Use Pretrained LM", store_true)=False,
    plant: Param("PLANT attention", bool_arg)=True,
    fit_sgdr: Param("PLANT attention", store_true)=False,
    attn_init: Param("Initial wgts for Linear, Diff. PLANT and Static PLANT", str)="(0, 0, 1)",
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
    track_train: Param("Record training metrics", store_true)=False,
    wandblog: Param("Experiment tracking in wandb.ai", store_true)=False,
    log: Param("Log loss and metrics after each epoch", store_true)=False,
    workers:   Param("Number of workers", int)=None,
    save_model: Param("Save model on improvement after each epoch", store_true)=False,
    root_dir: Param("Root dir for saving models", str)="..",
    fname: Param("Save model file", str)="mimic3-9k",
    infer: Param("Don't train, just validate", int)=0,
    metrics: Param("Metrics used in inference", str)="partial(precision_at_k, k=15)",
    files_lm: Param("MIMIC LM files (comma seperated fine-tuned lm, decoder, lm_vocab)", str)="mimic3-9k_lm_finetuned.pth,mimic3-9k_lm_decoder.pth,mimic3-9k_dls_lm_vocab.pkl",
    files_l2r: Param("MIMIC L2R files (comma seperated)", str)="mimic3-9k_tok_lbl_info.pkl,p_L.pkl,lin_lambdarank_full.pth",
    # model_path: Param("Model path for validation", str)="mimic3-9k"

):
    "Training of mimic classifier."

    source = rank0_first(untar_xxx, eval(source_url))
    source_l2r = rank0_first(untar_xxx, eval(source_url_l2r))

    # make tmp directory to save and load models and dataloaders
    # pdb.set_trace()
    tmp = Path(root_dir)/'tmp/models'
    tmp.mkdir(exist_ok=True, parents=True)
    tmp = tmp.parent
    # files_mimic = 'mimic3-9k_lm_finetuned.pth mimic3-9k_lm_decoder.pth'.split(' ')
    files_lm = files_lm.split(',')
    for f in files_lm:
        if not (tmp/'models'/f).exists():
            (tmp/'models'/f).symlink_to(source/f) 
    # files_mimic_l2r = 'mimic3-9k_tok_lbl_info.pkl p_L.pkl lin_lambdarank_full.pth'.split(' ')
    # for f in files_mimic_l2r:
    #     if not (tmp/'models'/f).exists():
    #         (tmp/'models'/f).symlink_to(source_l2r/f) 

    # loading dataloaders
    dls_file = join_path_file(data+'_dls_clas_'+str(bs), tmp, ext='.pkl')
    if dls_file.exists(): 
        dls_clas = torch.load(dls_file, map_location=torch.device('cpu'))
    else:
        dls_clas = get_dls(source, data, bs, workers=workers, lm_vocab_file=files_lm[2])
        torch.save(dls_clas, dls_file)

    epochs = json.loads(epochs)
    lrs = [L(match.split(',')).map(float) for match in re.findall(r'\((.*?)\)', lrs)]
    lrs_sgdr = [L(match.split(',')).map(float) for match in re.findall(r'\((.*?)\)', lrs_sgdr)]
    wd_linattn = json.loads(wd_linattn)
    for run in range(runs):
        set_seed(1, reproducible=True)
        pr(f'Rank[{rank_distrib()}] Run: {run}; epochs: {sum(epochs)}; lr: {lr}; bs: {bs}')

        cbs = SaveModelCallback(monitor='valid_precision_at_k', fname=fname, with_opt=True, reset_on_fit=False) if save_model else None
        if not infer and log: 
            logfname = join_path_file(fname, tmp, ext='.csv')
            if logfname.exists(): logfname.unlink()
            cbs += L(CSVLogger(fname=logfname, append=True))
        if wandblog: cbs += L(WandbCallback(log_preds=False, log_model=True, model_name=fname))
        learn = rank0_first(xmltext_classifier_learner, dls_clas, AWD_LSTM, drop_mult=0.1, max_len=72*40,
                                #    metrics=[partial(precision_at_k, k=15), F1ScoreMulti(thresh=0.5, average='macro')], path=tmp, cbs=cbs,
                                   metrics=[partial(precision_at_k, k=15)], path=tmp, cbs=cbs,
                                   pretrained=False,
                                   splitter=None,
                                   running_decoder=True,
                                   attn_init=ast.literal_eval(attn_init),
                                   )
        if track_train: 
            assert learn.cbs[1].__class__ is Recorder
            setattr(learn.cbs[1], 'train_metrics', true)

        if dump: pr(learn.model); exit()
        if fp16: learn = learn.to_fp16()
        # if lm: learn = rank0_first(learn.load_encoder, 'mimic3-9k_lm_finetuned')
        # import pdb; pdb.set_trace()
        if lm: learn = rank0_first(learn.load_encoder, files_lm[0].split('.')[0])
        if plant: 
            learn = rank0_first(learn.load_both, 'mimic3-9k_tok_lbl_info', 'p_L', 'lin_lambdarank_full', 'mimic3-9k_lm_decoder')
            setattr(learn, 'splitter', awd_lstm_xclas_split)
            learn.create_opt()
        if infer:
            learn.metrics = [eval(o) for o in metrics.split(';') if callable(eval(o))]
            dev_dl = get_dev_dl(source, data, bs, workers=workers)
            try: 
                learn = learn.load(learn.save_model.fname)
                # validate(learn, dl=dev_dl)
                pred, targ = learn.get_preds(dl=dev_dl)
                xs = torch.linspace(0.05, 0.95, 30)
                f1_macros = [compute_val(F1ScoreMulti(thresh=i, average='macro', sigmoid=False), pred, targ, bs=bs) for i in xs]
                f1_micros =  [compute_val(F1ScoreMulti(thresh=i, average='micro', sigmoid=False), pred, targ, bs=bs) for i in xs]
                thresh_macro = xs[f1_macros.index(max(f1_macros))]
                thresh_micro = xs[f1_micros.index(max(f1_micros))]
                learn.metrics += F1ScoreMulti(thresh=thresh_macro, average='macro')
                learn.metrics += F1ScoreMulti(thresh=thresh_micro, average='micro')
                vals = validate(learn)
                _print_metrics(vals, learn)
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
            if plant: train_plant(learn, epochs, lrs, lrs_sgdr, fit_sgdr=fit_sgdr)
            else: train_linear_attn(learn, epochs, lrs, lrs_sgdr, wd_linattn, fit_sgdr=fit_sgdr)

