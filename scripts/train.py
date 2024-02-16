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

@patch
def after_pred(self: RNNCallback): 
    "Save the raw and dropped-out outputs and only keep the true output for loss computation"
    self.learn.pred,self.raw_out,self.out, _, self.learn.loss_lm = [o[-1] if is_listy(o) else o for o in self.pred]

class AddLMLossCallback(Callback):
    order=1000
    def after_loss(self):
        self.learn.loss_grad.add_(self.learn.loss_lm)

class TestCallback(Callback):
    order = 2000 

    def before_backward(self):
        import pdb; pdb.set_trace()
    def after_backward(self):
        import pdb; pdb.set_trace()    
    def before_step(self): pass
        # import pdb; pdb.set_trace()
    def after_step(self): pass
        # import pdb; pdb.set_trace()

class RarePrecisionCallback(Callback):
    order=Recorder.order-1
    def __init__(self, rare_codes_fname):
        self.rare_codes = load_pickle(rare_codes_fname)
    def before_validate(self):
        # import pdb; pdb.set_trace()
        rare_idxs = mapt(self.dls.vocab[1].o2i.get, self.rare_codes)
        rare_prec = partial(rareprecision_at_k, rare_idxs=rare_idxs)
        self.learn.metrics += mk_metric(rare_prec)
    # def after_batch(self):
    #     if self.training: return
    #     import pdb; pdb.set_trace()

class ShortEpochCallback(Callback):
    "Fit just `pct` of an epoch, then stop"
    order=Recorder.order+1
    def __init__(self,pct=0.01,short_valid=True): self.pct,self.short_valid = pct,short_valid
    def after_batch(self):
        if self.iter/self.n_iter < self.pct: return
        if self.training:    raise CancelTrainException
        if self.short_valid: raise CancelValidException
    def after_cancel_train(self):
        if getattr(self.recorder, 'cancel_train', True):
            setattr(self.recorder, 'cancel_train', False)
    # def after_cancel_validate(self):
    #     import pdb; pdb.set_trace()
    #     if getattr(self.recorder, 'cancel_valid', True):
    #         setattr(self.recorder, 'cancel_valid', False)

def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].to_list()
    return train, valid

def get_dls(source, data, bs, sl=16, workers=None, lm_vocab_file='mimic3-9k_dls_lm_vocab.pkl', bwd=False):
    workers = ifnone(workers,min(8,num_cpus()))
    data = join_path_file(data, source, ext='.csv')
    if 'mimic3' in data.name.split('.')[0]: # mimic3
        df = pd.read_csv(data,
                    header=0,
                    names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid', 'split'],
                    dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool, 'split': str})
    else: # mimic4
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
    if bwd : x_tfms = x_tfms + [reverse_text] 
    y_tfms = [ColReader('labels', label_delim=';'), MultiCategorize(vocab=lbls), OneHotEncode()]
    tfms = [x_tfms, y_tfms]
    dsets = Datasets(df, tfms, splits=splits)
    dl_type = partial(SortedDL, shuffle=True)
    dls_clas = dsets.dataloaders(bs=bs, seq_len=sl,
                             dl_type=dl_type,
                             before_batch=pad_input_chunk, num_workers=workers)
    return dls_clas

# change dev_dl before using it
def get_dev_dl(source, data, bs, sl=16, workers=None, lm_vocab_file='mimic3-9k_dls_lm_vocab.pkl', bwd=False):
    workers = ifnone(workers,min(8,num_cpus()))
    data = join_path_file(data, source, ext='.csv')
    if 'mimic3' in data.name.split('.')[0]: # mimic3
        df = pd.read_csv(data,
                    header=0,
                    names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid', 'split'],
                    dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool, 'split': str})
    else: # mimic4
        df = pd.read_csv(data,
                    header=0,
                    usecols=['subject_id', '_id', 'text', 'labels', 'num_targets', 'is_valid', 'split'],
                    dtype={'subject_id': str, '_id': str, 'text': str, 'labels': str, 'num_targets': np.int64, 'is_valid': bool, 'split': str})
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)

    # pdb.set_trace()
    lbl_freqs = Counter()
    for labels in df.labels: lbl_freqs.update(labels.split(';'))
    lbls = list(lbl_freqs.keys())
    splits = splitter(df)
    lm_vocab = torch.load(source/lm_vocab_file)
    x_tfms = [Tokenizer.from_df('text', n_workers=workers), attrgetter("text"), Numericalize(vocab=lm_vocab)]
    y_tfms = [ColReader('labels', label_delim=';'), MultiCategorize(vocab=lbls), OneHotEncode()]
    tfms = [x_tfms, y_tfms]
    val_split_name = 'val' if 'val' in df['split'].unique() else 'dev'
    if val_split_name not in ('val', 'dev'): raise ValueError("The split field of the dataframe doesnot contain 'val' or 'dev'")
    dev_dset = Datasets(df[df['split']==val_split_name], tfms)
    dl_type = partial(SortedDL, shuffle=True)
    dev_dl = TfmdDL(dev_dset, bs=bs, seq_len=sl,
                             dl_type=dl_type,
                             before_batch=pad_input_chunk, num_workers=workers, device=default_device())
    return dev_dl

def train_linear_attn(learn, epochs, lrs, lrs_sgdr, wd_linattn, fit_sgdr=False, sgdr_n_cycles=4):
    
    ic(lrs_sgdr)
    if epochs[0] or epochs[1]:
        print("unfreezing the last layer...")
        if fit_sgdr: learn.fit_sgdr(sgdr_n_cycles, 1, lr_max=lrs_sgdr[0][0], wd=wd_linattn[0])
        else:  learn.fit(epochs[0]+epochs[1], lr=lrs[0][0])

    if epochs[2]:
        print("unfreezing one LSTM...")
        learn.freeze_to(-2)
        learn.fit(epochs[2], lr=lrs[2][0], wd=wd_linattn[1])

    if epochs[3]:
        print("unfreezing one more LSTM...")
        learn.freeze_to(-3)
        learn.fit(epochs[3], lr=lrs[3][0], wd=wd_linattn[2])

    if epochs[4]:
        print("unfreezing the entire model...")
        learn.unfreeze()
        learn.fit(epochs[4], lr=lrs[4][0], wd=wd_linattn[3])

    print("Done!!!")
    # print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

def train_plant(learn, epochs, lrs, lrs_sgdr, wd_plant, wd_mul_plant, fit_sgdr=False, unfreeze_l2r=False,sgdr_n_cycles=4):
    if epochs[0]: # unfreeze the clas decoder and the l2r
        print("unfreezing the last layer and potentially the pretrained l2r...")
        learn.freeze_to(-2 if unfreeze_l2r else -1) 
        # learn.fit_sgdr(4, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-3, 0.2], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01]) #top
        ic(f"classification layer: {learn.opt.param_lists[-1].attrgot('requires_grad')}")
        ic(f"pretrained l2r layer: {learn.opt.param_lists[-2].attrgot('requires_grad')}")
        ic(f"pretrained l2r layer: {learn.opt.param_lists[-2].attrgot('shape')}")
        ic(f"lm decoder layer: {learn.opt.param_lists[-3].attrgot('requires_grad')}")
        ic(lrs_sgdr)
        if fit_sgdr: learn.fit_sgdr(sgdr_n_cycles, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs_sgdr[0][1], lrs_sgdr[0][0]], wd=wd_mul_plant[0]*array(wd_plant), ) #rare
        else: learn.fit(epochs[0], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[0][1], lrs[0][0]], wd=wd_mul_plant[0]*array(wd_plant))
        # learn.fit_sgdr(4, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-2, 0.6], wd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01]) #tiny
        # print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")
        # print(learn.opt.hypers)

    if epochs[1]: # unfreeze the lm decoder
        print("unfreezing upto the LM decoder...")
        learn.freeze_to(-3) 
        ic(f"classification layer: {learn.opt.param_lists[-1].attrgot('requires_grad')}")
        ic(f"pretrained l2r layer: {learn.opt.param_lists[-2].attrgot('requires_grad')}")
        ic(f"lm decoder layer: {learn.opt.param_lists[-3].attrgot('requires_grad')}")
        ic(lrs_sgdr)
        if fit_sgdr: learn.fit_sgdr(sgdr_n_cycles, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 8e-2, lrs_sgdr[1][1], lrs_sgdr[1][0]], wd=wd_mul_plant[1]*array(wd_plant)) # changed lmdecoder lr
        else: learn.fit(epochs[1], lr=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[1][1], lrs[1][0]], wd=wd_mul_plant[1]*array(wd_plant))
        print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

    if epochs[2]: # unfreeze one LSTM
        print("unfreezing one LSTM...")
        learn.freeze_to(-4) 
        ic(f"classification layer: {learn.opt.param_lists[-1].attrgot('requires_grad')}")
        ic(f"pretrained l2r layer: {learn.opt.param_lists[-2].attrgot('requires_grad')}")
        ic(f"lm decoder layer: {learn.opt.param_lists[-3].attrgot('requires_grad')}")
        ic(f"one LSTM layer: {learn.opt.param_lists[-4].attrgot('requires_grad')}")
        ic(lrs_sgdr)
        ic(lrs)
        # if fit_sgdr: learn.fit_sgdr(sgdr_n_cycles, 1, lr_max=[1e-6, 1e-6, 1e-6, lstm_lr, 8e-2, lrs_sgdr[2][1], lrs_sgdr[2][0]], wd=wd_mul_plant[2]*array(wd_plant))
        learn.fit(epochs[2], lr=[1e-6, 1e-6, 1e-6, 1e-2, 1e-7, lrs[2][1], lrs[2][0]], wd=wd_mul_plant[2]*array(wd_plant))
        # learn.fit_sgdr(sgdr_n_cycles, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs[2][1], 0.15], wd=wd_mul_plant[2]*array(wd_plant))
        # print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

    if epochs[3]: # unfreeze one more LSTM
        print("unfreezing one more LSTM...")
        learn.freeze_to(-5) 
        ic(f"classification layer: {learn.opt.param_lists[-1].attrgot('requires_grad')}")
        ic(f"pretrained l2r layer: {learn.opt.param_lists[-2].attrgot('requires_grad')}")
        ic(f"lm decoder layer: {learn.opt.param_lists[-3].attrgot('requires_grad')}")
        ic(f"one LSTM layer: {learn.opt.param_lists[-4].attrgot('requires_grad')}")
        ic(f"one LSTM layer: {learn.opt.param_lists[-5].attrgot('requires_grad')}")
        ic(lrs_sgdr, lrs)
        learn.fit(epochs[3], lr=[1e-6, 1e-6, 1e-2, 1e-2, 1e-6, lrs[3][1], lrs[3][0]], wd=wd_mul_plant[3]*array(wd_plant))
        # learn.fit_sgdr(sgdr_n_cycles, 1, lr_max=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, lrs_sgdr[3][1], lrs_sgdr[3][0]], wd=[0.3]*7)
        # print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

    if epochs[4]: # unfreeze the rest
        print("unfreezing the entire model...")
        learn.unfreeze() 
        ic(lrs_sgdr, lrs)
        # wd_plant = [0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01]
        learn.fit(epochs[4], lr=[1e-5, 1e-5, 1e-5, 1e-5, 1e-6, lrs[4][1], lrs[4][0]], wd=array(wd_plant))
        

    print("Done!!!")
    # print(f"lin_wt = {learn.model[1].pay_attn.wgts[0]}, plant_wt = {learn.model[1].pay_attn.wgts[1]}, splant_wt = {learn.model[1].pay_attn.wgts[2]}")

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
    def before_validate(self): 
        import pdb; pdb.set_trace()
        raise CancelValidException()

def _print_metrics(vals, learn):
    print(f"test_loss = {vals[0]}")
    for m,v in zip(learn.metrics[:4], vals[1:5]):
        if not isinstance(m.func, partial): raise AssertionError 
        func = m.func.func
        name = '_'.join(L(m.name.split('_')[:-1] + [m.func.keywords['k']]).map(str))
        print(f"{name} = {v}")
    for m,v in zip(learn.metrics[4:], vals[5:]):
        name = m.name + '_' + m.kwargs['average']
        print(f"{name} = {v}")
        

@call_parse
def main(
    source_url: Param("Source url", str)="XURLs.MIMIC3",
    source_url_l2r: Param("Source url", str)="XURLs.MIMIC3_L2R",
    data:  Param("Filename of the raw data", str)="mimic3-9k",
    rarecodes_fname: Param("Filename of the rare codes", str)="xxx",
    lr:    Param("base Learning rate", float)=1e-2,
    bs:    Param("Batch size", int)=16,
    epochs:Param("Number of epochs", str)="[10, 5, 5, 5, 10]",
    lrs_linattn:   Param("Learning rates for gradual unfreezing of the layers in linear attention", str)="[(3e-2,1e-3), (1e-2,1e-3), (1e-2, 1e-3), (1e-2,1e-3), (1e-6,1e-6)]",
    lrs_plant:   Param("Learning rates of the last layer and lm decoder for gradual unfreezing in plant", str)="[(3e-2,1e-3), (1e-2,1e-3), (1e-2, 1e-3), (1e-2,1e-3), (1e-6,1e-6)]",
    lrs_sgdr_linattn:   Param("Learning rates for gradual unfreezing of the layers in linear attention with sgd", str)="[(3e-2,1e-3), (1e-2,1e-3), (1e-2, 1e-3), (1e-2,1e-3), (1e-6,1e-6)]",
    lrs_sgdr_plant:   Param("Learning rates of the last layer and lm decoder for gradual unfreezing in plant with sgd", str)="[(3e-2,1e-3), (1e-2,1e-3), (1e-2, 1e-3), (1e-2,1e-3), (1e-6,1e-6)]",
    wd_linattn:Param("Weight decays for the gradual unfreezing", str)="[0.01, 0.01, 0.01, 0.3]",
    wd_plant:Param("Discriminative weight decays", str)="[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01]",
    wd_mul_plant:Param("Multipliers for weight decays for the gradual unfreezing", str)="[1.0, 1.0, 1.0, 1.0, 30.0]",
    fp16:  Param("Use mixed precision training", store_true)=False,
    lm:    Param("Use Pretrained LM", store_true)=False,
    plant: Param("PLANT attention", bool_arg)=True,
    static_inattn:    Param("base Learning rate", int)=5,
    diff_inattn:    Param("base Learning rate", int)=30,
    fit_sgdr: Param("PLANT attention", store_true)=False,
    unfreeze_l2r: Param("Unfreeze L2R along with last layer while gradual unfreezing", store_true)=False,
    unfreeze_lm_decoder: Param("Unfreeze LM Decoder along with last layer while gradual unfreezing", store_true)=False,
    no_running_decoder: Param("Train XMTC model with stateful decoder", bool_arg)=True,
    sgdr_n_cycles:    Param("base Learning rate", int)=4,
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
    trn_frm_cpt: Param("Train from saved checkpoint", store_true)=False,
    bwd: Param("Train the bwd classifier", store_true)=False
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
    files_l2r = files_l2r.split(',')
    for f in files_l2r:
        if not (tmp/'models'/f).exists():
            (tmp/'models'/f).symlink_to(source_l2r/f) 
    # loading dataloaders
    dls_name = '_dls_clas_bwd_' if bwd else '_dls_clas_' 
    dls_file = join_path_file(data+dls_name+str(bs), tmp, ext='.pkl')
    if dls_file.exists(): 
        dls_clas = torch.load(dls_file, map_location=torch.device('cpu'))
    else:
        dls_clas = get_dls(source, data, bs, workers=workers, lm_vocab_file=files_lm[2], bwd=bwd)
        torch.save(dls_clas, dls_file)

    epochs = json.loads(epochs)
    lrs_linattn = [L(match.split(',')).map(float) for match in re.findall(r'\((.*?)\)', lrs_linattn)]
    lrs_plant = [L(match.split(',')).map(float) for match in re.findall(r'\((.*?)\)', lrs_plant)]
    lrs_sgdr_linattn = [L(match.split(',')).map(float) for match in re.findall(r'\((.*?)\)', lrs_sgdr_linattn)]
    lrs_sgdr_plant = [L(match.split(',')).map(float) for match in re.findall(r'\((.*?)\)', lrs_sgdr_plant)]
    wd_linattn = json.loads(wd_linattn)
    wd_plant = json.loads(wd_plant)
    wd_mul_plant = json.loads(wd_mul_plant)
    for run in range(runs):
        set_seed(1, reproducible=True)
        pr(f'Rank[{rank_distrib()}] Run: {run}; epochs: {sum(epochs)}; lr: {lr}; bs: {bs}')

        cbs = SaveModelCallback(monitor='valid_precision_at_k', fname=fname, with_opt=True, reset_on_fit=False) if save_model else None
        if not infer and log: 
            logfname = join_path_file(fname, tmp, ext='.csv')
            if not trn_frm_cpt and logfname.exists(): logfname.unlink() # don't delete if from training from chkpt
            cbs += L(CSVLogger(fname=logfname, append=True))
        if wandblog: cbs += L(WandbCallback(log_preds=False, log_model=True, model_name=fname))
        # cbs += L(ShortEpochCallback(pct=0.7, short_valid=False))
        cbs += L(AddLMLossCallback())
        # cbs += L(TestCallback())
        print(f"Training with running_decoder={not no_running_decoder}")
        learn = rank0_first(xmltext_classifier_learner, dls_clas, AWD_LSTM, drop_mult=0.1, max_len=72*40,
                                #    metrics=[partial(precision_at_k, k=15), F1ScoreMulti(thresh=0.5, average='macro')], path=tmp, cbs=cbs,
                                   metrics=[partial(precision_at_k, k=15)], path=tmp, cbs=cbs,
                                   pretrained=False,
                                   splitter=None,
                                   running_decoder=not no_running_decoder,
                                   attn_init=ast.literal_eval(attn_init),
                                   static_inattn=static_inattn,
                                   diff_inattn=diff_inattn,
                                   unfreeze_lm_decoder=unfreeze_lm_decoder
                                   )
        if track_train: 
            assert learn.cbs[1].__class__ is Recorder
            setattr(learn.cbs[1], 'train_metrics', true)

        if dump: pr(learn.model); exit()
        if fp16: learn = learn.to_fp16()
        # if lm: learn = rank0_first(learn.load_encoder, 'mimic3-9k_lm_finetuned')
        # import pdb; pdb.set_trace()
        if lm: learn = rank0_first(learn.load_encoder, files_lm[0].split('.')[0]) # change for bwd
        if plant: 
            # import IPython; IPython.embed()
            # 'tok_lbl_info', 'p_L', 'lin_lambdarank', 'lm_decoder'
            brain = L(*files_l2r, files_lm[1]).map(lambda o: o.split('.')[0])
            # learn = rank0_first(learn.load_both, 'mimic3-9k_tok_lbl_info', 'p_L', 'lin_lambdarank_full', 'mimic3-9k_lm_decoder')
            learn = rank0_first(learn.load_both, *brain)
            setattr(learn, 'splitter', awd_lstm_xclas_split)
            learn.create_opt()
            # import IPython; IPython.embed()
        if infer:
            # learn.add_cb(RarePrecisionCallback(join_path_file(rarecodes_fname, source, ext='.pkl')))
            # setattr(learn.model[0], 'max_len', None)
            learn.metrics = [eval(o) for o in metrics.split(';') if callable(eval(o))]
            # dev_dl = get_dev_dl(source, data, bs, workers=workers, lm_vocab_file=files_lm[2], bwd=bwd)
            try: 
                learn = learn.load(learn.save_model.fname)
                # validate(learn, dl=dev_dl)
                # pred, targ = learn.get_preds(dl=dev_dl) # dont comment  
                # now comment
                # xs = torch.linspace(0.05, 0.95, 30)
                # f1_macros = [compute_val(F1ScoreMulti(thresh=i, average='macro', sigmoid=False), pred, targ, bs=bs) for i in xs]
                # f1_micros =  [compute_val(F1ScoreMulti(thresh=i, average='micro', sigmoid=False), pred, targ, bs=bs) for i in xs]
                # thresh_macro = xs[f1_macros.index(max(f1_macros))]
                # thresh_micro = xs[f1_micros.index(max(f1_micros))]
                # learn.metrics += F1ScoreMulti(thresh=thresh_macro, average='macro')
                # learn.metrics += F1ScoreMulti(thresh=thresh_micro, average='micro')
                # now comment
                vals = validate(learn) # dont comment
                # _print_metrics(vals, learn) # dont comment
                # pred, targ = learn.get_preds()
                # print(auc_metrics(pred, targ))
                # print(compute_val(F1ScoreMulti(thresh=0.07, average='macro', sigmoid=False), pred, targ, bs=bs)) 
                # print(compute_val(F1ScoreMulti(thresh=0.07, average='micro', sigmoid=False), pred, targ, bs=bs))

            except FileNotFoundError as e: 
                print("Exception:", e)
                print("Trained model not found!")
            except Exception as e:
                print("Exception:", e)
                # import pdb; pdb.set_trace()
            finally: exit()
        if trn_frm_cpt:
            try:
                ic(learn.save_model.fname)
                ic(learn.save_model.reset_on_fit)
                assert learn.save_model.reset_on_fit is False
                learn = learn.load(learn.save_model.fname)
                print("Validating the checkpointed model so that we can run from where we left of...")
                # vals = validate(learn) # remove comment later
                # print(f"We are monitoring {learn.save_model.monitor}. Set the best so far = {vals[1]}") # remove comment later
                print(f"We are monitoring {learn.save_model.monitor}. Set the best so far = {0.5569504763828573}")
                # learn.save_model.best = vals[1] # remove comment later
                learn.save_model.best = 0.5569504763828573
            except FileNotFoundError as e: 
                print("Exception:", e)
                print("Checkpoint model not found!")

        # Workaround: In PyTorch 2.0.1 need to set DistributedDataParallel() with find_unused_parameters=True,
        # to avoid a crash that only happens in distributed mode of xmltext_clasifier_learner.fit()
        ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
        cms = learn.distrib_ctx(kwargs_handlers=[ddp_scaler])
        if wandblog: cms += L(wandb.init())
        with ContextManagers(cms):
            if plant: train_plant(learn, epochs, lrs_plant, lrs_sgdr_plant, wd_plant, wd_mul_plant, fit_sgdr=fit_sgdr, unfreeze_l2r=unfreeze_l2r, sgdr_n_cycles=sgdr_n_cycles)
            else: train_linear_attn(learn, epochs, lrs_linattn, lrs_sgdr_linattn, wd_linattn, fit_sgdr=fit_sgdr, sgdr_n_cycles=sgdr_n_cycles)

