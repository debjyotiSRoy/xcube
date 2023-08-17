from accelerate.utils import DistributedDataParallelKwargs
from fastcore.script import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.text.all import *
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

def get_dls(source, bs, sl=72, workers=None):
    workers = ifnone(workers,min(8,num_cpus()))
    data = source/'mimic3-9k.csv'
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
    lr:    Param("base Learning rate", float)=1e-2,
    bs:    Param("Batch size", int)=16,
    epochs:Param("Number of epochs", int)=1,
    fp16:  Param("Use mixed precision training", store_true)=False,
    lm:    Param("Use Pretrained LM", store_true)=False,
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
    track_train: Param("Record training metrics", store_true)=False,
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
    if not (tmp/'models'/'mimic3-9k_lm_finetuned.pth').exists():
        (tmp/'models'/'mimic3-9k_lm_finetuned.pth').symlink_to(source/'mimic3-9k_lm_finetuned.pth')

    # loading dataloaders
    dls_file = join_path_file('mimic3-9k_dls_clas_'+str(bs), tmp, ext='.pkl')
    if dls_file.exists(): 
        dls_clas = torch.load(dls_file, map_location=torch.device('cpu'))
    else:
        dls_clas = get_dls(source, bs, workers=workers)
        torch.save(dls_clas, dls_file)

    # import pdb; pdb.set_trace()
    for run in range(runs):
        set_seed(run, reproducible=True)
        pr(f'Rank[{rank_distrib()}] Run: {run}; epochs: {epochs}; lr: {lr}; bs: {bs}')

        cbs=SaveModelCallback(monitor='valid_precision_at_k', fname=fname, with_opt=True, reset_on_fit=True) if save_model else None
        learn = rank0_first(xmltext_classifier_learner, dls_clas, AWD_LSTM, drop_mult=0.1, max_len=72*40,
                                   metrics=partial(precision_at_k, k=15), path=tmp, cbs=cbs,
                                   pretrained=False,
                                   splitter=None,#awd_lstm_xclas_split,
                                   running_decoder=True,
                                   )
        if track_train: 
            assert learn.cbs[1].__class__ is Recorder
            setattr(learn.cbs[1], 'train_metrics', True)

        if dump: pr(learn.model); exit()
        if fp16: learn = learn.to_fp16()
        if lm: learn = rank0_first(learn.load_encoder, 'mimic3-9k_lm_finetuned')

        # Workaround: In PyTorch 2.0.1 need to set DistributedDataParallel() with find_unused_parameters=True,
        # to avoid a crash that only happens in distributed mode of xmltext_clasifier_learner.fit()
        ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
        # import pdb; pdb.set_trace()
        with learn.distrib_ctx(kwargs_handlers=[ddp_scaler]): # distributed traing requires "accelerate launch train.py --help"
            learn.fit(1, 3e-2)

