from fastcore.script import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.data.core import *
from xcube.l2r.all import *

# extra imports <remove later>
import warnings; warnings.filterwarnings(action='ignore')
# end extra imports

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_info(source, data, bs=8, chnk_sz=200, workers=None):
    workers = ifnone(workers, min(8, num_cpus()))
    data = join_path_file(data, source, ext='.csv')
    df = pd.read_csv(data,
                 header=0,
                 usecols=['text', 'labels'],
                 dtype={'text': str, 'labels': str})
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)

    # sampling for quick iterations: remove later 
    bs = 8
    cut = len(df) - len(df)%bs
    df = df[:cut]
    len(df)

    _arr = np.arange(0, len(df), bs)
    # mask = (_arr > 4000) & (_arr < 5000)
    mask = (_arr > 500) & (_arr < 1000)
    _n = np.random.choice(_arr[mask], 1)
    df = df.sample(n=_n, random_state=89, ignore_index=True)
    len(df)
    # sampling for quick iterations: remove later 

    info = MutualInfoGain(df, bs=bs, chnk_sz=chnk_sz, lbs_desc=None) # provide lbs_desc if you have it
    return info

def test_nanegs(**kwargs):
    for k, v in kwargs.items():
        has_nans = v.isnan().all() # check for nans
        has_negs = not torch.where(v>=0, True, False).all()
        # if has_nans: raise Exception(f"{namestr(o, locals())[0]} has nans")
        # if has_negs: raise Exception(f"{namestr(o, locals())[0]} has negs")
        if has_nans: raise Exception(f"{k} has nans")
        if has_negs: raise Exception(f"{k} has negs")

def stat_transform(mut_infos, save_dir):
    orig_shape = mut_infos.shape
    mut_infos = mut_infos.cpu().numpy().reshape(-1)
    ic(mut_infos.min(), mut_infos.max(), mut_infos.mean())
    ic(skew(mut_infos))
    print("The mutual-info values are incredibly skewed. So we need to apply some transformation. Sometimes `mut_infos` might contain negs, we need to convert those to eps.")
    # np.where(mut_infos<0, 1, 0).sum() # or, better yet
    where_negs = mut_infos < 0
    ic(np.sum(where_negs))
    eps = np.float32(1e-20)
    mut_infos[where_negs] = eps
    test_eq(np.sum(mut_infos<0), 0)
    ic(np.min(mut_infos), np.max(mut_infos), np.mean(mut_infos));
    hist, bins, _ = plt.hist(mut_infos, bins=50)
    plt.savefig(save_dir/'mut_infos_hist.png')
    print("Applying box-cox transformation...")
    bcx_mut_infos, *_ = boxcox(mut_infos+eps)
    ic(np.min(bcx_mut_infos), np.max(bcx_mut_infos), np.mean(bcx_mut_infos), np.median(bcx_mut_infos))
    ic(np.isnan(bcx_mut_infos).sum(), np.isinf(bcx_mut_infos).sum(), np.isneginf(bcx_mut_infos).sum())
    ic(skew(bcx_mut_infos))
    hist, bins, _ = plt.hist(bcx_mut_infos, bins=50)
    plt.savefig(save_dir/'bcx_mut_infos_hist.png')
    return bcx_mut_infos.reshape(*orig_shape)


@call_parse
def main(
    source_url: Param("Source url", str)="XURLs.MIMIC4_L2R",
    data: Param("Filename of the raw data", str)="mimic4_icd10_full",
    root_dir: Param("Root dir for saving models", str)="..",
    workers:   Param("Number of workers", int)=None,
    bs:    Param("Batch size", int)=16,
    chnk_sz:    Param("Chunk size", int)=200,
):
    "Bootstrapping a learning-to-rank model"

    source = rank0_first(untar_xxx, eval(source_url))

    # make tmp directory to save and load models and dataloaders
    # pdb.set_trace()
    tmp = Path(root_dir)/'tmp/models'
    tmp.mkdir(exist_ok=True, parents=True)
    tmp = tmp.parent

    info = get_info(source, data, bs=bs, chnk_sz=chnk_sz, workers=None)
    dsets = info.onehotify() # test
    toks, lbs = dsets.vocab # test
    x, y = dsets[0] # test
    test_eq(tensor(dsets.tfms[1][2].decode(y)), torch.where(y==1)[0]) # test
    test_eq(tensor(dsets.tfms[0][-1].decode(x)), torch.where(x==1)[0])
    dls = info.lbs_chunked()
    assert isinstance(dls[0], TfmdDL) # test
    test_eq(len(dls),  np.ceil(len(lbs)/chnk_sz)) # test
    test_eq(len(dls[0]), np.ceil(len(dsets)/bs)) # drop_last is False # test
    # test to prove that the labels for each data point is split across multiple dataloaders # test
    lbs_0 = torch.cat([yb[0] for dl in dls for _,yb in itertools.islice(dl, 1)]) # test
    y = y.to(default_device()) # test
    test_eq(lbs_0, y) # test
    p_TL = info.joint_pmf()
    test_eq(p_TL.shape, (info.toksize, info.lblsize, 2, 2)) # test
    p_T, p_L, p_TxL, H_T, H_L, I_TL = info.compute()
    test_eq(p_TL.shape, (info.toksize, info.lblsize, 2, 2))
    test_eq(p_T.shape, (info.toksize, 2, 1))
    test_eq(p_L.shape, (info.lblsize, 1, 2))
    test_eq(p_TxL.shape, (info.toksize, info.lblsize, 2, 2))
    test_eq(H_T.shape, [info.toksize])
    test_eq(H_L.shape, [info.lblsize])
    test_eq(I_TL.shape, (info.toksize, info.lblsize))
    howmany = torch.where(I_TL < 0, True, False).sum().item()
    negs = torch.where(I_TL < 0, I_TL, I_TL.new_zeros(I_TL.shape))
    print("Avg value of negs:", negs.sum()/howmany)
    print("Those negs on an avg are pretty close to zero. So we need not worry. Let's roll!")
    test_fail(test_nanegs, kwargs=dict(p_T=p_T, p_L=p_L, p_TxL=p_TxL, H_T=H_T, H_L=H_L, I_TL=I_TL), contains='I_TL has negs')
    torch.topk(I_TL.flatten(), 10, largest=False)
    torch.save(p_TL, tmp/'p_TL.pkl')
    torch.save((p_T, p_L, p_TxL, H_T, H_L, I_TL), tmp/'info.pkl')
    eps = I_TL.new_empty(1).fill_(1e-15)
    info_lbl_entropy = I_TL/(H_L + eps)
    info_jaccard = I_TL/(H_T.unsqueeze(-1) + H_L.unsqueeze(0) - I_TL + eps)
    assert not info_lbl_entropy.isnan().all(); assert not info_jaccard.isnan().all()
    l2r_bootstrap = {'toks': toks, 'lbs': lbs, 'mut_info_lbl_entropy': info_lbl_entropy, 'mutual_info_jaccard': info_jaccard}
    fname = '_'.join(data.split('_')[:-1])
    torch.save(l2r_bootstrap, join_path_file(fname +'_tok_lbl_info', tmp, ext='.pkl'))

    # getting ready
    info = l2r_bootstrap.get('mutual_info_jaccard', None)
    test_eq(info.shape, (len(toks), len(lbs)))
    bcx_mut_infos = torch.from_numpy(stat_transform(info, tmp)).to(default_device())
    ranked = bcx_mut_infos.argsort(descending=True, dim=0).argsort(dim=0)
    info_ranked = torch.stack((bcx_mut_infos, ranked), dim=2).flatten(start_dim=1)
    cols = pd.MultiIndex.from_product([range(len(lbs)), ['bcx_mutual_info', 'rank']], names=['label', 'key2'])
    df_l2r = pd.DataFrame(info_ranked, index=range(len(toks)), columns=cols)
    df_l2r.index.name='token'
    df_l2r = df_l2r.stack(level=0).reset_index().rename_axis(None, axis=1)
    df_l2r[['token', 'label']] = df_l2r[['token', 'label']].astype(np.int32) 
    test_eq(len(df_l2r), len(toks) * len(lbs))
    df_toks = pd.DataFrame([(i, w) for i,w in enumerate(toks)], columns=['token', 'tok_val'])
    df_lbs = pd.DataFrame([(i,w) for i, w in enumerate(lbs)], columns=['lbl', 'lbl_val'])
    df_toks.to_feather(join_path_file(fname + '_tok', tmp, ext='.ft'))
    df_lbs.to_feather(join_path_file(fname + '_lbl', tmp, ext='.ft'))
    df_l2r.to_feather(join_path_file(fname + '_tok_lbl', tmp, ext='.ft'))