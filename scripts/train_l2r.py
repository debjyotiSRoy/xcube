from fastcore.script import *
from fastai.distributed import *
from fastprogress import fastprogress
from xcube.l2r.all import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_dls(source, data, workers=None):
    workers = ifnone(workers, min(8, num_cpus()))
    data = join_path_file(data, source, ext='.ft')
    df_l2r = pd.read_feather(data)
    df_l2r = df_l2r.drop(['bcx_mutual_info'], axis=1)
    pdl = PreLoadTrans(df_l2r, device=torch.device('cpu'))
    scored_toks = pdl.quantized_score()
    test_eqs(scored_toks.shape, 
        (df_l2r.label.nunique(), df_l2r.token.nunique(), 4), 
        (pdl.num_lbs, pdl.num_toks, 4))
    scored_toks, binned_toks, probs, is_valid, bin_size, bin_bds = pdl.train_val_split()
    val_sl = pdl.pad_split()
    test_eq(is_valid.sum(dim=-1).unique().item(), val_sl)
    print(f"{val_sl=}")
    import IPython; IPython.embed()
    return scored_toks

@call_parse
def main(
    source_url: Param("Source url", str)="XURLs.MIMIC4_L2R",
    data: Param("Bootstrapped data", str)="",
    root_dir: Param("Root dir for saving models", str)="..",
    workers:   Param("Number of workers", int)=None,
):
    "Training a learning-to-rank model."


    source = rank0_first(untar_xxx, eval(source_url))
    # make tmp directory to save and load models and dataloaders
    # pdb.set_trace()
    tmp = Path(root_dir)/'tmp/models'
    tmp.mkdir(exist_ok=True, parents=True)
    tmp = tmp.parent

    # loading dataloaders
    scored_toks = get_dls(source, data, workers=workers)