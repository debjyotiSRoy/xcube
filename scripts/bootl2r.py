from fastcore.script import *
from fastai.distributed import *
from fastai.data.core import *
from xcube.l2r.all import *

@call_parse
def main(
    data: Param("Filename of the raw data", str)="mimic3-9k",
):
    "Bootstrapping a learning-to-rank model"

    source = rank0_first(untar_xxx, XURLs.MIMIC4_L2R)
    pdb.set_trace()