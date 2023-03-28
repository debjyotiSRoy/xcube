# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_layers.ipynb.

# %% auto 0
__all__ = ['ElemWiseLin', 'LinBnFlatDrop', 'LinBnDrop', 'Embedding', 'Linear_Attention', 'Ranked_Attention', 'lincomb', 'topkmax',
           'XMLAttention']

# %% ../nbs/01_layers.ipynb 2
from fastai.imports import *
from fastai.torch_imports import *
from fastai.torch_core import *
from fastai.layers import *
from fastai.text.models.awdlstm import EmbeddingDropout, RNNDropout

from .utils import *

# %% ../nbs/01_layers.ipynb 13
def _create_bias(size, with_zeros=False):
    if with_zeros: return nn.Parameter(torch.zeros(*size))
    return nn.Parameter(torch.zeros(*size).uniform_(-0.1, 0.1))

# %% ../nbs/01_layers.ipynb 16
class ElemWiseLin(Module):
    initrange=0.1
    def __init__(self, dim0, dim1, add_bias=False, **kwargs):
        store_attr()
        self.lin = nn.Linear(dim1, dim0, **kwargs)
        # init_default(self.lin, func=partial(torch.nn.init.uniform_, a=-self.initrange, b=self.initrange))
        init_default(self.lin)
        if self.add_bias: self.bias = _create_bias((1, ))
        
    def forward(self, x):
        res = torch.addcmul(self.bias if self.add_bias else x.new_zeros(1), x, self.lin.weight)# * self.lin.weight
        return res #+ self.bias if self.add_bias else res

# %% ../nbs/01_layers.ipynb 19
class LinBnFlatDrop(nn.Sequential):
    "Module grouping `BatchNorm1dFlat`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [BatchNorm1dFlat(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

# %% ../nbs/01_layers.ipynb 20
class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out=None, bn=True, ln=True, p=0., act=None, lin_first=False, ndim=1):
        if not ln and lin_first: raise Exception(AssertionError)
        layers = [BatchNorm(n_out if ln and lin_first else n_in, ndim=ndim)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)] if ln else []
        if ln and act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

# %% ../nbs/01_layers.ipynb 26
class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    def __init__(self, ni, nf, std=0.01, **kwargs):
        super().__init__(ni, nf, **kwargs)
        trunc_normal_(self.weight.data, std=std)

# %% ../nbs/01_layers.ipynb 28
def _linear_attention(sentc:Tensor, # Sentence typically `(bs, bptt, nh)`
                   based_on: nn.Embedding|Module # xcube's `Embedding(n_lbs, nh)` layer holding the label embeddings or a full fledged model
                  ):
    return sentc @ based_on.weight.transpose(0,1)

# %% ../nbs/01_layers.ipynb 30
class _Pay_Attention:
    def __init__(self, f, based_on): store_attr('f,based_on')
    def __call__(self, sentc): return self.f(sentc, self.based_on)

# %% ../nbs/01_layers.ipynb 31
def Linear_Attention(based_on: Module): return _Pay_Attention(_linear_attention, based_on)

# %% ../nbs/01_layers.ipynb 32
def Ranked_Attention(based_on: Module):
    # TODO: Deb Create an architecture same as the Learning2Rank Model here, so that we can preload it just like fastai preloads LM encoder during text classification.
    pass

# %% ../nbs/01_layers.ipynb 34
def lincomb(t, wgts=None):
    "returns the linear combination of the dim1 of a 3d tensor of `t` based on `wgts` (if `wgts` is `None` just adds the rows)"
    if wgts is None: wgts = t.new_ones(t.size(0), 1, t.size(1))
    return torch.bmm(wgts, t) # wgts@t

# %% ../nbs/01_layers.ipynb 36
@torch.no_grad()
def topkmax(x, k=None, dim=1):
    """
    returns softmax of the 1th dim of 3d tensor x after zeroing out values in x smaller than `k`th largest.
    If k is `None` behaves like `x.softmax(dim=dim). Intuitively, `topkmax` hedges more compared to `F.softmax``
    """
    if dim!=1: raise NotImplementedError
    k = min(k if k is not None else np.inf, x.size(dim)-1)
    kth_largest = x.sort(dim=dim, descending=True).values[:,k,:][:,None,:].repeat(1, x.size(dim), 1)
    x[x < kth_largest] = 0.
    return x.softmax(dim=1)

# %% ../nbs/01_layers.ipynb 39
class XMLAttention(Module):
    "Compute label specific attention weights for each token in a sequence"
    def __init__(self, n_lbs, emb_sz, embed_p=0.0):
        store_attr('n_lbs,emb_sz,embed_p')
        self.lbs = Embedding(n_lbs, emb_sz)
        # self.lbs_weight_dp = EmbeddingDropout(self.lbs_weight, embed_p)
        self.LinAttn = Lambda(Linear_Attention(self.lbs))

    def forward(self, sentc, mask):
        # sent is the ouput of SentenceEncoder i.e., (bs, max_len tokens, nh)
        attn_wgts = F.softmax(self.LinAttn(sentc), dim=1).masked_fill(mask[:,:,None], 0) # lbl specific wts for each token (bs, max_len, n_lbs)
        return lincomb(sentc, wgts=attn_wgts.transpose(1,2)), attn_wgts # for each lbl do a linear combi of all the tokens based on attn_wgts (bs, num_lbs, nh)
