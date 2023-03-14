# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_layers.ipynb.

# %% auto 0
__all__ = ['LinBnFlatDrop', 'LinBnDrop', 'Embedding', 'XMLAttention']

# %% ../nbs/01_layers.ipynb 2
from fastai.imports import *
from fastai.torch_imports import *
from fastai.torch_core import *
from fastai.layers import *
from fastai.text.models.awdlstm import EmbeddingDropout, RNNDropout

# %% ../nbs/01_layers.ipynb 10
class LinBnFlatDrop(nn.Sequential):
    "Module grouping `BatchNorm1dFlat`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [BatchNorm1dFlat(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

# %% ../nbs/01_layers.ipynb 13
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

# %% ../nbs/01_layers.ipynb 19
class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    def __init__(self, ni, nf, std=0.01, **kwargs):
        super().__init__(ni, nf, **kwargs)
        trunc_normal_(self.weight.data, std=std)

# %% ../nbs/01_layers.ipynb 21
class XMLAttention(Module):
    "Compute label specific attention weights for each token in a sequence"
    def __init__(self, n_lbs, emb_sz, embed_p=0.0):
         store_attr('n_lbs,emb_sz,embed_p')
         self.lbs_weight = Embedding(n_lbs, emb_sz)
         # self.lbs_weight_dp = EmbeddingDropout(self.lbs_weight, embed_p)
         # self.lbs_weight.weight.data.normal_(0, 0.01)   
         # self.input_dp = RNNDropout(0.02)

    def forward(self, x):
        # x is the ouput of SentenceEncoder i.e., (bs, max_len tokens, nh)
        # lbs_emb = self.lbs_weight(torch.arange(self.n_lbs, device=x.device)) # pulling out the lbs embeddings
        lbs_emb = self.lbs_weight.weight
        # x_dp = self.input_dp(x)
        attn_wgts = F.softmax(x @ lbs_emb.transpose(0,1), dim=1) # lbl specific wts for each token (bs, max_len, n_lbs)
        return attn_wgts.transpose(1,2) @ x # for each lbl do a linear combi of all the tokens based on attn_wgts (bs, num_lbs, nh)
    
