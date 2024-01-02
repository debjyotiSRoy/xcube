# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_layers.ipynb.

# %% ../nbs/01_layers.ipynb 2
from __future__ import annotations
from typing import Union
from fastai.imports import *
from fastai.torch_imports import *
from fastai.torch_core import *
from fastai.layers import *
from fastai.text.models.awdlstm import EmbeddingDropout, RNNDropout

from .utils import *

# %% auto 0
__all__ = ['ElemWiseLin', 'LinBnFlatDrop', 'LinBnDrop', 'Embedding', 'Linear_Attention', 'Planted_Attention', 'PlantedLMDecoder',
           'Diffntble_Planted_Attention', 'lincomb', 'split_sort', 'XMLAttention']

# %% ../nbs/01_layers.ipynb 13
def _create_bias(size, with_zeros=False):
    if with_zeros: return nn.Parameter(torch.zeros(*size))
    return nn.Parameter(torch.zeros(*size).uniform_(-0.1, 0.1))

# %% ../nbs/01_layers.ipynb 14
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

# %% ../nbs/01_layers.ipynb 17
class LinBnFlatDrop(nn.Sequential):
    "Module grouping `BatchNorm1dFlat`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [BatchNorm1dFlat(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

# %% ../nbs/01_layers.ipynb 18
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

# %% ../nbs/01_layers.ipynb 24
class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    def __init__(self, ni, nf, std=0.01, **kwargs):
        super().__init__(ni, nf, **kwargs)
        trunc_normal_(self.weight.data, std=std)

# %% ../nbs/01_layers.ipynb 26
def _linear_attention(sentc:Tensor, # Sentence typically `(bs, bptt, nh)` output of `SentenceEncoder`
                      based_on: nn.Embedding|Module # xcube's `Embedding(n_lbs, nh)` layer holding the label embeddings or a full fledged model
                  ):
    return sentc @ based_on.weight.transpose(0,1)

# %% ../nbs/01_layers.ipynb 28
def _planted_attention(sentc: Tensor, # Sentence typically `(bs, bptt)` containing the vocab idxs that goes inside the encoder
                       brain: Tensor # label specific attn wgts for each token in vocab, typically of shape `(vocab_sz, n_lbs)`
                     ):
    return brain[sentc.long()]

# %% ../nbs/01_layers.ipynb 30
def _diffntble_planted_attention(sentc_dec: Tensor, # Sentence `(bs, bptt)` typically containing the vocab idxs obtained after decoding what comes out of the encoder
                         l2r: nn.ModuleDict # containing `nn.Embedding` for `token_factors`, `token_bias`, `label_factors` and `label_bias` from pretrained L2R model
                        ):
    
    return l2r['token_factors'](sentc_dec.long()) @ l2r['label_factors'].weight.T + l2r['token_bias'](sentc_dec.long()) + l2r['label_bias'].weight.T

# %% ../nbs/01_layers.ipynb 32
class _Pay_Attention:
    def __init__(self, f, based_on): store_attr('f,based_on')
    def __call__(self, sentc): return self.f(sentc, self.based_on)

# %% ../nbs/01_layers.ipynb 33
def Linear_Attention(based_on: Module): return _Pay_Attention(_linear_attention, based_on)

# %% ../nbs/01_layers.ipynb 35
def Planted_Attention(brain: Tensor): return _Pay_Attention(_planted_attention, brain)

# %% ../nbs/01_layers.ipynb 37
class PlantedLMDecoder(Module):
    def __init__(self, 
        n_out:int, # vocab_sz 
        n_hid:int, # Number of features in encoder last layer output
        output_p:float=0.1, # Input dropout probability
        plant_wgts:dict=None, # If supplied loads `plant_wgts` into decoder
        bias:bool=True # If `False` the layer will not learn additive bias
    ):
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.output_dp = RNNDropout(output_p)
        if plant_wgts: self.load_state_dict(plant_wgts)

    def forward(self, input):
        dp_inp = self.output_dp(input)
        return self.decoder(dp_inp)
        # return self.decoder(dp_inp).softmax(dim=-1).argmax(dim=-1)

# %% ../nbs/01_layers.ipynb 40
def Diffntble_Planted_Attention(l2r: nn.ModuleDict): return _Pay_Attention(_diffntble_planted_attention, l2r)

# %% ../nbs/01_layers.ipynb 42
def lincomb(t, wgts=None):
    "returns the linear combination of the dim1 of a 3d tensor of `t` based on `wgts` (if `wgts` is `None` just adds the rows)"
    if wgts is None: wgts = t.new_ones(t.size(0), 1, t.size(1))
    return torch.bmm(wgts, t) # wgts@t

# %% ../nbs/01_layers.ipynb 44
@patch
@torch.no_grad()
def topkmax(self:Tensor, k=None, dim=1):
    """
    returns softmax of the 1th dim of 3d tensor x after zeroing out values in x smaller than `k`th largest.
    If k is `None` behaves like `x.softmax(dim=dim). Intuitively, `topkmax` hedges more compared to `F.softmax``
    """
    if dim!=1: raise NotImplementedError
    k = min(k if k is not None else np.inf, self.size(dim)-1)
    kth_largest = self.sort(dim=dim, descending=True).values[:,k,:][:,None,:].repeat(1, self.size(dim), 1)
    self[self < kth_largest] = 0.
    return self.softmax(dim=1)

# %% ../nbs/01_layers.ipynb 45
def split_sort(t, sp_dim, sort_dim, sp_sz=500, **kwargs):
    if t.ndim==1: return t.sort(dim=sort_dim, **kwargs).values
    return torch.cat([s.sort(dim=sort_dim, **kwargs).values for s in torch.split(t, split_size_or_sections=sp_sz, dim=sp_dim)], dim=sp_dim)

# %% ../nbs/01_layers.ipynb 47
@patch
@torch.no_grad()
def inattention(self:Tensor, k=None, sort_dim=0, sp_dim=0):
    """
    returns `self` after zeroing out values smaller than `k`th largest in dimension `dim`.
    If k is `None` behaves like returns self.
    """
    k = min(k if k is not None else np.inf, self.size(sort_dim)-1)
    k_slice= [slice(None)]*self.ndim
    # rep = [1]*self.ndim
    k_slice[sort_dim] = k
    if len(k_slice) == 1: k_slice=k_slice[0]
    # rep[sort_dim] = self.size(sort_dim)
    kth_largest = split_sort(self, sp_dim=sp_dim, sort_dim=sort_dim, descending=True)[k_slice].unsqueeze(dim=sort_dim)#.repeat(*rep)
    # [This line wasted 6 months of deb's life :(] clone = self.detach().clone()
    self[self < kth_largest] = 0.
    return self

# %% ../nbs/01_layers.ipynb 51
from fastai.losses import CrossEntropyLossFlat
from .utils import *

# %% ../nbs/01_layers.ipynb 52
class XMLAttention(Module):
    "Compute label specific attention weights for each token in a sequence"
    def __init__(self, n_lbs, emb_sz, embed_p=0.0, plant=0.5, attn_init=(0, 0, 1), attn_damps=(1, 1, 1), static_inattn=5, diff_inattn=30, lowshot=False, unfreeze_lm_decoder=False):
        store_attr('n_lbs,emb_sz,embed_p,plant')
        self.lbs = Embedding(n_lbs, emb_sz)
        # self.lbs_weight_dp = EmbeddingDropout(self.lbs_weight, embed_p)
        self.attn = Lambda(Linear_Attention(self.lbs))
        # self.plant_wt = nn.Parameter(torch.zeros(1)) #nn.Parameter(torch.empty(1).uniform_(0,1))
        # self.splant_wt = nn.Parameter(torch.zeros(1)) #nn.Parameter(torch.empty(1).uniform_(0,1))
        # self.lin_wt = nn.Parameter(torch.ones(1)) #nn.Parameter(torch.empty(1).uniform_(0,1))
        self.wgts = nn.Parameter(tensor(attn_init).float()) # (lin_wt, plant_wt, splant_wt)
        self.damps = tensor(attn_damps).float() # (lin_wt, plant_wt, splant_wt)
        # self.k = nn.Parameter(torch.empty(1).uniform_(0, n_lbs))
        # self.k = nn.Parameter(torch.empty(1).uniform_(0, 50))
        # self.k = nn.Parameter(torch.normal(mean=torch.tensor(30.0), std=0.01))
        self.compress = nn.Linear(3*emb_sz, 400)
        init_default(self.compress)
        self.static_inattn = static_inattn
        self.diff_inattn = diff_inattn
        self.lowshot = lowshot
        self.unfreeze_lm_decoder = unfreeze_lm_decoder
    
    @property
    def attn(self): return self._attn
    @attn.setter
    def attn(self, a): self._attn = a
    
    def forward(self, inp, sentc, mask):
        # sent is the ouput of SentenceEncoder i.e., (bs, max_len tokens, nh)
        test_eqs(inp.shape, sentc.shape[:-1], mask.shape)
        if self.attn.func.f is _linear_attention:
            top_tok_attn_wgts = F.softmax(self.attn(sentc), dim=1).masked_fill(mask[:,:,None], 0) # lbl specific wts for each token (bs, max_len, n_lbs)
            lbs_cf = None
            lm_loss = 0
            return lincomb(sentc, wgts=top_tok_attn_wgts.transpose(1,2)), top_tok_attn_wgts, lbs_cf, lm_loss # for each lbl do a linear combi of all the tokens based on attn_wgts (bs, num_lbs, nh)
        elif self.attn.func.f is _planted_attention:
            attn_wgts = self.attn(inp).masked_fill(mask[:,:,None], 0)
            top_tok_attn_wgts =attn_wgts.inattention(k=15, sort_dim=1)
            top_lbs_attn_wgts = attn_wgts.clone().permute(0,2,1).inattention(k=5, sort_dim=1).permute(0,2,1).contiguous() # applying `inattention` across the lbs dim
            lbs_cf = top_lbs_attn_wgts.sum(dim=1) #shape (bs, n_lbs)
            return lincomb(sentc, wgts=top_tok_attn_wgts.transpose(1,2)), top_tok_attn_wgts, lbs_cf # for each lbl do a linear combi of all the tokens based on attn_wgts (bs, num_lbs, nh)
        elif self.attn.func.f is _diffntble_planted_attention: 
            
            # import pdb; pdb.set_trace()
            top_tok_splant_attn_wgts =  self.plant_attn(inp).masked_fill(mask[:,:,None], 0).inattention(k=self.static_inattn, sort_dim=1)
            lbs_cf = None
            # lbs_cf = self.plant_attn(inp).masked_fill(mask[:,:,None], 0).inattention(k=10, sort_dim=-1).sum(dim=1).inattention(k=15, sort_dim=-1) #shape (bs, n_lbs)
            
            top_tok_lin_attn_wgts = self.lin_attn(sentc).softmax(dim=1).masked_fill(mask[:,:,None], 0) # lbl specific wts for each token (bs, max_len, n_lbs)
            # change
            
            # import pdb; pdb.set_trace()
            
            # [real] top_tok_plant_attn_wgts = self.attn(self.lm_decoder(sentc)).masked_fill(mask[:,:,None], 0).inattention(k=self.diff_inattn, sort_dim=1).softmax(dim=1) # lbl specific wts for each token (bs, max_len, n_lbs)
            # doing research
            pred_sentc = self.lm_decoder(sentc)
            dec_sentc = pred_sentc.argmax(dim=-1)
            top_tok_plant_attn_wgts = self.attn(dec_sentc).masked_fill(mask[:,:,None], 0).inattention(k=self.diff_inattn, sort_dim=1).softmax(dim=1) # lbl specific wts for each token (bs, max_len, n_lbs)
            # import pdb; pdb.set_trace()
            lm_loss = CrossEntropyLossFlat()(pred_sentc[:,:-1], inp.masked_fill(mask,0)[:, 1:]) if self.unfreeze_lm_decoder else 0
            # doing research
            
            # top_tok_plant_attn_wgts = self.attn(self.lm_decoder(sentc)).masked_fill(mask[:,:,None], 0).inattention(k=self.k.int().item(), sort_dim=1).softmax(dim=1) # lbl specific wts for each token (bs, max_len, n_lbs)
            # top_tok_plant_attn_wgts = self.attn(self.lm_decoder(sentc)).masked_fill(mask[:,:,None], 0).softmax(dim=1) # lbl specific wts for each token (bs, max_len, n_lbs)
            # top_tok_plant_attn_wgts = self.attn(inp).masked_fill(mask[:,:,None], 0).inattention(k=30, sort_dim=1).softmax(dim=1) # lbl specific wts for each token (bs, max_len, n_lbs)
            # change
            
            # top_tok_attn_wgts = (1-self.plant)*top_tok_lin_attn_wgts + self.plant*top_tok_plant_attn_wgts
            # self.lin_wt, self.plant_wt, self.splant_wt = tensor(self.lin_wt, self.plant_wt, self.splant_wt).softmax(dim=-1)
            # self.wgts.data = self.wgts.softmax(dim=-1)
            top_tok_attn_wgts = self.wgts[0]*top_tok_lin_attn_wgts + self.wgts[1]*top_tok_plant_attn_wgts + self.wgts[2]*top_tok_splant_attn_wgts
            # import pdb; pdb.set_trace()
            # top_tok_attn_wgts = self.damps[0]*self.wgts[0]*top_tok_lin_attn_wgts + self.damps[1]*self.wgts[1]*top_tok_plant_attn_wgts + self.damps[2]*self.wgts[2]*top_tok_splant_attn_wgts
            # lbs_cf = None

            lin_comb_lin = lincomb(sentc, wgts=top_tok_lin_attn_wgts.transpose(1,2)) # real
            lin_comb_plant = lincomb(sentc, wgts=top_tok_plant_attn_wgts.transpose(1,2)) # real
            lin_comb_splant = lincomb(sentc, wgts=top_tok_splant_attn_wgts.transpose(1,2)) # real
            # lin_comb_lin_plant = self.compress(torch.cat((lin_comb_lin, lin_comb_plant, lin_comb_splant), dim=-1)).relu()
            lin_comb_lin_plant = lin_comb_lin + lin_comb_plant + lin_comb_splant # real
            
            # only for lowshot
            if self.lowshot: lin_comb_lin_plant = lincomb(sentc, wgts=top_tok_attn_wgts.transpose(1,2))

            # lin_comb_lin_plant = lincomb(sentc, wgts=top_tok_attn_wgts.transpose(1,2))
            # lin_comb_lin_plant = lin_comb_plant

            return lin_comb_lin_plant, top_tok_attn_wgts, lbs_cf, lm_loss # for each lbl do a linear combi of all the tokens based on attn_wgts (bs, num_lbs, nh)
            
        # return lincomb(sentc, wgts=top_tok_attn_wgts.transpose(1,2)), top_tok_attn_wgts, lbs_cf # for each lbl do a linear combi of all the tokens based on attn_wgts (bs, num_lbs, nh)
