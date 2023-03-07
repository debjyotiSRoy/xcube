# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/07_l2r.models.core.ipynb.

# %% auto 0
__all__ = ['L2R_DotProductBias', 'L2R_NN']

# %% ../../../nbs/07_l2r.models.core.ipynb 2
from fastai.torch_imports import *
from fastai.layers import *
from ...layers import *

# %% ../../../nbs/07_l2r.models.core.ipynb 6
class L2R_DotProductBias(nn.Module):
    def __init__(self, num_lbs, num_toks, num_factors, y_range=None):
        super().__init__()
        self.num_toks, self.num_lbs = num_toks+1, num_lbs+1 # +1 for the `padding_idx` 
        self.num_factors = num_factors
        self.token_factors = nn.Embedding(self.num_toks, num_factors, padding_idx=-1)
        self.token_bias = nn.Embedding(self.num_toks, 1, padding_idx=-1)
        self.label_factors = nn.Embedding(self.num_lbs, num_factors, padding_idx=-1)
        self.label_bias = nn.Embedding(self.num_lbs, 1, padding_idx=-1)
        self.y_range = y_range
        
    def forward(self, xb):
        # import pdb; pdb.set_trace()
        xb_toks = xb[:, :, :, 0].long() # xb[...,0] # shape (64, 2233, 64)
        xb_lbs = torch.unique(xb[:, :, :, 1], dim=-1).flatten(start_dim=1).long() # shape (64, 2233, )
        # To convert -1 which is the padding index to the last index:
        xb_toks, xb_lbs= xb_toks%(self.num_toks), xb_lbs%(self.num_lbs)
        
        toks_embs = self.token_factors(xb_toks) # shape (64, 2233, 64, 400)
        toks_shape = toks_embs.shape
        toks_embs = toks_embs.view(-1, *toks_shape[2:]) # shape (64*2233, 64, 400)

        lbs_embs = self.label_factors(xb_lbs) # shape (64, 2233, 400)
        lbs_shape = lbs_embs.shape
        lbs_embs = lbs_embs.view(-1, *lbs_shape[2:]).unsqueeze(dim=-1) # shape (64*2233, 400, 1)
        
        res = torch.bmm(toks_embs, lbs_embs) # shape (64*2233, 64, 1)
        # res = torch.matmul(toks_embs, lbs_embs)
        res = res.view(toks_shape[0], toks_shape[1], *res.shape[1:]) + self.token_bias(xb_toks) + self.label_bias(xb_lbs).unsqueeze(2) # shape (64, 2233, 64, 1)
        
        return sigmoid_range(res, *self.y_range) if self.y_range is not None else res
        # return res

# %% ../../../nbs/07_l2r.models.core.ipynb 8
class L2R_NN(nn.Module):
    def __init__(self, num_lbs, 
                 num_toks, 
                 num_factors, 
                 layers,
                 ps=None,
                 use_bn=True,
                 bn_final=False,
                 lin_first=True,
                 embed_p=0.0,
                 y_range=None):
        super().__init__()
        self.num_toks, self.num_lbs = num_toks+1, num_lbs+1 # +1 for the `padding_idx` 
        self.num_factors, self.embed_p, self.ps = num_factors, embed_p, ps
        self.token_factors = nn.Embedding(self.num_toks, num_factors, padding_idx=-1)
        self.label_factors = nn.Embedding(self.num_lbs, num_factors, padding_idx=-1)
        self.emb_drop = nn.Dropout(embed_p)
        self.y_range = y_range
        if ps is None: self.ps = [0. for _ in range(len(layers))]
        sizes = [self.token_factors.embedding_dim] + layers + [1]
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        _layers = [LinBnFlatDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first)
                   for i, (p,a) in enumerate(zip(self.ps+[0.],actns))]
        self.layers= nn.Sequential(*_layers)
        # self.layers = nn.Sequential(
        #     nn.Linear(num_factors, n_act),
        #     nn.ReLU(),
        #     nn.Linear(n_act, 1),
        #     nn.Dropout(self.ps),
        # )
        
    # def __str__(self): return super().__repr__() + f"\n {self.n_act = }, {self.embed_p = }"
    # __repr__ = __str__

    
    def forward(self, xb):
        # import pdb; pdb.set_trace()
        xb_toks = xb[:, :, :, 0].long() # xb[...,0] # shape (64, 2233, 64)
        xb_lbs = torch.unique(xb[:, :, :, 1], dim=-1).flatten(start_dim=1).long() # shape (64, 2233, )
        # To convert -1 which is the padding index to the last index:
        xb_toks, xb_lbs= xb_toks%(self.num_toks), xb_lbs%(self.num_lbs)
        
        toks_embs = self.token_factors(xb_toks) # shape (64, 2233, 64, 200)

        lbs_embs = self.label_factors(xb_lbs) # shape (64, 2233, 200)
        lbs_embs = lbs_embs.unsqueeze(2) # shape (64, 2233, 1, 200)
        lbs_embs = lbs_embs.expand(-1, -1, xb.shape[2], -1)
        
        # embs = torch.cat((toks_embs, lbs_embs), dim=-1) # shape (64, 2233, 64, 400)
        embs = toks_embs + lbs_embs
        embs = self.emb_drop(embs)
        res = self.layers(embs)
        
        return sigmoid_range(res, *self.y_range) if self.y_range is not None else res
        # return res
