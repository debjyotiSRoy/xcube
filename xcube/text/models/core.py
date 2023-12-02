# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/02_text.models.core.ipynb.

# %% auto 0
__all__ = ['SequentialRNN', 'SentenceEncoder', 'AttentiveSentenceEncoder', 'masked_concat_pool', 'XPoolingLinearClassifier',
           'LabelAttentionClassifier', 'get_xmltext_classifier', 'awd_lstm_xclas_split', 'get_xmltext_classifier2']

# %% ../../../nbs/02_text.models.core.ipynb 1
from fastai.data.all import *
from fastai.text.models.core import *
from fastai.text.models.awdlstm import *
from ...layers import *

# %% ../../../nbs/02_text.models.core.ipynb 4
_model_meta = {AWD_LSTM: {'hid_name':'emb_sz', 'url':URLs.WT103_FWD, 'url_bwd':URLs.WT103_BWD,
                          'config_lm':awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas':awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split},}

# %% ../../../nbs/02_text.models.core.ipynb 8
class SequentialRNN(nn.Sequential):
    "A sequential pytorch module that passes the reset call to its children."
    def reset(self):
        for c in self.children(): getattr(c, 'reset', noop)()

# %% ../../../nbs/02_text.models.core.ipynb 10
def _pad_tensor(t, bs):
    if t.size(0) < bs: return torch.cat([t, t.new_zeros(bs-t.size(0), *t.shape[1:])])
    return t

# %% ../../../nbs/02_text.models.core.ipynb 12
class SentenceEncoder(Module):
    "Create an encoder over `module` that can process a full sentence."
    def __init__(self, bptt, module, pad_idx=1, max_len=None): store_attr('bptt,module,pad_idx,max_len')
    def reset(self): getattr(self.module, 'reset', noop)()

    def forward(self, input):
        bs,sl = input.size()
        self.reset()
        mask = input == self.pad_idx
        outs,masks = [],[]
        for i in range(0, sl, self.bptt):
            #Note: this expects that sequence really begins on a round multiple of bptt
            real_bs = (input[:,i] != self.pad_idx).long().sum()
            o = self.module(input[:real_bs,i: min(i+self.bptt, sl)])
            if self.max_len is None or sl-i <= self.max_len:
                outs.append(o)
                masks.append(mask[:,i: min(i+self.bptt, sl)])
        outs = torch.cat([_pad_tensor(o, bs) for o in outs], dim=1)
        inps = input[:, -outs.shape[1]:] # the ofsetted tokens for the outs
        mask = torch.cat(masks, dim=1)
        return inps,outs,mask

# %% ../../../nbs/02_text.models.core.ipynb 14
class AttentiveSentenceEncoder(Module):
    "Create an encoder over `module` that can process a full sentence."
    def __init__(self, bptt, module, decoder, pad_idx=1, max_len=None, running_decoder=True): 
        store_attr('bptt,module,decoder,pad_idx,max_len,running_decoder')
        self.n_lbs = getattr(self.decoder, 'n_lbs', None)
        
    def reset(self): 
        getattr(self.module, 'reset', noop)()

    def forward(self, input):
        bs,sl = input.size()
        self.reset()
        self.decoder.hl = input.new_zeros((bs, self.n_lbs))
        # print(f"Starting to read a btch of docs start hl.sum() = {self.decoder.hl.sum()}", end='\n')
        mask = input == self.pad_idx
        outs,masks = [],[]
        for i in range(0, sl, self.bptt):
            #Note: this expects that sequence really begins on a round multiple of bptt
            real_bs = (input[:,i] != self.pad_idx).long().sum()
            chunk = slice(i, min(i+self.bptt, sl))
            o = self.module(input[:real_bs, chunk]) # shape (bs, bptt, nh)
            if self.max_len is None or sl-i <= self.max_len:
                outs.append(o)
                masks.append(mask[:, chunk])
                # print(f"\t\t (Within max_len) After reading bptt chunk: hl.sum() = {self.decoder.hl.sum()}", end='\n')
            elif self.running_decoder:
                mask_slice = mask[:real_bs, chunk] 
                inp = input[:real_bs, chunk]
                # import pdb; pdb.set_trace()
                hl, *_ = self.decoder((inp, o, mask_slice))
                self.decoder.hl = hl.sigmoid()#.detach()
                # print(f"\t (Outside max_len) After reading bptt chunk: hl.sum() = {self.decoder.hl.sum()}", end='\n')
                
        # import pdb; pdb.set_trace()
        outs = torch.cat([_pad_tensor(o, bs) for o in outs], dim=1)
        inps = input[:, -outs.shape[1]:] # the ofsetted tokens for the outs
        mask = torch.cat(masks, dim=1)
        return inps, outs, mask

# %% ../../../nbs/02_text.models.core.ipynb 15
def masked_concat_pool(output, mask, bptt):
    "Pool `MultiBatchEncoder` outputs into one vector [last_hidden, max_pool, avg_pool]"
    lens = output.shape[1] - mask.long().sum(dim=1)
    last_lens = mask[:,-bptt:].long().sum(dim=1)
    avg_pool = output.masked_fill(mask[:, :, None], 0).sum(dim=1)
    avg_pool.div_(lens.type(avg_pool.dtype)[:,None])
    max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0]
    x = torch.cat([output[torch.arange(0, output.size(0)),-last_lens-1], max_pool, avg_pool], 1) #Concat pooling.
    return x

# %% ../../../nbs/02_text.models.core.ipynb 16
class XPoolingLinearClassifier(Module):
    def __init__(self, dims, ps, bptt, y_range=None):
        self.layer = LinBnDrop(dims[0], dims[1], p=ps, act=None)
        self.bptt = bptt

    def forward(self, input):
        out, mask = input
        x = masked_concat_pool(out, mask, self.bptt)
        x = self.layer(x)
        return x, out, out

# %% ../../../nbs/02_text.models.core.ipynb 19
from ...layers import _create_bias

# %% ../../../nbs/02_text.models.core.ipynb 20
from ...utils import *

# %% ../../../nbs/02_text.models.core.ipynb 21
class LabelAttentionClassifier(Module):
    initrange=0.1
    def __init__(self, n_hidden, n_lbs, plant=0.5, attn_init=(0,0,1), y_range=None, static_inattn=5, diff_inattn=30):
        store_attr('n_hidden,n_lbs,y_range')
        self.pay_attn = XMLAttention(self.n_lbs, self.n_hidden, plant=plant, attn_init=attn_init, static_inattn=static_inattn, diff_inattn=diff_inattn)
        # self.boost_attn = ElemWiseLin(self.n_lbs, self.n_hidden)
        self.boost_attn = ElemWiseLin(self.n_lbs, 400)
        self.label_bias = _create_bias((self.n_lbs,), with_zeros=False)
        self.hl = torch.zeros(1)
    
    def forward(self, sentc):
        if isinstance(sentc, tuple): inp, sentc, mask = sentc # sentc is the stuff coming outta SentenceEncoder i.e., shape (bs, max_len, nh) in other words the concatenated output of the AWD_LSTM
        test_eqs(inp.shape, sentc.shape[:-1], mask.shape)
        sentc = sentc.masked_fill(mask[:, :, None], 0)
        # import pdb; pdb.set_trace()
        attn, wgts, lbs_cf = self.pay_attn(inp, sentc, mask) #shape (bs, n_lbs, n_hidden)
        attn = self.boost_attn(attn) # shape (bs, n_lbs, n_hidden)
        bs = self.hl.size(0)
        self.hl = self.hl.to(sentc.device)
        pred = self.hl + _pad_tensor(attn.sum(dim=2), bs) + self.label_bias  # shape (bs, n_lbs)
        
        if lbs_cf is not None: 
            lbs_cf = _pad_tensor(lbs_cf, bs)
            pred.add_(lbs_cf) 
        
        if self.y_range is not None: pred = sigmoid_range(pred, *self.y_range)
        return pred, attn, wgts, lbs_cf 

# %% ../../../nbs/02_text.models.core.ipynb 24
def get_xmltext_classifier(arch, vocab_sz, n_class, seq_len=72, config=None, drop_mult=1., pad_idx=1, max_len=72*20, y_range=None):
    "Create a text classifier from `arch` and its `config`, maybe `pretrained`"
    meta = _model_meta[arch]
    config = ifnone(config, meta['config_clas']).copy()
    for k in config.keys():
        if k.endswith('_p'): config[k] *= drop_mult
    n_hidden = config[meta['hid_name']]
    config.pop('output_p')
    init = config.pop('init') if 'init' in config else None
    encoder = SentenceEncoder(seq_len, arch(vocab_sz, **config), pad_idx=pad_idx, max_len=max_len)
    decoder = LabelAttentionClassifier(n_hidden, n_class, y_range=y_range)
    model = SequentialRNN(encoder, decoder)
    return model if init is None else model.apply(init)

# %% ../../../nbs/02_text.models.core.ipynb 25
import re
def _get_filter_params(m, but:re.Pattern):
    return [p for n, p in list(m.named_parameters()) if not but.match(n)]

# %% ../../../nbs/02_text.models.core.ipynb 26
def awd_lstm_xclas_split(model):
    "Split a RNN `model` in groups for differential learning rates."
    groups = [nn.Sequential(model[0].module.encoder, model[0].module.encoder_dp)]
    groups += [nn.Sequential(rnn, dp) for rnn, dp in zip(model[0].module.rnns, model[0].module.hidden_dps)]
    return L(groups).map(params) + [params(model[1].pay_attn.lm_decoder)] + [params(model[1].pay_attn.l2r)] + L(model[1]).map(partial(_get_filter_params, but=re.compile('^(pay_attn.l2r|pay_attn.lm_decoder)')))

# %% ../../../nbs/02_text.models.core.ipynb 28
def get_xmltext_classifier2(arch, vocab_sz, n_class, seq_len=72, config=None, drop_mult=1., pad_idx=1, max_len=72*20, y_range=None, running_decoder=True, 
                           plant=0.5, attn_init=(0, 0, 1), static_inattn=5, diff_inattn=30):
    "Create a text classifier from `arch` and its `config`, maybe `pretrained`"
    meta = _model_meta[arch]
    config = ifnone(config, meta['config_clas']).copy()
    for k in config.keys():
        if k.endswith('_p'): config[k] *= drop_mult
    n_hidden = config[meta['hid_name']]
    config.pop('output_p')
    init = config.pop('init') if 'init' in config else None
    decoder = LabelAttentionClassifier(n_hidden, n_class, plant=plant, attn_init=attn_init, y_range=y_range, static_inattn=static_inattn, diff_inattn=diff_inattn)
    encoder = AttentiveSentenceEncoder(seq_len, arch(vocab_sz, **config), decoder, pad_idx=pad_idx, max_len=max_len, running_decoder=running_decoder)
    model =  SequentialRNN(encoder, decoder)
    return model if init is None else model.apply(init)
