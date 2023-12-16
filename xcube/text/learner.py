# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03_text.learner.ipynb.

# %% auto 0
__all__ = ['match_collab', 'brainsplant', 'brainsplant_diffntble', 'load_collab_keys', 'TextLearner',
           'xmltext_classifier_learner']

# %% ../../nbs/03_text.learner.ipynb 1
from fastai.basics import *
from fastai.text.learner import *
from fastai.callback.rnn import *
from fastai.text.models.awdlstm import *
from fastai.text.models.core import get_text_classifier
from fastprogress.fastprogress import master_bar, progress_bar
from .models.core import *

# %% ../../nbs/03_text.learner.ipynb 7
def _get_text_vocab(dls:DataLoaders) -> list:
    "Get text vocabulary from `DataLoaders`"
    vocab = dls.vocab
    if isinstance(vocab, L): vocab = vocab[0]
    return vocab

# %% ../../nbs/03_text.learner.ipynb 8
def _get_label_vocab(dls:DataLoaders) -> list:
    "Get label vocabulary from `DataLoaders`"
    vocab = dls.vocab
    if isinstance(vocab, L): vocab = vocab[1]
    return vocab

# %% ../../nbs/03_text.learner.ipynb 9
def match_collab(
    old_wgts:dict, # Embedding weights of the colab model
    collab_vocab:dict, # Vocabulary of `token` and `label` used for colab pre-training
    lbs_vocab:list # Current labels vocabulary
) -> dict:
    "Convert the label embedding in `old_wgts` to go from `old_vocab` in colab to `lbs_vocab`"
    bias, wgts = old_wgts.get('i_bias.weight', None), old_wgts.get('i_weight.weight')
    wgts_m = wgts.mean(0)
    new_wgts = wgts.new_zeros((len(lbs_vocab), wgts.size(1)))
    if bias is not None:
        bias_m = bias.mean(0)
        new_bias = bias.new_zeros((len(lbs_vocab), 1))
    collab_lbs_vocab = collab_vocab['label']
    collab_o2i = collab_lbs_vocab.o2i if hasattr(collab_lbs_vocab, 'o2i') else {w:i for i,w in enumerate(collab_lbs_vocab)}
    missing = 0
    for i,w in enumerate(lbs_vocab):
        idx = collab_o2i.get(w, -1)
        new_wgts[i] = wgts[idx] if idx>=0 else wgts_m
        if bias is not None: new_bias[i] = bias[idx] if idx>=0 else bias_m
        if idx == -1: missing = missing + 1
    old_wgts['i_weight.weight'] = new_wgts
    if bias is not None: old_wgts['i_bias.weight'] = new_bias
    return old_wgts, missing

# %% ../../nbs/03_text.learner.ipynb 24
def _xml2brain(xml_vocab, brain_vocab, parent_bar=None):
    "Creates a mapping between the indices of the xml vocab and the brain vocab"
    pbar = progress_bar(xml_vocab, parent=parent_bar, leave=True)
    xml2brain = {i: brain_vocab.index(o) if o in brain_vocab else np.inf  for i,o in enumerate(pbar)}
    xml2brain_notfnd = [o for o in xml2brain if xml2brain[o] is np.inf]
    return xml2brain, xml2brain_notfnd

# %% ../../nbs/03_text.learner.ipynb 27
def brainsplant(xml_vocab, brain_vocab, brain, brain_bias, device=None):
    toks_lbs = 'toks lbs'.split()
    mb = master_bar(range(2))
    for i in mb:
        globals().update(dict(zip((toks_lbs[i]+'_xml2brain', toks_lbs[i]+'_notfnd'), (_xml2brain(xml_vocab[i], brain_vocab[i], parent_bar=mb)))))
        mb.write = f"Finished Loop {i}"
    xml_brain = torch.zeros(*xml_vocab.map(len)).to(default_device() if device is None else device) # initialize empty brain
    xml_lbsbias = torch.zeros(len(xml_vocab[1])).to(default_device() if device is None else device)
    toks_map = L((xml_idx, brn_idx) for xml_idx, brn_idx in toks_xml2brain.items() if brn_idx is not np.inf) 
    lbs_map = L((xml_idx, brn_idx) for xml_idx, brn_idx in lbs_xml2brain.items() if brn_idx is not np.inf) 
    xml_brain[toks_map.itemgot(0)] = brain[toks_map.itemgot(1)][:, lbs_map.itemgot(1)] # permute toks dim to match xml and brain
    xml_brain[:, lbs_map.itemgot(0)] = xml_brain.clone() # permute lbs dim to match xml and brain
    xml_lbsbias[lbs_map.itemgot(0)] = brain_bias[lbs_map.itemgot(1)].clone() # permute toks dim to match xml and brain
    return xml_brain, xml_lbsbias, toks_map, lbs_map, toks_xml2brain, lbs_xml2brain

# %% ../../nbs/03_text.learner.ipynb 41
def brainsplant_diffntble(xml_vocab, brain_vocab, l2r_wgts, device=None):
    toks_lbs = 'toks lbs'.split()
    mb = master_bar(range(2))
    for i in mb:
        globals().update(dict(zip((toks_lbs[i]+'_xml2brain', toks_lbs[i]+'_notfnd'), (_xml2brain(xml_vocab[i], brain_vocab[i], parent_bar=mb)))))
        mb.write = f"Finished Loop {i}" 
    toks_map = L((xml_idx, brn_idx) for xml_idx, brn_idx in toks_xml2brain.items() if brn_idx is not np.inf) 
    lbs_map = L((xml_idx, brn_idx) for xml_idx, brn_idx in lbs_xml2brain.items() if brn_idx is not np.inf) 
    tf_xml = torch.zeros(len(xml_vocab[0]), l2r_wgts['token_factors.weight'].shape[1]).to(default_device() if device is None else device) 
    tb_xml = torch.zeros(len(xml_vocab[0]), 1).to(default_device() if device is None else device) 
    lf_xml = torch.zeros(len(xml_vocab[1]), l2r_wgts['label_factors.weight'].shape[1]).to(default_device() if device is None else device) 
    lb_xml = torch.zeros(len(xml_vocab[1]), 1).to(default_device() if device is None else device) 
    tf_l2r, tb_l2r, lf_l2r, lb_l2r = list(l2r_wgts.values())
    tf_xml[toks_map.itemgot(0)] = tf_l2r[toks_map.itemgot(1)].clone()
    tb_xml[toks_map.itemgot(0)] = tb_l2r[toks_map.itemgot(1)].clone()
    lf_xml[lbs_map.itemgot(0)] = lf_l2r[lbs_map.itemgot(1)].clone()
    lb_xml[lbs_map.itemgot(0)] = lb_l2r[lbs_map.itemgot(1)].clone()
    # import pdb; pdb.set_trace()
    xml_wgts = {k: xml_val for k, xml_val in zip(l2r_wgts.keys(), (tf_xml, tb_xml, lf_xml, lb_xml))}
    mod_dict = nn.ModuleDict({k.split('.')[0]: nn.Embedding(*v.size()) for k,v in xml_wgts.items()}).to(default_device() if device is None else device) 
    mod_dict.load_state_dict(xml_wgts)
    return mod_dict, toks_map, lbs_map

# %% ../../nbs/03_text.learner.ipynb 58
def load_collab_keys(
    model, # Model architecture
    wgts:dict # Model weights
) -> tuple:
    "Load only collab `wgts` (`i_weight` and `i_bias`) in `model`, keeping the rest as is"
    sd = model.state_dict()
    lbs_weight, i_weight = sd.get('1.attn.lbs_weight.weight', None), wgts.get('i_weight.weight', None)
    lbs_bias, i_bias = sd.get('1.attn.lbs_weight.bias', None), wgts.get('i_bias.weight', None) 
    if lbs_weight is not None and i_weight is not None: lbs_weight.data = i_weight.data
    if lbs_bias is not None and i_bias is not None: lbs_bias.data = i_bias.data
    if '1.attn.lbs_weight_dp.emb.weight' in sd:
        sd['1.attn.lbs_weight_dp.emb.weight'] = i_weight.data.clone()
    return model.load_state_dict(sd)

# %% ../../nbs/03_text.learner.ipynb 62
from ..layers import *
from ..layers import _planted_attention, _diffntble_planted_attention

# %% ../../nbs/03_text.learner.ipynb 63
@delegates(Learner.__init__)
class TextLearner(Learner):
    "Basic class for a `Learner` in NLP."
    def __init__(self, 
        dls:DataLoaders, # Text `DataLoaders`
        model, # A standard PyTorch model
        alpha:float=2., # Param for `RNNRegularizer`
        beta:float=1., # Param for `RNNRegularizer`
        moms:tuple=(0.8,0.7,0.8), # Momentum for `Cosine Annealing Scheduler`
        **kwargs
    ):
        super().__init__(dls, model, moms=moms, **kwargs)
        self.add_cbs(rnn_cbs())

    def save_encoder(self, 
        file:str # Filename for `Encoder` 
    ):
        "Save the encoder to `file` in the model directory"
        if rank_distrib(): return # don't save if child proc
        encoder = get_model(self.model)[0]
        if hasattr(encoder, 'module'): encoder = encoder.module
        torch.save(encoder.state_dict(), join_path_file(file, self.path/self.model_dir, ext='.pth'))
    
    @delegates(save_model)
    def save(self,
        file:str, # Filename for the state_directory of the model
        **kwargs
    ):
        """
        Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`
        Save `self.dls.vocab` to `self.path/self.model_dir/clas_vocab.pkl`
        """
        model_file = join_path_file(file, self.path/self.model_dir, ext='.pth')
        vocab_file = join_path_file(file+'_vocab', self.path/self.model_dir, ext='.pkl')
        save_model(model_file, self.model, getattr(self, 'opt', None), **kwargs)
        save_pickle(vocab_file, self.dls.vocab)
        return model_file

    def load_encoder(self, 
        file:str, # Filename of the saved encoder 
        device:(int,str,torch.device)=None # Device used to load, defaults to `dls` device
    ):
        "Load the encoder `file` from the model directory, optionally ensuring it's on `device`"
        encoder = get_model(self.model)[0]
        if device is None: device = self.dls.device
        if hasattr(encoder, 'module'): encoder = encoder.module
        distrib_barrier()
        wgts = torch.load(join_path_file(file,self.path/self.model_dir, ext='.pth'), map_location=device)
        encoder.load_state_dict(clean_raw_keys(wgts))
        self.freeze()
        return self

    def load_pretrained(self, 
        wgts_fname:str, # Filename of saved weights 
        vocab_fname:str, # Saved vocabulary filename in pickle format
        model=None # Model to load parameters from, defaults to `Learner.model`
    ):
        "Load a pretrained model and adapt it to the data vocabulary."
        old_vocab = load_pickle(vocab_fname)
        new_vocab = _get_text_vocab(self.dls)
        distrib_barrier()
        wgts = torch.load(wgts_fname, map_location = lambda storage,loc: storage)
        if 'model' in wgts: wgts = wgts['model'] #Just in case the pretrained model was saved with an optimizer
        wgts = match_embeds(wgts, old_vocab, new_vocab)
        load_ignore_keys(self.model if model is None else model, clean_raw_keys(wgts))
        self.freeze()
        return self

    #For previous versions compatibility. Remove at release
    @delegates(load_model_text)
    def load(self, 
        file:str, # Filename of saved model 
        with_opt:bool=None, # Enable to load `Optimizer` state
        device:(int,str,torch.device)=None, # Device used to load, defaults to `dls` device
        **kwargs
    ):
        if device is None: device = self.dls.device
        if self.opt is None: self.create_opt()
        file = join_path_file(file, self.path/self.model_dir, ext='.pth')
        load_model_text(file, self.model, self.opt, device=device, **kwargs)
        return self
    
    def load_collab(self,
        wgts_fname:str, # Filename of the saved collab model
        collab_vocab_fname:str, # Saved Vocabulary of collab labels in pickle format 
        model=None # Model to load parameters from, defaults to `Learner.model`
    ):
        "Load the label embeddings learned by collab model`, and adapt it to the label vocabulary."
        collab_vocab = load_pickle(collab_vocab_fname)
        lbs_vocab = _get_label_vocab(self.dls)
        distrib_barrier()
        wgts = torch.load(wgts_fname, map_location=lambda storage,loc: storage)
        if 'model' in wgts: wgts = wgts['model'] #Just in case the pretrained model was saved with an optimizer
        wgts, _ = match_collab(wgts, collab_vocab, lbs_vocab)
        load_collab_keys(self.model if model is None else model, wgts)
        self.freeze()
        return self

# %% ../../nbs/03_text.learner.ipynb 65
@patch
def save_decoder(self:LMLearner,
                 file:str # Filename for `Decoder`
    ):
    "Save the decoder to `file` in the model directory"
    if rank_distrib(): return # don't save if child proc
    decoder = get_model(self.model)[1]
    if hasattr(decoder, 'module'): decoder = decoder.module
    torch.save(decoder.state_dict(), join_path_file(file, self.path/self.model_dir, ext='.pth'))

# %% ../../nbs/03_text.learner.ipynb 66
@patch
def load_brain(self:TextLearner,
               file_wgts: str, # Filename of the saved attention wgts
               file_bias: str, # Filename of the saved label bias
               device:(int,str,torch.device)=None # Device used to load, defaults to `default_device()`
              ):
    """Load the pre-learnt label specific attention weights for each token from `file` located in the model directory, 
    optionally ensuring it's one `device`
    """
    brain_path = join_path_file(file_wgts, self.path/self.model_dir, ext='.pkl')
    bias_path = join_path_file(file_bias, self.path/self.model_dir, ext='.pkl')
    brain_bootstrap = torch.load(brain_path, map_location=default_device() if device is None else device)
    *brain_vocab, brain = mapt(brain_bootstrap.get, ['toks', 'lbs', 'mutual_info_jaccard'])
    brain_vocab = L(brain_vocab).map(listify)
    vocab = L(_get_text_vocab(self.dls), _get_label_vocab(self.dls)).map(listify)
    brain_bias = torch.load(bias_path, map_location=default_device() if device is None else device)
    brain_bias = brain_bias[:, :, 0].squeeze(-1)
    print("Performing brainsplant...")
    self.brain, self.lbsbias, *_ = brainsplant(vocab, brain_vocab, brain, brain_bias)
    print("Successfull!")
    # import pdb; pdb.set_trace()
    plant_attn_layer = Lambda(Planted_Attention(self.brain))
    setattr(self.model[1].pay_attn, 'attn', plant_attn_layer)
    assert self.model[1].pay_attn.attn.func.f is _planted_attention
    return self

# %% ../../nbs/03_text.learner.ipynb 67
@patch
def load_diffntble_brain(self:TextLearner,
                         file_brain:str, # Filename of the bootstrap info for l2r
                         file_l2r_wgts:str, # Filename of the pretrained l2r wgts
                         file_lm_decoder_wgts:str, # Filename of the pretrained LM decoder wgts
                         device:(int,str,torch.device)=None # # Device used to load, defaults to `default_device()`
                        ):
    """Loads the pre-learnt L2R and LM decoder wgts from `file_l2r_wgts` and ` file_lm_decoder_wgts` located in the
    model directory, optionally ensuring it's on `device`
    """
    brain_bootstrap = torch.load(join_path_file(file_brain, self.path/self.model_dir, ext='.pkl'), map_location=default_device() if device is None else device)
    brain_vocab = mapt(brain_bootstrap.get, ['toks', 'lbs'])
    brain_vocab = L(brain_vocab).map(listify)
    vocab = L(_get_text_vocab(self.dls), _get_label_vocab(self.dls)).map(listify)
    l2r_wgts = torch.load(join_path_file(file_l2r_wgts, self.path/self.model_dir, ext='.pth'), map_location=default_device() if device is None else device)
    if 'model' in l2r_wgts: l2r_wgts = l2r_wgts['model']
    print("Performing 'differentiable' brainsplant...")
    l2r, toks_map, lbs_map = brainsplant_diffntble(vocab, brain_vocab, l2r_wgts)
    print("Successfull!")
    lm_decoder_pretrained_wgts = torch.load(join_path_file(file_lm_decoder_wgts, self.path/self.model_dir, ext='.pth'), map_location=default_device() if device is None else device)
    config = awd_lstm_lm_config.copy()
    emb_sz, output_p, out_bias = map(config.get, ['emb_sz', 'output_p', 'out_bias'])
    lm_decoder = PlantedLMDecoder(len(vocab[0]), emb_sz, output_p=output_p*0.3, plant_wgts=lm_decoder_pretrained_wgts, bias=out_bias).to(default_device() if device is None else device)
    test_eq(lm_decoder.decoder.weight, lm_decoder_pretrained_wgts['decoder.weight'])
    test_eq(lm_decoder.decoder.bias, lm_decoder_pretrained_wgts['decoder.bias'])
    plant_attn_layer = Lambda(Diffntble_Planted_Attention(l2r))
    setattr(self.model[1].pay_attn, 'attn', plant_attn_layer)
    assert self.model[1].pay_attn.attn.func.f is _diffntble_planted_attention
    setattr(self.model[1].pay_attn, 'lm_decoder', lm_decoder)
    return self

# %% ../../nbs/03_text.learner.ipynb 68
@patch
def load_both(self:TextLearner,
                         file_brain:str, # Filename of the bootstrap info for l2r
                         file_bias: str, # Filename of the saved label bias
                         file_l2r_wgts:str, # Filename of the pretrained l2r wgts
                         file_lm_decoder_wgts:str, # Filename of the pretrained LM decoder wgts
                         device:(int,str,torch.device)=None # # Device used to load, defaults to `default_device()`
                        ):
    bias_path = join_path_file(file_bias, self.path/self.model_dir, ext='.pkl')
    brain_path = join_path_file(file_brain, self.path/self.model_dir, ext='.pkl')
    brain_bootstrap = torch.load(brain_path, map_location=default_device() if device is None else device)
    *brain_vocab, brain = mapt(brain_bootstrap.get, ['toks', 'lbs', 'mutual_info_jaccard'])
    brain_vocab = L(brain_vocab).map(listify)
    vocab = L(_get_text_vocab(self.dls), _get_label_vocab(self.dls)).map(listify)
    brain_bias = torch.load(bias_path, map_location=default_device() if device is None else device)
    brain_bias = brain_bias[:, :, 0].squeeze(-1)
    print("Performing static brainsplant...")
    self.brain, self.lbsbias, *_ = brainsplant(vocab, brain_vocab, brain, brain_bias)
    print("Successfull!")
    plant_attn_layer = Lambda(Planted_Attention(self.brain))
    setattr(self.model[1].pay_attn, 'plant_attn', plant_attn_layer)
    assert self.model[1].pay_attn.plant_attn.func.f is _planted_attention
    
    l2r_wgts = torch.load(join_path_file(file_l2r_wgts, self.path/self.model_dir, ext='.pth'), map_location=default_device() if device is None else device)
    if 'model' in l2r_wgts: l2r_wgts = l2r_wgts['model']
    print("Performing 'differentiable' brainsplant...")
    l2r, toks_map, lbs_map = brainsplant_diffntble(vocab, brain_vocab, l2r_wgts)
    print("Successfull!")
    lm_decoder_pretrained_wgts = torch.load(join_path_file(file_lm_decoder_wgts, self.path/self.model_dir, ext='.pth'), map_location=default_device() if device is None else device)
    config = awd_lstm_lm_config.copy()
    emb_sz, output_p, out_bias = map(config.get, ['emb_sz', 'output_p', 'out_bias'])
    lm_decoder = PlantedLMDecoder(len(vocab[0]), emb_sz, output_p=output_p*0.3, plant_wgts=lm_decoder_pretrained_wgts, bias=out_bias).to(default_device() if device is None else device)
    test_eq(lm_decoder.decoder.weight, lm_decoder_pretrained_wgts['decoder.weight'])
    test_eq(lm_decoder.decoder.bias, lm_decoder_pretrained_wgts['decoder.bias'])
    plant_attn_layer = Lambda(Diffntble_Planted_Attention(l2r))
    
    lin_attn = getattr(self.model[1].pay_attn, 'attn')
    setattr(self.model[1].pay_attn, 'attn', plant_attn_layer)
    setattr(self.model[1].pay_attn, 'l2r', l2r)
    setattr(self.model[1].pay_attn, 'lin_attn', lin_attn)
    assert self.model[1].pay_attn.attn.func.f is _diffntble_planted_attention
    setattr(self.model[1].pay_attn, 'lm_decoder', lm_decoder)
    
    return self 

# %% ../../nbs/03_text.learner.ipynb 71
from .models.core import _model_meta 

# %% ../../nbs/03_text.learner.ipynb 72
@delegates(Learner.__init__)
def xmltext_classifier_learner(dls, arch, seq_len=72, config=None, backwards=False, pretrained=True, collab=False, drop_mult=0.5, n_out=None,
                           lin_ftrs=None, ps=None, max_len=72*20, y_range=None, splitter=None, running_decoder=True, plant=0.5, attn_init=(0, 0, 1), static_inattn=5, diff_inattn=30, lowshot=False, **kwargs):
    "Create a `Learner` with a text classifier from `dls` and `arch`."
    vocab = _get_text_vocab(dls)
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from the data, set `dls.c` or pass `n_out`"
    model = get_xmltext_classifier2(arch, len(vocab), n_out, seq_len=seq_len, config=config, y_range=y_range,
                                drop_mult=drop_mult, max_len=max_len, running_decoder=running_decoder, plant=plant, attn_init=attn_init,
                                static_inattn=static_inattn, diff_inattn=diff_inattn, lowshot=lowshot)
    # model = get_xmltext_classifier(arch, len(vocab), n_out, seq_len=seq_len, config=config, y_range=y_range,
                                # drop_mult=drop_mult, max_len=max_len)
    meta = _model_meta[arch]
    learn = TextLearner(dls, model, splitter=splitter if splitter is not None else meta['split_clas'], **kwargs)
    url = 'url_bwd' if backwards else 'url'
    if pretrained:
        if url not in meta:
            warn("There are no pretrained weights for that architecture yet!")
            return learn
        model_path = untar_data(meta[url], c_key='model')
        try: fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        except IndexError: print(f'The model in {model_path} is incomplete, download again'); raise
        learn = learn.load_pretrained(*fnames, model=learn.model[0])
    if collab:
        try: fnames = [list(learn.path.glob(f'**/collab/*collab*.{ext}'))[0] for ext in ['pth', 'pkl']]
        except IndexError: print(f'The collab model in {learn.path} is incomplete, re-train it!'); raise
        learn = learn.load_colab(*fnames, model=learn.model[1])
    learn.freeze()
    return learn   
