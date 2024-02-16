from fastcore.script import *
from fastprogress import fastprogress


import requests
from bs4 import BeautifulSoup
from fastai.basics import *
from fastprogress.fastprogress import master_bar, progress_bar
from xcube.l2r.all import *
from xcube.text.learner import brainsplant, brainsplant_diffntble

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def line():
    print("*"*fastprogress.MAX_COLS)

def get_description(codes):
    # Initialize an empty dictionary to store code descriptions
    icd10_descriptions = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    with requests.Session() as session:
        for code in codes:
            url_cm = f"https://icd10coded.com/cm/{code}"
            url_pcs = f"https://icd10coded.com/pcs/{code}"
            
            for url in (url_pcs, url_cm):   
                with session.get(url, headers=headers, timeout=5) as response:
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        description_element = soup.find('span', {'class': 'lead'})
                        if description_element:
                            description = description_element.text.strip()
                            icd10_descriptions[code] = description
                            break
                        else:
                            icd10_descriptions[code] = "Description not found"
                    else:
                        # print(f"An error occurred.")
                        icd10_descriptions[code] = "Code not found"
    return icd10_descriptions

@call_parse
def main(
    source_url: Param("Source url", str)="XURLs.MIMIC4",
    source_url_l2r: Param("Source url", str)="XURLs.MIMIC4_L2R",
    xml_vocab_file: Param("Classifier vocab file", str)="mimic4_icd10_clas_full_vocab.pkl",
    mutual_info_file: Param("Mutual Info file", str)="mimic4_icd10_tok_lbl_info",
    label_bias_file: Param("Mutual Info file", str)="mimic4_icd10_p_L",
    l2r_model_file: Param("Pretrained L2R Model", str)="mimic4_icd10_l2r_lin_lambdarank",

):
    "Compare Actual Brain and Differentiable Brain"
    source = untar_xxx(eval(source_url))
    xml_vocab = load_pickle(source/xml_vocab_file)
    xml_vocab = L(xml_vocab).map(listify)

    source_l2r = untar_xxx(eval(source_url_l2r))
    boot_path = join_path_file(mutual_info_file, source_l2r, ext='.pkl')
    bias_path = join_path_file(label_bias_file, source_l2r, ext='.pkl')
    l2r_bootstrap = torch.load(boot_path, map_location=default_device())
    brain_bias = torch.load(bias_path, map_location=default_device())


    *brain_vocab, brain = mapt(l2r_bootstrap.get, ['toks', 'lbs', 'mutual_info_jaccard'])
    brain_vocab = L(brain_vocab).map(listify)
    toks, lbs = brain_vocab
    print(f"last two places in brain vocab has {toks[-2:]}")
    # toks = CategoryMap(toks, sort=False)
    brain_bias = brain_bias[:, :, 0].squeeze(-1)
    # lbs_des = load_pickle(source_mimic/'code_desc.pkl')
    # assert isinstance(lbs_des, dict)
    test_eq(brain.shape, (len(toks), len(lbs))) # last two places has 'xxfake'
    test_eq(brain_bias.shape, [len(lbs)])

    not_found_in_brain = L(set(xml_vocab[0]).difference(set(brain_vocab[0])))
    print(f"The tokens which are there in the xml vocab but not in the brain:{not_found_in_brain}")
    print(f"The tokens which are in the brain but were not present in the xml vocab:{set(brain_vocab[0]).difference(xml_vocab[0])}")

    test_shuffled(xml_vocab[1], brain_vocab[1])
    print("Performing Static Brainsplant...")
    xml_brain, xml_lbsbias, toks_map, lbs_map, toks_xml2brain, lbs_xml2brain = brainsplant(xml_vocab, brain_vocab, brain, brain_bias)
    some_lbs = random.sample(lbs, 10)
    # some_lbs = ['518.81', '530.81', '37.23', '934.1',] #'428.20', '784.2', '585.9']
    # some_lbs = ['428.20', '784.2', '585.9']
    stripped_codes = [''.join(filter(str.isalnum, s)) for s in some_lbs]
    _map = dict(zip(some_lbs, stripped_codes))
    desc = get_description(stripped_codes)
    print(f"Some labels: {desc}")


    l2r_wgts = torch.load(join_path_file(l2r_model_file, source_l2r, ext='.pth'), map_location=default_device())
    if 'model' in l2r_wgts: l2r_wgts = l2r_wgts['model']
    print("Performing Differentiable Brainsplant...")
    mod_dict, toks_map, lbs_map = brainsplant_diffntble(xml_vocab, brain_vocab, l2r_wgts)
    assert isinstance(mod_dict, nn.Module)
    assert nn.Module in mod_dict.__class__.__mro__ 
    test_eq(mod_dict['token_factors'].weight.data[toks_map.itemgot(0)], l2r_wgts['token_factors.weight'][toks_map.itemgot(1)])
    test_eq(mod_dict['token_bias'].weight.data[toks_map.itemgot(0)], l2r_wgts['token_bias.weight'][toks_map.itemgot(1)])
    test_eq(mod_dict['label_factors'].weight.data[lbs_map.itemgot(0)], l2r_wgts['label_factors.weight'][lbs_map.itemgot(1)])
    test_eq(mod_dict['label_bias'].weight.data[lbs_map.itemgot(0)], l2r_wgts['label_bias.weight'][lbs_map.itemgot(1)])
    
    print("The Performance of the Actual Brain:")
    for lbl in some_lbs:
        lbl_idx_from_brn = brain_vocab[1].index(lbl)
        tok_vals_from_brn, top_toks_from_brn= L(brain[:, lbl_idx_from_brn].topk(k=20)).map(Self.cpu())
        lbl_idx_from_xml = xml_vocab[1].index(lbl)
        tok_vals_from_xml, top_toks_from_xml = L(xml_brain[:, lbl_idx_from_xml].topk(k=20)).map(Self.cpu())
        test_eq(lbs_xml2brain[lbl_idx_from_xml], lbl_idx_from_brn)
        test_eq(tok_vals_from_brn, tok_vals_from_xml)
        test_eq(array(brain_vocab[0])[top_toks_from_brn], array(xml_vocab[0])[top_toks_from_xml])
        test_eq(brain_bias[lbl_idx_from_brn], xml_lbsbias[lbl_idx_from_xml])
        print(f"For the lbl {lbl} ({desc[_map[lbl]]}), the top tokens that needs attention are:")
        print('\n'.join(L(array(xml_vocab[0])[top_toks_from_xml], use_list=True).zipwith(L(tok_vals_from_xml.numpy(), use_list=True)).map(str).map(lambda o: "+ "+o)))
        line()

    print("The Performance of the Differentiable Brain:")
    lbs_idx = tensor(mapt(xml_vocab[1].index, some_lbs)).to(default_device())
    lbs_idx
    apprx_brain = mod_dict['token_factors'].weight @ mod_dict['label_factors'](lbs_idx).T + mod_dict['token_bias'].weight + mod_dict['label_bias'](lbs_idx).T
    _df = pd.DataFrame(array(xml_vocab[0])[apprx_brain.topk(dim=0, k=40).indices.cpu()], columns=L(some_lbs))
    print(_df)
    _df.to_csv('diff_brain.csv')
    for i, lbl in enumerate(some_lbs):
        lbl_idx_from_xml = xml_vocab[1].index(lbl)
        # import pdb; pdb.set_trace()
        tok_vals_from_apprx_brain, top_toks_from_apprx_brain = L(apprx_brain[:, i].topk(k=40)).map(Self.cpu())
        print(f"For the lbl {lbl} ({desc[_map[lbl]]}), the top tokens that needs attention are:")
        print('\n'.join(L(array(xml_vocab[0])[top_toks_from_apprx_brain], use_list=True).zipwith(L(tok_vals_from_apprx_brain.detach().numpy(), use_list=True)).map(str).map(lambda o: "+ "+o)))
        line()
    line()

    some_toks = random.sample(toks, 2)
    print(f"Some tokens {some_toks}")
    for tok in some_toks:
        tok_idx_from_brn = brain_vocab[0].index(tok)
        lbs_vals_from_brn, top_lbs_from_brn = L(brain[tok_idx_from_brn].topk(k=20)).map(Self.cpu())
        tok_idx_from_xml = xml_vocab[0].index(tok)
        test_eq(tok_idx_from_brn, toks_xml2brain[tok_idx_from_xml])
        lbs_vals_from_xml, top_lbs_from_xml = L(xml_brain[tok_idx_from_xml].topk(k=20)).map(Self.cpu())
        test_eq(lbs_vals_from_brn, lbs_vals_from_xml)
        try: 
            test_eq(array(brain_vocab[1])[top_lbs_from_brn], array(xml_vocab[1])[top_lbs_from_xml])
        except AssertionError as e: 
            pass
            # print(type(e).__name__, "due to instability in sorting (nothing to worry!)");
            # test_shuffled(array(brain_vocab[1])[top_lbs_from_brn], array(xml_vocab[1])[top_lbs_from_xml])
        print('')
        print(f"For the token {tok}, the top labels that needs attention are:")
        print('\n'.join(L(array(xml_vocab[1])[top_lbs_from_xml], use_list=True).zipwith(L(lbs_vals_from_xml.numpy(), use_list=True)).map(str).map(lambda o: "+ "+o)))

    print("Let's generate some random tokens and see how the actual brain and the differentiable brain ranks those:")
    toks_idx = torch.randint(0, len(xml_vocab[0]), (72,)).to(default_device())
    print("-"+'\n-'.join(array(xml_vocab[0])[toks_idx.cpu()].tolist()))
    apprx_brain = mod_dict['token_factors'](toks_idx) @ mod_dict['label_factors'](lbs_idx).T + mod_dict['token_bias'](toks_idx) + mod_dict['label_bias'](lbs_idx).T
    test_eq(apprx_brain.shape, (72,len(some_lbs)))
    print("The static brain would rank these as follows:")
    _df = pd.DataFrame(array(xml_vocab[0])[toks_idx[xml_brain[:, lbs_idx][toks_idx].argsort(descending=True, dim=0)].cpu()], columns=L(some_lbs)).head(20)
    print(_df)
    line()
    print("The differentiable brain would rank these as follows:")
    _df = pd.DataFrame(array(xml_vocab[0])[toks_idx[apprx_brain.argsort(dim=0, descending=True)].cpu()], columns=L(some_lbs)).head(20)
    print(_df)

