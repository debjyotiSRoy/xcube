xcube
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

## E**X**plainable E**X**treme Multi-Label Te**X**t Classification:

- *What is XMTC?* Extreme Multi-Label Text Classification (XMTC)
  addresses the problem of automatically assigning each data point with
  most relevant subset of labels from an extremely large label set. One
  major application of XMTC is in the global healthcare system,
  specifically in the context of the International Classification of
  Diseases (ICD). ICD coding is the process of assigning codes
  representing diagnoses and procedures performed during a patient visit
  using clinical notes documented by health professionals.

- *Datasets?* Examples of ICD coding dataset:
  [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and
  [MIMIC-IV](https://physionet.org/content/mimic-iv-note/2.2/). Please
  note that you need to be a credentialated user and complete a training
  to acces the data.

- *What is xcube?* xcube trains and explains XMTC models using LLM
  fine-tuning.

## Install

- Create new conda environment:

``` sh
conda create -n xxx python=3.10
```

``` sh
conda activate xxx
```

- Install PyTorch with cuda enabled: \[Optional\]

``` sh
conda search pytorch
```

<img alt="output of conda search pytorch" width="400" src="pics/pytorch.png" caption="Pictorial representation of mutual information gain" id="img_mut_info">

use the build string that matches the python and cuda version, replacing
the pytorch version and build string appropriately:

``` sh
conda install pytorch=2.0.0=cuda118py310h072bc4c pytorch-cuda=11.8 -c pytorch -c nvidia
```

Update cuda-toolkit:

``` sh
sudo apt install nvidia-cuda-toolkit
```

Verify cuda is available: Run `python` and
`import torch; torch.cuda.is_available()`

- Install using:

``` sh
pip install xcube
```

Configure accelerate by:

``` sh
accelerate config
```

## How to use

You can either clone the repo and open it in your own machine. Or if you
don’t want to setup a python development environment, an even easier and
quicker approach is to open this repo using [Google
Colab](https://colab.research.google.com/). You can open this readme
page in Colab using this
[link](https://colab.research.google.com/github/debjyotiSRoy/xcube/blob/plant/nbs/index.ipynb).

``` python
IN_COLAB = is_colab()
```

    Not running in Google Colab

``` python
source_mimic3 = untar_xxx(XURLs.MIMIC3_DEMO)
source_mimic4 = untar_xxx(XURLs.MIMIC4)
path = Path.cwd().parent/f"{'xcube' if IN_COLAB else ''}" # root of the repo
(path/'tmp/models').mkdir(exist_ok=True, parents=True)
tmp = path/'tmp'
# os.chdir( f"{path/'scripts'}") # To launch our train/infer scripts
```

Check your GPU memory! If you are running this on google colab be sure
to turn on the GPU runtime. You should be able to train and infer all
the models with atleast 16GB of memory. However, note that training the
full versions of the datasets from scratch requires atleast 48GB memory.

``` python
cudamem()
```

    GPU: Quadro RTX 8000
    You are using 0.0 GB
    Total GPU memory = 44.99969482421875 GB

### Train and Infer on MIMIC3-rare50

MIMIC3-rare50 refers to a split of
[MIMIC-III](https://physionet.org/content/mimiciii/1.4/) that contains
the 50 most rare codes (Refer to [Knowledge Injected Prompt Based
Fine-Tuning for Multi-label Few-shot ICD
Coding](https://aclanthology.org/2022.findings-emnlp.127/) for split
creation).

``` python
data = join_path_file('mimic3-9k_rare50', source_mimic3, ext='.csv')
!head -n 1 {data}
```

    subject_id,hadm_id,text,labels,length,is_valid,split

``` python
df = df = pd.read_csv(data,
                 header=0,
                 names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid', 'split'],
                 dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool, 'split': str})
df.head(2)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>hadm_id</th>
      <th>text</th>
      <th>labels</th>
      <th>length</th>
      <th>is_valid</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2707</td>
      <td>100626</td>
      <td>admission date discharge date date of birth sex f service nsu history of present illness the patient is a year old patient with down syndrome who was transferred to hospital3 hospital for an expanding left subdural hematoma with change in mental status and aspiration pneumonia allergies the patient has no known allergies physical exam temp bp heart rate respiratory rate sats percent on room air the patient was awake noncommunicative at baseline attends examiner noncooperative pupils down to mm and briskly reactive eoms full face symmetric follows commands in the upper extremity moves the l...</td>
      <td>318.2</td>
      <td>334</td>
      <td>False</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16650</td>
      <td>176541</td>
      <td>admission date discharge date date of birth sex m service surgery allergies mirtazapine attending first name3 lf chief complaint multiple self inflicted stab wounds major surgical or invasive procedure closure of stab wounds history of present illness patient was found in a park non verbal at the scene after self inflicted stab wounds to l chest x past medical history depression si sa x2 dm2 htn social history depression quit lost job years ago after a divorce lost health insurance afterwards multipl suicide attempts family history non contributory physical exam heent wnl cv rrr no mrg che...</td>
      <td>34.71</td>
      <td>424</td>
      <td>False</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>

To launch the training of an XMTC model on MIMIC3-rare50:

``` python
!./run_scripts.sh --script_list_file script_list_mimic3_rare50train
```

### Train and Infer on MIMIC3-top50

MIMIC3-top50 refers to a split of
[MIMIC-III](https://physionet.org/content/mimiciii/1.4/) that contains
50 most frequent codes (Refer to [Explainable Prediction of Medical
Codes from Clinical Text](https://aclanthology.org/N18-1100/) for split
creation)

``` python
data = join_path_file('mimic3-9k_top50', source_mimic3, ext='.csv')
!head -n 1 {data}
```

    subject_id,hadm_id,text,labels,length,is_valid,split

``` python
df = pd.read_csv(data,
                 header=0,
                 names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid', 'split'],
                 dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool, 'split': str})
df.head(2)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>hadm_id</th>
      <th>text</th>
      <th>labels</th>
      <th>length</th>
      <th>is_valid</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86006</td>
      <td>111912</td>
      <td>admission date discharge date date of birth sex f service surgery allergies patient recorded as having no known allergies to drugs attending first name3 lf chief complaint 60f on coumadin was found slightly drowsy tonight then fell down stairs paramedic found her unconscious and she was intubated w o any medication head ct shows multiple iph transferred to hospital1 for further eval major surgical or invasive procedure none past medical history her medical history is significant for hypertension osteoarthritis involving bilateral knee joints with a dependence on cane for ambulation chronic...</td>
      <td>414.01;427.31;V58.61;401.9;96.71</td>
      <td>230</td>
      <td>False</td>
      <td>dev</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85950</td>
      <td>189769</td>
      <td>admission date discharge date service neurosurgery allergies sulfa sulfonamides attending first name3 lf chief complaint cc cc contact info major surgical or invasive procedure none history of present illness hpi 88m who lives with family had fall yesterday today had decline in mental status ems called pt was unresponsive on arrival went to osh head ct showed large r sdh pt was intubated at osh and transferred to hospital1 for further care past medical history cad s p mi in s p cabg in ventricular aneurysm at that time cath in with occluded rca unable to intervene chf reported ef 1st degre...</td>
      <td>250.00;403.90;V45.81;96.71;585.9</td>
      <td>304</td>
      <td>False</td>
      <td>dev</td>
    </tr>
  </tbody>
</table>
</div>

To infer one our pretrained XMTC models on MIMIC3-top50 (Metrics for
inference - Precision@3,5,8,15):

``` python
model_fnames = L(source_mimic3.glob("**/*top50*.pth")).map(str)
print('\n'.join(model_fnames))
fname = Path(shutil.copy(model_fnames[2], tmp/'models')).name.split('.')[0]
print(f"We are going to infer model {fname}.")
```

    /home/deb/.xcube/data/mimic3_demo/mimic3_clas_top50_plant_L2Runfrozen.pth
    /home/deb/.xcube/data/mimic3_demo/mimic3_clas_top50_plant_L2Rfrozen.pth
    /home/deb/.xcube/data/mimic3_demo/mimic3_clas_top50.pth
    We are going to infer model mimic3_clas_top50.

``` python
!./launches/launch_top50_mimic3 --fname {fname} --no_running_decoder --infer 1
```

To launch the training of an XMTC model on MIMIC3-top50 from scratch:

``` python
!./run_scripts.sh --script_list_file script_list_mimic3_top50train
```

### Train and Infer on MIMIC3-full:

MIMIC3-full refers to the full
[MIMIC-III](https://physionet.org/content/mimiciii/1.4/) dataset. (Refer
to [Explainable Prediction of Medical Codes from Clinical
Text](https://aclanthology.org/N18-1100/) for details of how the data
was curated)

``` python
data = join_path_file('mimic3-9k_full', source_mimic3, ext='.csv')
!head -n 1 {data}
```

    subject_id,hadm_id,text,labels,length,is_valid,split

``` python
df = pd.read_csv(data,
                 header=0,
                 names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid', 'split'],
                 dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool, 'split': str})
df.head(2)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>hadm_id</th>
      <th>text</th>
      <th>labels</th>
      <th>length</th>
      <th>is_valid</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86006</td>
      <td>111912</td>
      <td>admission date discharge date date of birth sex f service surgery allergies patient recorded as having no known allergies to drugs attending first name3 lf chief complaint 60f on coumadin was found slightly drowsy tonight then fell down stairs paramedic found her unconscious and she was intubated w o any medication head ct shows multiple iph transferred to hospital1 for further eval major surgical or invasive procedure none past medical history her medical history is significant for hypertension osteoarthritis involving bilateral knee joints with a dependence on cane for ambulation chronic...</td>
      <td>801.35;348.4;805.06;807.01;998.30;707.24;E880.9;427.31;414.01;401.9;V58.61;V43.64;707.00;E878.1;96.71</td>
      <td>230</td>
      <td>False</td>
      <td>dev</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85950</td>
      <td>189769</td>
      <td>admission date discharge date service neurosurgery allergies sulfa sulfonamides attending first name3 lf chief complaint cc cc contact info major surgical or invasive procedure none history of present illness hpi 88m who lives with family had fall yesterday today had decline in mental status ems called pt was unresponsive on arrival went to osh head ct showed large r sdh pt was intubated at osh and transferred to hospital1 for further care past medical history cad s p mi in s p cabg in ventricular aneurysm at that time cath in with occluded rca unable to intervene chf reported ef 1st degre...</td>
      <td>852.25;E888.9;403.90;585.9;250.00;414.00;V45.81;96.71</td>
      <td>304</td>
      <td>False</td>
      <td>dev</td>
    </tr>
  </tbody>
</table>
</div>

Lets’s look at some of the ICD9 codes description:

``` python
des = load_pickle(source_mimic3/'code_desc.pkl')
lbl_dict = dict()
for lbl in df.labels[1].split(';'):
    lbl_dict[lbl] = des.get(lbl, 'NF')
pd.DataFrame(lbl_dict.items(), columns=['icd9_code', 'desccription'])
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>icd9_code</th>
      <th>desccription</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>852.25</td>
      <td>Subdural hemorrhage following injury, without mention of open intracranial wound, with prolonged [more than 24 hours] loss of consciousness, without return to pre-existing conscious level</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E888.9</td>
      <td>Unspecified fall</td>
    </tr>
    <tr>
      <th>2</th>
      <td>403.90</td>
      <td>Hypertensive renal disease, unspecified, without mention of renal failure</td>
    </tr>
    <tr>
      <th>3</th>
      <td>585.9</td>
      <td>Chronic kidney disease, unspecified</td>
    </tr>
    <tr>
      <th>4</th>
      <td>250.00</td>
      <td>type II diabetes mellitus [non-insulin dependent type] [NIDDM type] [adult-onset type] or unspecified type, not stated as uncontrolled, without mention of complication</td>
    </tr>
    <tr>
      <th>5</th>
      <td>414.00</td>
      <td>Coronary atherosclerosis of unspecified type of vessel, native or graft</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V45.81</td>
      <td>Postsurgical aortocoronary bypass status</td>
    </tr>
    <tr>
      <th>7</th>
      <td>96.71</td>
      <td>Continuous mechanical ventilation for less than 96 consecutive hours</td>
    </tr>
  </tbody>
</table>
</div>

To infer one our pretrained XMTC models on MIMIC3-full (Metrics for
inference - Precision@3,5,8,15):

``` python
model_fnames = L(source_mimic3.glob("**/*full*.pth")).map(str)
print('\n'.join(model_fnames))
fname = Path(shutil.copy(model_fnames[0], tmp/'models')).name.split('.')[0]
print(f"Let's infer the pretrained model {fname}.")
```

    /home/deb/.xcube/data/mimic3_demo/mimic3-9k_clas_full.pth
    Let's infer the pretrained model mimic3-9k_clas_full.

``` python
!./launches/launch_complete_mimic3 --fname {fname} --infer 1 --no_running_decoder
```

### Train and Infer on MIMIC4-full:

MIMIC4-full refers to the full
[MIMIC-IV](https://physionet.org/content/mimiciii/1.4/) dataset using
ICD10 codes. (Refer to [Automated Medical Coding on MIMIC-III and
MIMIC-IV: A Critical Review and Replicability
Study](https://arxiv.org/pdf/2304.10909.pdf) for details of how the data
was curated)

``` python
data = join_path_file('mimic4_icd10_full', source_mimic4, ext='.csv')
!head -n 1 {data}
```

    note_id,subject_id,_id,note_type,note_seq,charttime,storetime,text,icd10_proc,icd10_diag,labels,num_words,num_targets,split,is_valid

``` python
df = pd.read_csv(data,
                    header=0,
                    usecols=['subject_id', '_id', 'text', 'labels', 'num_targets', 'is_valid', 'split'],
                    dtype={'subject_id': str, '_id': str, 'text': str, 'labels': str, 'num_targets': np.int64, 'is_valid': bool, 'split': str})
df.head(2)
```

Let’s look at some of the descriptions of ICD10 codes:

``` python
stripped_codes = [''.join(filter(str.isalnum, s)) for s in df.labels[0].split(';')]
desc = get_description(stripped_codes)
pd.DataFrame(desc.items(), columns=['icd10_code', 'desccription'])
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>icd10_code</th>
      <th>desccription</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E785</td>
      <td>Hyperlipidemia, unspecified</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F0280</td>
      <td>ICD-10-PCS code structure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>G3183</td>
      <td>Neurocognitive disorder with Lewy bodies</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R296</td>
      <td>Repeated falls</td>
    </tr>
    <tr>
      <th>4</th>
      <td>R441</td>
      <td>Visual hallucinations</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Z8546</td>
      <td>Personal history of malignant neoplasm of prostate</td>
    </tr>
  </tbody>
</table>
</div>

To infer one our pretrained XMTC models on MIMIC4-full (Metrics for
inference - Precision@5,8,15):

``` python
print('\n'.join(L(source_mimic4.glob("**/*full*.pth")).map(str)))
model_fname = Path('/home/deb/.xcube/data/mimic4/mimic4_icd10_clas_full.pth')
fname = Path(shutil.copy(model_fname, tmp/'models')).name.split('.')[0]
print(f"Let's infer the pretrained model {fname}.")
```

    /home/deb/.xcube/data/mimic4/mimic4_icd10_clas_full_plant_experiment.pth
    /home/deb/.xcube/data/mimic4/mimic4_icd10_clas_bwd_full.pth
    /home/deb/.xcube/data/mimic4/mimic4_icd10_clas_full.pth
    Let's infer the pretrained model mimic4_icd10_clas_full.

``` python
!./launches/launch_complete_mimic4_icd10 --fname mimic4_icd10_clas_full --no_running_decoder --infer 1
```

    /home/deb/xcube/scripts
    fname is: mimic4_icd10_clas_full
    infer is: 1
    diff_inattn is: 40
    lin_sgdr_lr0 is: 1e-1
    l2r_sgdr_lr0 is: 1e-1
    plant is false
    Training XMTC without Stateful Decoder
    All arguments:
    --source_url=XURLs.MIMIC4
    --source_url_l2r=XURLs.MIMIC4_L2R
    --data=mimic4_icd10_full
    --rarecodes_fname=mimic4_icd10_rarecodes
    --files_lm=mimic4_icd10_lm_finetuned.pth,mimic4_icd10_lm_decoder.pth,mimic4_icd10_dls_lm_vocab.pkl
    --files_l2r=mimic4_icd10_tok_lbl_info.pkl,mimic4_icd10_p_L.pkl,mimic4_icd10_l2r_lin_lambdarank.pth
    --fp16
    --workers=16
    --track_train
    --log
    --lm
    --no_running_decoder=true
    --epochs=[0, 0, 0, 0, 6]
    --lrs_linattn=[(6e-2,1e-6), (1e-2,1e-6), (1e-2, 1e-6), (1e-2,1e-6), (1e-6,1e-6)]
    --lrs_plant=[(6e-2,1e-6), (1e-2,1e-2), (1e-2, 1e-2), (1e-3,1e-3), (1e-5,1e-5)]
    --fit_sgdr
    --unfreeze_l2r
    --sgdr_n_cycles=3
    --lrs_sgdr_linattn=[(1e-1,1e-1), (1e-1,1e-1), (1e-2, 1e-6), (1e-2,1e-6), (1e-6,1e-6)]
    --lrs_sgdr_plant=[(1e-1,1e-1), (1e-1,1e-1), (1e-1,1e-1), (1e-2,1e-6), (1e-6,1e-6)]
    --wd_linattn=[0.01, 0.01, 0.01, 0.3]
    --wd_plant=[0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01]
    --wd_mul_plant=[1.0, 1.0, 1.0, 1.0, 30.0]
    --static_inattn=5
    --diff_inattn=40
    --bs=8
    --save_model
    --fname=mimic4_icd10_clas_full
    --runs=1
    --plant=false
    --attn_init=(1,0,0)
    --infer=1
    --metrics=partial(precision_at_k, k=5); partial(precision_at_k, k=8); partial(precision_at_k, k=15)
    --trn_frm_cpt
    The following values were not passed to `accelerate launch` and had defaults used instead:
        `--num_machines` was set to a value of `1`
        `--dynamo_backend` was set to a value of `'no'`
    To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
    Rank[0] Run: 0; epochs: 6; lr: 0.01; bs: 8
    Training with running_decoder=False
    best so far = None                                       
    loss = 0.005077285226434469                                                                                              
    precision_at_k = 0.7713261286738703
    precision_at_k = 0.6909339965660034
    precision_at_k = 0.5442615224051447
    best so far = None

## Acknowledgement

This repository is my attempt to create Extreme Multi-Label Text
Classifiers using Language Model Fine-Tuning as proposed by [Jeremy
Howard](https://jeremy.fast.ai) and [Sebastian
Ruder](https://www.ruder.io) in
[ULMFit](https://arxiv.org/pdf/1801.06146v5.pdf). I am also heavily
influenced by the [fast.ai’s](https://fast.ai) course [Practical Deep
Learning for Coders](https://course.fast.ai/) and the excellent library
[fastai](https://github.com/fastai/fastai). I have adopted the style of
coding from [fastai](https://github.com/fastai/fastai) using the jupyter
based dev environment [nbdev](https://nbdev.fast.ai/). Since this is one
of my fast attempt to create a full fledged python library, I have at
times replicated implementations from fastai with some modifications. A
big thanks to Jeremy and his team from [fast.ai](https://fast.ai) for
everything they have been doing to make AI accessible to everyone.
