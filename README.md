```
#|hide
#| eval: false
! [ -e /content ] && pip install -q condacolab && python3 -c  "import condacolab; condacolab.install()" # create conda environment in colab
! [ -e /content ] && pip install -Uqq xcube  # upgrade xcube on colab
```


```
#| hide
from xcube.text.all import *

%load_ext autoreload
%autoreload 2
```

# xcube

> xcube trains and explains XMTC models

## E**X**plainable E**X**treme Multi-Label Te**X**t Classification:

-   *What is XMTC?* Extreme Multi-Label Text Classification (XMTC) addresses the problem of automatically assigning each data point with most relevant subset of labels from an extremely large label set. One major application of XMTC is in the global healthcare system, specifically in the context of the International Classification of Diseases (ICD). ICD coding is the process of assigning codes representing diagnoses and procedures performed during a patient visit using clinical notes documented by health professionals.

-   *Datasets?* Examples of ICD coding dataset: [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [MIMIC-IV](https://physionet.org/content/mimic-iv-note/2.2/). Please note that you need to be a credentialated user and complete a training to acces the data.

-   *What is xcube?* xcube trains and explains XMTC models using LLM fine-tuning.

## Install

- Create new conda environment:
```sh
conda create -n xxx python=3.10
```
```sh
conda activate xxx
```

- Install PyTorch with cuda enabled: [Optional]
```sh 
conda search pytorch
```
<img alt="output of conda search pytorch" width="400" src="pics/pytorch.png" caption="Pictorial representation of mutual information gain" id="img_mut_info">

use the build string that matches the python and cuda version, replacing the pytorch version and build string appropriately:
```sh
conda install pytorch=2.0.0=cuda118py310h072bc4c pytorch-cuda=11.8 -c pytorch -c nvidia
```

Update cuda-toolkit:
```sh
sudo apt install nvidia-cuda-toolkit
```

Verify cuda is available:  Run `python` and `import torch; torch.cuda.is_available()`


- Install using:
```sh
pip install xcube
```
Configure accelerate by:
```sh
accelerate config
```

## How to use

You can either clone the repo and open it in your own machine. Or if you don't want to setup a python development environment, an even easier and quicker approach is to open this repo using [Google Colab](https://colab.research.google.com/). You can open this readme page in Colab using this [link](https://colab.research.google.com/github/debjyotiSRoy/xcube/blob/plant/nbs/index.ipynb).


```
IN_COLAB = is_colab()
```


```
source_mimic3 = untar_xxx(XURLs.MIMIC3_DEMO)
source_mimic4 = untar_xxx(XURLs.MIMIC4)
path = Path.cwd().parent/f"{'xcube' if IN_COLAB else ''}" # root of the repo
(path/'tmp/models').mkdir(exist_ok=True, parents=True)
tmp = path/'tmp'
```

Check your GPU memory! If you are running this on google colab be sure to turn on the GPU runtime. You should be able to train and infer all the models with atleast 16GB of memory. However, note that training the full versions of the datasets from scratch requires atleast 48GB memory.


```
cudamem()
```

### Train and Infer on MIMIC3-rare50

MIMIC3-rare50 refers to a split of [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) that contains the 50 most rare codes (Refer to [Knowledge Injected Prompt Based Fine-Tuning for Multi-label Few-shot ICD Coding](https://aclanthology.org/2022.findings-emnlp.127/) for split creation).


```
data = join_path_file('mimic3-9k_rare50', source_mimic3, ext='.csv')
!head -n 1 {data}
```


```
df = pd.read_csv(data,
                 header=0,
                 names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid', 'split'],
                 dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool, 'split': str})
df.head(2)
```

To launch the training of an XMTC model on MIMIC3-rare50:


```
os.chdir(path/'scripts')
!./run_scripts.sh --script_list_file script_list_mimic3_rare50train
```

### Train and Infer on MIMIC3-top50

MIMIC3-top50 refers to a split of [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) that contains 50 most frequent codes (Refer to [Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) for split creation)


```
data = join_path_file('mimic3-9k_top50', source_mimic3, ext='.csv')
!head -n 1 {data}
```


```
df = pd.read_csv(data,
                 header=0,
                 names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid', 'split'],
                 dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool, 'split': str})
df.head(2)
```

To infer one our pretrained XMTC models on MIMIC3-top50 (Metrics for inference - Precision@3,5,8,15):


```
model_fnames = L(source_mimic3.glob("**/*top50*.pth")).map(str)
print('\n'.join(model_fnames))
fname = Path(shutil.copy(model_fnames[2], tmp/'models')).name.split('.')[0]
print(f"We are going to infer model {fname}.")
```


```
os.chdir(path/'scripts')
!./launches/launch_top50_mimic3 --fname {fname} --no_running_decoder --infer 1
```

To launch the training of an XMTC model on MIMIC3-top50 from scratch:


```
#| eval: false
os.chdir(path/'scripts')
!./run_scripts.sh --script_list_file script_list_mimic3_top50train
```

### Train and Infer on MIMIC3-full:

MIMIC3-full refers to the full [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) dataset. (Refer to [Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) for details of how the data was curated)


```
data = join_path_file('mimic3-9k_full', source_mimic3, ext='.csv')
!head -n 1 {data}
```


```
df = pd.read_csv(data,
                 header=0,
                 names=['subject_id', 'hadm_id', 'text', 'labels', 'length', 'is_valid', 'split'],
                 dtype={'subject_id': str, 'hadm_id': str, 'text': str, 'labels': str, 'length': np.int64, 'is_valid': bool, 'split': str})
df.head(2)
```

Lets's look at some of the ICD9 codes description:


```
des = load_pickle(source_mimic3/'code_desc.pkl')
lbl_dict = dict()
for lbl in df.labels[1].split(';'):
    lbl_dict[lbl] = des.get(lbl, 'NF')
pd.DataFrame(lbl_dict.items(), columns=['icd9_code', 'desccription'])
```

To infer one our pretrained XMTC models on MIMIC3-full (Metrics for inference - Precision@3,5,8,15):


```
model_fnames = L(source_mimic3.glob("**/*full*.pth")).map(str)
print('\n'.join(model_fnames))
fname = Path(shutil.copy(model_fnames[0], tmp/'models')).name.split('.')[0]
print(f"Let's infer the pretrained model {fname}.")
```


```
os.chdir(path/'scripts')
!./launches/launch_complete_mimic3 --fname {fname} --infer 1 --no_running_decoder
```

### Train and Infer on MIMIC4-full:

MIMIC4-full refers to the full [MIMIC-IV](https://physionet.org/content/mimiciii/1.4/) dataset using ICD10 codes. (Refer to [Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study](https://arxiv.org/pdf/2304.10909.pdf) for details of how the data was curated)


```
data = join_path_file('mimic4_icd10_full', source_mimic4, ext='.csv')
!head -n 1 {data}
```


```
df = pd.read_csv(data,
                    header=0,
                    usecols=['subject_id', '_id', 'text', 'labels', 'num_targets', 'is_valid', 'split'],
                    dtype={'subject_id': str, '_id': str, 'text': str, 'labels': str, 'num_targets': np.int64, 'is_valid': bool, 'split': str})
df.head(2)
```

Let's look at some of the descriptions of ICD10 codes: 



```
stripped_codes = [''.join(filter(str.isalnum, s)) for s in df.labels[0].split(';')]
desc = get_description(stripped_codes)
pd.DataFrame(desc.items(), columns=['icd10_code', 'desccription'])
```

To infer one our pretrained XMTC models on MIMIC4-full (Metrics for inference - Precision@5,8,15):


```
print('\n'.join(L(source_mimic4.glob("**/*full*.pth")).map(str)))
model_fname = Path('/home/deb/.xcube/data/mimic4/mimic4_icd10_clas_full.pth')
fname = Path(shutil.copy(model_fname, tmp/'models')).name.split('.')[0]
print(f"Let's infer the pretrained model {fname}.")
```


```
#| eval: false
os.chdir(path/'scripts')
!./launches/launch_complete_mimic4_icd10 --fname mimic4_icd10_clas_full --no_running_decoder --infer 1
```

## Acknowledgement

This repository is my attempt to create Extreme Multi-Label Text Classifiers using Language Model Fine-Tuning as proposed by [Jeremy Howard](https://jeremy.fast.ai) and [Sebastian Ruder](https://www.ruder.io) in [ULMFit](https://arxiv.org/pdf/1801.06146v5.pdf). I am also heavily influenced by the [fast.ai's](https://fast.ai) course [Practical Deep Learning for Coders](https://course.fast.ai/) and the excellent library [fastai](https://github.com/fastai/fastai). I have adopted the style of coding from [fastai](https://github.com/fastai/fastai) using the jupyter based dev environment [nbdev](https://nbdev.fast.ai/). Since this is one of my fast attempt to create a full fledged python library, I have at times replicated implementations from fastai with some modifications. A big thanks to Jeremy and his team from [fast.ai](https://fast.ai) for everything they have been doing to make AI accessible to everyone.
