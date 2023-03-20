# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/12_data.external.ipynb.

# %% ../../nbs/12_data.external.ipynb 2
from __future__ import annotations
from fastdownload import FastDownload
from functools import lru_cache
from ..imports import *
import xcube.data

# %% auto 0
__all__ = ['xcube_cfg', 'xcube_path', 'XURLs', 'untar_xxx']

# %% ../../nbs/12_data.external.ipynb 7
@lru_cache(maxsize=None)
def xcube_cfg() -> Config: # Config that contains default download paths for `data`, `model`, `storage` and `archive`
    "`Config` object for xcube's `config.ini`"
    return Config(Path(os.getenv('XCUBE_HOME', '~/.xcube')), 'config.ini', create=dict(
        data = 'data', archive = 'archive', storage = 'tmp', model = 'models'))

# %% ../../nbs/12_data.external.ipynb 10
def xcube_path(folder:str) -> Path: 
    "Local path to `folder` in `Config`"
    return xcube_cfg().path(folder)

# %% ../../nbs/12_data.external.ipynb 13
class XURLs():
    "Global cosntants for datasets and model URLs."
    LOCAL_PATH = Path.cwd()
    S3 = 'https://xcubebucket.s3.us-east-2.amazonaws.com/'
    
    #main datasets
    MIMIC3 = f'{S3}mimic3/mimic3.tgz'
    MIMIC3_L2R = f'{MIMIC3}mimic3_l2r/mimic3_l2r.tgz'
    
    def path(
        url:str='.', # File to download
        c_key:str='archive' # Key in `Config` where to save URL
    ) -> Path:
        "Local path where to download based on `c_key`"
        fname = url.split('/')[-1]
        local_path = XURLs.LOCAL_PATH/('models' if c_key=='model' else 'data')/fname
        if local_path.exists(): return local_path
        return xcube_path(c_key)/fname

# %% ../../nbs/12_data.external.ipynb 18
def untar_xxx(
    url:str, # File to download
    archive:Path=None, # Optional override for `Config`'s `archive` key
    data:Path=None, # Optional override for `Config`'s `data` key
    c_key:str='data', # Key in `Config` where to extract file
    force_download:bool=False, # Setting to `True` will overwrite any existing copy of data
    base:str='~/.xcube' # Directory containing config file and base of relative paths
) -> Path: # Path to extracted file(s)
    "Download `url` using `FastDownload.get`"
    d = FastDownload(xcube_cfg(), module=xcube.data, archive=archive, data=data, base=base)
    return d.get(url, force=force_download, extract_key=c_key)
