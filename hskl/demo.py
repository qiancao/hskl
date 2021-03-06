# -*- coding: utf-8 -*-

"""
hskl

Demo Module

* sample data

Useful links:
    
    http://lesun.weebly.com/hyperspectral-data-set.html
    https://www.kaggle.com/c/ipsa-ma511/overview

"""

# Author: Qian Cao

# License: BSD 3-Clause

import requests
from zipfile import ZipFile
import os

from tqdm import tqdm

from skimage import io

def dl_url(url, path, chunk_size=128):
    """ 
    Download dataset.
    
    Parameters
    ----------
    url : string
        Location of dataset in URL.
    path : string
        Location where the dataset is saved.
    chunk_size : int, optional
        Chunk size for download. The default is 128.

    Returns
    -------
    None.

    """
    
    r = requests.get(url, stream=True)
    total_size_in_bytes = int(r.headers.get('content-length'))
    
    with open(path, 'wb') as f:
        
        # Note: tqdm does not work well with print.
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', 
                            unit_scale=True, position=0, leave=True)
        
        for chunk in r.iter_content(chunk_size=chunk_size):
            # percent = int(count * chunk_size * 100 / total_size_in_bytes)
            progress_bar.update(len(chunk))
            f.write(chunk)            
            
        progress_bar.close()
            
def unzip(source_zip, target_path):
    """
    

    Parameters
    ----------
    source_zip : string
        Source zip file.
    target_path : string
        Location to be extracted.

    Returns
    -------
    None.

    """
    with ZipFile(source_zip, 'r') as zip: 
        zip.printdir()  
        zip.extractall() 
        print('Extraction complete.')

def dl_hyrank(path):
    """ 
    HyRANK Dataset
    
    https://www2.isprs.org/commissions/comm3/wg4/hyrank/
    
    Warning: This dataset is >400 MB.

    Returns
    -------
    h5py path containing trainng and test data arrays

    """
    
    url = "https://zenodo.org/record/1222202/files/HyRANK_satellite.zip?download=1"
    fn = "HyRANK_satellite.zip"
    
    path_fn = os.path.join(path,fn)
    dl_url(url, path_fn, chunk_size=128)

    unzip(path_fn, path)
    
def load_hyrank(path):
    """
    Load HyRANK dataset

    Parameters
    ----------
    path : string
        Directory containining the extracted hyrank dataset folder ("PATH/HyRANK_satellite")

    Returns
    -------
    List of training and validation image arrays

    """
    
    # Image names
    hyrank_dirname = "HyRANK_satellite"
    train_dirname = "TrainingSet"
    validation_dirname = "ValidationSet"
    
    train_fn_list = ["Dioni.tif","Loukia.tif"]
    train_label_fn_list = ["Dioni_GT.tif","Loukia_GT.tif"]
    validation_fn_list = ["Erato.tif","Kirki.tif","Nefeli.tif"]
    
    # Read TIF images as numpy arrays and append to list
    train = []
    train_label = []
    validation = []
    
    for ind, img_fn in enumerate(train_fn_list):
        
        train.append(io.imread(os.path.join(hyrank_dirname,
                                            train_dirname,
                                            img_fn)))
        
        train_label.append(io.imread(os.path.join(hyrank_dirname,
                                            train_dirname,
                                            train_label_fn_list[ind])))
    
    for ind, img_fn in enumerate(validation_fn_list):
        validation.append(io.imread(os.path.join(hyrank_dirname,
                                                 validation_dirname,
                                                 img_fn)))
    
    return train, train_label, validation