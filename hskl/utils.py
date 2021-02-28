"""
hyperspectral-sklearn

Utilities Module
   
"""

# Author: Qian Cao

# License: BSD 3-Clause

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from spectral import principal_components
import h5py
from scipy.io import loadmat # for older .mat files

def overlay(X,Y,fig_params={},ax_params={}):
    
    # Y==0 corresponds to alpha=0
    alphas = (Y>0).astype(float)*0.6
    cmap = plt.get_cmap('Set3')
    norm = mpl.colors.Normalize(vmin=1,vmax=np.max(Y))
    RGBA = cmap(norm(Y))
    RGBA[:,:,3] = alphas
    
    fig = plt.figure(**fig_params)
    ax1 = plt.imshow(np.mean(X,axis=2), cmap='gray',interpolation='nearest',**ax_params)
    plt.axis('off')
    ax2 = plt.imshow(RGBA, interpolation='nearest', **ax_params)
    # ax2 = plt.imshow(Y, cmap='Set3',alpha=0.6,interpolation='nearest',
    #                  vmin=1, vmax=np.max(Y), **ax_params)
    plt.axis('off')
    
    return fig, ax1, ax2

# Power normalization
def normalize_channels(x):
    # Assume (X,Y,Channels)
    rms = np.sqrt(np.mean(x**2,axis=2))
    return x/rms[:,:,None]

# Principal Component Analysis Perserving Fixed Variance
def pca_fit(x):
    return principal_components(x)
    
def pca_apply(x,pc,variance=0.8):
    pc_reduced = pc.reduce(fraction=variance)
    return pc_reduced.transform(x), pc_reduced.eigenvalues

# Read MATLAB file
def read_mat(filepath,keyind=0):
    """MATLAB file reader
    filepath: path to .mat file
    keyind: array in the .mat file to retrieve
    """
    try:
        f = h5py.File(filepath,'r')
        if hasattr(keyind, '__iter__'): # keyind can be a list or integer
            mat = []
            for kk in f.keys():
                mat.append(f['/'+kk].value)
        else:
            mat = f['/'+list(f.keys())[0]][()]
        return mat
    except OSError:
        print("Unable to open file, trying legacy matlab file loader.")
        
        try:
            fdict = loadmat(filepath)
            
            fdictkeys = list(fdict.keys()) 
            for key in fdictkeys: 
                if key.startswith("__"): 
                    fdict.pop(key)
            
            fdictkeys = list(fdict.keys()) 
            mat = fdict[fdictkeys[keyind]]
            
            mat = np.swapaxes(mat,0,2) # compatibility with h5py-loaded array
            
            return mat
        except:
            print(".mat file not readable using h5py or scipy.")

if __name__ == "__main__":
    pass