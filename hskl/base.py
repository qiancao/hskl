# -*- coding: utf-8 -*-

"""
hyperspectral-sklearn

Base Module

I/O Convention:
    Input array, output array
    Input list, output list (except flatten_with_la)
   
"""

# Author: Qian Cao

# License: BSD 3-Clause

import numpy as np
from sklearn.utils.validation import check_is_fitted
from multimethod import multimethod

# Integer label to ignore in label image
LABEL_IGNORE = 0

@multimethod
def shapeXYZ(X: np.ndarray):
    """ Returns XYZ dimensions as tuple for use in unflatten
    """
    return X.shape

@multimethod
def shapeXYZ(Xs: list):
    """ Returns XY dimensions as tuple for use in unflatten
    """
    Xshape_list = []
    for ind in range(len(Xs)):
        Xshape_list.append(shapeXYZ(Xs[ind]))
    return Xshape_list

@multimethod
def flatten3(X: np.ndarray):
    """ Flattens a single image <np.ndarray> (3D to 2D)
    Returns original array if already flattened
    """
    if X.ndim == 3: # assumes (DimX, DimY, Channels)
        s = X.shape
        Y = np.reshape(X, newshape=(s[0]*s[1],s[2]), order='C')
    elif X.ndim == 2: # image considered to be already flat, ignore
        Y = X # this should be (n_samples, n_features)
    else:
        raise ValueError('X.ndim must for 3 or 2 (already flat)')
    return Y

@multimethod
def flatten2(X: np.ndarray):
    """ Flattens a single LABEL image <np.ndarray> (2D to 1D)
    Returns original array if already flat
    """
    if X.ndim == 2: # assumes (DimX, DimY, Channels)
        s = X.shape
        Y = np.reshape(X, newshape=(s[0]*s[1]), order='C')
    elif X.ndim == 1: # image considered to be already flat, ignore
        Y = X # this should be (n_samples, n_features)
    else:
        raise ValueError('X.ndim must for 2 or 1 (already flat)')
    return Y

@multimethod
def flatten2(X: np.ndarray, Y: np.ndarray):
    """ Flattens a single LABEL image <np.ndarray> (2D to 1D) with MASK Y
    Returns original array if already flat
    """
    
    X, Y = flatten2(X), flatten2(Y)
    
    ind = np.squeeze(Y!=LABEL_IGNORE)
    X = X[np.squeeze(Y!=LABEL_IGNORE)]
    return X

@multimethod
def flatten3(X: np.ndarray, Y: np.ndarray):
    """ Flattens image with mask defined by Y (3D to 2D)
    Returns masked array if already flat
    """
    if X.ndim == 3 and Y.ndim == 3: # assumes (DimX, DimY, Channels)
        X, Y = flatten3(X), flatten2(Y)
        X = X[np.squeeze(Y!=LABEL_IGNORE),:]
    elif X.ndim == 2 and Y.ndim == 1 and X.shape[0] == Y.shape[0]:
        X = X[np.squeeze(Y!=LABEL_IGNORE),:] # images already flattened
    else:
        raise ValueError('(X.ndim, Y.ndim) must for (3,2) or (2,1)')
    return X

@multimethod
def flatten3(Xs: list):
    """ Flattens list of images
    Applies flatten() to each element in the list
    """
    n_images_ = len(Xs)
    n_features_list = []
    images_flattened = []
    
    for ind in range(n_images_):
        X = flatten3(Xs[ind])
        images_flattened.append(X)
        n_features_list.append(X.shape[1])
    
    if not len(set(n_features_list))==1:
        raise ValueError("All arrays in input list must be have the same number of features.")
        
    return np.concatenate(images_flattened,axis=0)

@multimethod
def flatten2(Xs: list):
    """ Flattens list of LABEL images
    Applies flatten() to each element in the list
    """
    n_images_ = len(Xs)
    n_features_list = []
    images_flattened = []
    
    for ind in range(n_images_):
        X = flatten2(Xs[ind])
        images_flattened.append(X)
        n_features_list.append(X.shape[1])
    
    if not len(set(n_features_list))==1:
        raise ValueError("All arrays in input list must be have the same number of features.")
        
    return np.concatenate(images_flattened,axis=0)

@multimethod
def flatten_with_label(X: np.ndarray, Y: np.ndarray):
    """ Flattens image X with label Y
    """
    X, Y = flatten3(X), flatten2(Y)
    X = X[np.squeeze(Y!=LABEL_IGNORE),:]
    Y = Y[Y!=LABEL_IGNORE]
    
    return X, Y
    
@multimethod
def flatten_with_label(Xs: list, Ys: list):
    """ Flattens list of images X with list of label images Y
    """
    n_images_ = len(Xs)
    
    n_features_list = []
    X_flattened = []
    Y_flattened = []
    
    for ind in range(n_images_):
        X, Y = flatten_with_label(Xs[ind], Ys[ind])
        n_features_list.append(X.shape[1])
        X_flattened.append(X)
        Y_flattened.append(Y)
        
    if not len(set(n_features_list))==1:
        raise ValueError("All arrays in input list must be have the same number of features.")
        
    X = np.concatenate(X_flattened,axis=0)
    Y = np.concatenate(Y_flattened,axis=0)
    
    return X, Y

def sample_fraction_random(X, Y, frac):
    """ Samples a random fraction of rows from sklearn matrix for training
    """
    Ntotal = X.shape[0]
    Nselect = int(Ntotal*frac)
    indices = np.random.choice(Ntotal,Nselect,replace=False)
    
    X = X[indices,:]
    Y = Y[indices]
    
    return X, Y

@multimethod
def unflatten(X: np.ndarray, s: tuple):
    """ Unflattens image with shape defined by tuple s (1D to 2D)
    s denotes dimensions of the *INPUT* image
    len(s) == 3 : reshape to 2D label image
    len(s) == 2 : input is flattened image, ignore.
    """
    if len(s) == 3: # input array was hyperspectral image, X is 2D
        Y = np.squeeze(np.reshape(X, newshape=(s[0], s[1]), order='C'))
    elif len(s) == 2: # input array was sklearn matrix, no need to reshape, X is 1D
        Y = X
    else:
        raise ValueError("Incompatible dimension")
    return Y

@multimethod
def unflatten(X: np.ndarray, shapes: list):
    """ Unflattens images with shape defined by list of tuples s
    X is an array (1D), unflattened to 2D
    s denotes dimensions of the *INPUT* image
    len(s) == 3 : reshape to 2D label image
    len(s) == 2 : input is flattened image, ignore.
    """
    n_images_ = len(shapes)
    n_samples_list = [] # Number of sampled pixel/spectra for each image
    
    for ind in range(n_images_):
        s = shapes[ind]
        if len(s) == 3:
            n_samples_list.append(s[0]*s[1]) # image pixel list
        elif len(s) == 2:
            n_samples_list.append(s[0])
            
    X_list = []
    
    sinds = np.cumsum(np.array(n_samples_list)) # sample indices
    sinds = np.concatenate(([0],sinds))
    
    for ind in range(n_images_):
        s = shapes[ind]
        sslice = slice(sinds[ind],sinds[ind+1]) # sample slice
        X_list.append(unflatten(X[sslice],s))
    
    return X_list

class HyperspectralMixin:
    
    @classmethod
    def _fit(cls, est, X, Y, sample_fraction=None):
        """ Fit hyperspectral image to estimator
        X and Y may be np.ndarrays or list of np.ndarrays
        handles both flattened and original sklearn formats
        """
        
        # For now, only supervised estimators are considered
        
        X, Y = flatten_with_label(X,Y)
        
        # fit on partial data
        if sample_fraction != None:
            X, Y = sample_fraction_random(X, Y, sample_fraction)
        
        cls.n_features_ = X.shape[1]
        
        est = est.fit(X,Y)
        est.is_fitted_ = True
        
        return est
    
    @classmethod
    def _predict(cls, est, X, Y=None):
        """ Predict using estimator
        X and Y may be np.ndarrays or list of np.ndarrays
        handles both flattened and original sklearn formats
        """
        
        # For now, prediction with mask is not supported
        # crop input X to achieve same result as output
        check_is_fitted(est, 'is_fitted_')
        
        Xshape = shapeXYZ(X) # tuples or list of tuples
        X = flatten3(X)
        X = est.predict(X)
        X = unflatten(X, Xshape) # only Xshape[0] and Xshape[1] are used.
        
        return X