# -*- coding: utf-8 -*-

"""
hskl

Regression Module

Data I/O conventions
X: (DimX, DimY, NumSpectralChannels) <np.array>
Y: (DimX, DimY) <np.array,dtype=uint8>
   0 - ignored
   1 - first label
   2 - second label
   ...
   
"""

# Author: Qian Cao

# License: BSD 3-Clause

import numpy as np

from sklearn.utils import all_estimators
from sklearn.base import BaseEstimator, RegressorMixin
from .base import flatten_with_label, HyperspectralMixin
from sklearn.metrics import accuracy_score

_sklearn_methods = dict(all_estimators(type_filter='regressor'))

_methods_list = list(_sklearn_methods.keys())

_methods = _sklearn_methods

def list_sklearn_methods():
    """List Available Regressor Methods in Scikit-Learn
    
    Returns
    -------
    list of regressor names

    """
    return list(_sklearn_methods.keys())

def list_methods():
    """List Available Regressor Methods Tested for Hyperspectral-sklearn
    
    Returns
    -------
    list of regressor names

    """
    return list(_sklearn_methods.keys())

class HyperspectralRegressor(BaseEstimator, HyperspectralMixin, RegressorMixin):
    """Hyperspectral Regressor
    
    Wrapper of regressors in sklearn
    
    Parameters
    ----------
    method_name : str, default='RandomForest'
        Invokes correponding classification algorithm in scikit-learn.
    method_param : dict, default={}
        A parameter used for demonstation of how to pass and store paramters.

    """
    
    _estimator_type = "regressor"
    
    def __init__(self, method_name='RandomForest', method_params={}):
        
        self.__dict__.update(method_params)
        
        if (method_name in _methods):
            self.est = _methods[method_name](**method_params)
        else:   
            raise KeyError(" ".join(["Method ", method_name, " not found. \
                         Use list_methods() to view available regressors."]))
                         
    def fit(self, X, Y, sample_fraction=None):
        """ Fit model to inputs and labels.
        """
        self.est = self._fit(self.est, X, Y, sample_fraction)
        
        return self

    def predict(self, X):
        """ Use model to predict input labels.
        """
        Y = self._predict(self.est, X)
        
        return Y
    
    def score(self, X, Y):
        """ Evaluate Regression accuracy
        Adapted from Regressor Mixin
        """
        sample_weight = None # TODO: Implement sample weighting
        
        X, Y = flatten_with_label(X, Y)
        return super().score(X, Y, sample_weight)
    
    def _more_tags(self):
        return {'requires_y': True}

if __name__ == "__main__":
    pass
