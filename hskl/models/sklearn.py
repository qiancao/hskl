# -*- coding: utf-8 -*-

"""
hskl

sklearn models module

***Refactoring placeholder***

list and organize models from sklearn

"""

from sklearn.utils import all_estimators
import inspect # retrieve estimator arguments

_classifier = dict(all_estimators(type_filter='classifier'))
_regressor = dict(all_estimators(type_filter='regressor'))
_cluster = dict(all_estimators(type_filter='cluster'))
_transformer = dict(all_estimators(type_filter='transformer'))

def get_params(f):
    """
    Returns parameters and default values to class constructor
    
    Example:
        get_params(_classifier['RandomForestClassifier'])
        >>
            {'n_estimators': 100,
             'criterion': 'gini',
             ...
             'ccp_alpha': 0.0,
             'max_samples': None}

    Parameters
    ----------
    f : class
        Scikit-learn estimator class

    Returns
    -------
    param_dict : dictionary
        dictionary of parameters and default values

    """
    sig = inspect.signature(f)
    param_dict = {key:value.default for (key,value) in dict(sig.parameters).items()}
    return param_dict