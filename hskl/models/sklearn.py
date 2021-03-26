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

def get_args(f):
    # Get argument signature (e.g. get_args(_classifier['RandomForestClassifier']))
    return inspect.signature(f)