# -*- coding: utf-8 -*-

"""
hskl

sklearn models module

***Refactoring placeholder***

list and organize models from sklearn

"""

from sklearn.utils import all_estimators

_sklearn_classifiers = dict(all_estimators(type_filter='classifier'))

_sklearn_regressors = dict(all_estimators(type_filter='regressor'))