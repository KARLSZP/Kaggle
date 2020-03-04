# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 20:12:10 2019

@author: Karl
"""

import sklearn
from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

print(iris.data)