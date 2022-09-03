#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages


setup(name='train_utils',
      version='1.0',
      packages=find_packages(),
      description='Add pruning, train and test model', 
      install_requires=[
        'numpy', 
        'matplotlib',
        'tqdm',
        'torch',
        'scikit-learn'
    ]
     )
