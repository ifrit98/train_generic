#!/usr/bin/python3
import os
from warnings import warn
from distutils.core import setup
from setuptools import find_packages, find_namespace_packages

REQ_FILE = 'requirements.txt'

if not os.path.exists(REQ_FILE):
      warn("No requirements file found.  Using defaults deps")
      deps = [
            'numpy', # 1.19.5
            'pandas', #  1.2.2, 1.1.5 
            'matplotlib',
            'scipy', 
            'seaborn',
            'scikit-learn',
            'soundfile',
            'tensorflow>=2.5.0',
            'pyyaml',
            'tensorflow_probability>=0.13.0',
            'pynvml',
            'ulid']
      warn(', '.join(deps))
else:
      with open(REQ_FILE, 'r') as f:
            deps = f.read().splitlines()


setup(name='train_generic',
      version='1.0.0',
      description='Abstract Train and Eval pipeline for rapid development',
      author='Jason St George',
      author_email='stgeorge@brsc.com',
      packages=find_packages(),
      install_requires=deps)

