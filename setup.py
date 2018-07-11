"""
pyprf_feature setup.

For development installation:
    pip install -e /path/to/pyprf_feature
"""

import numpy as np
from setuptools import setup, Extension

setup(name='pyprf_feature',
      version='1.0.0',
      description=('A free & open source package for finding best-fitting \
                    population receptive field (PRF) models and feature \
                    weights for fMRI data.'),
      url='https://github.com/MSchnei/pyprf_feature',
      author='Marian Schneider, Ingo Marquardt',
      author_email='marian.schneider@maastrichtuniversity.nl',
      license='GNU General Public License Version 3',
      install_requires=['numpy', 'scipy', 'nibabel',
                        'cython==0.27.1', 'tensorflow==1.4.0',
                        'scikit-learn==0.19.1'],
      keywords=['pRF', 'fMRI', 'retinotopy', 'feauture weights'],
      packages=['pyprf_feature.analysis'],
      py_modules=['pyprf_feature.analysis'],
      entry_points={
          'console_scripts': [
              'pyprf_feature = pyprf_feature.analysis.__main__:main',
              ]},
      ext_modules=[Extension('pyprf_feature.analysis.cython_leastsquares',
                             ['pyprf_feature/analysis/cython_leastsquares.c'],
                             include_dirs=[np.get_include()]
                             )],
      )
