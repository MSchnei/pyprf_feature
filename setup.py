"""
pyprf_feature setup.

For development installation:
    pip install -e /path/to/pyprf_feature
"""

from setuptools import setup

with open('README.rst') as f:
    long_description = f.read()

setup(name='pyprf_feature',
      version='1.1.2',
      description=('A free & open source package for finding best-fitting \
                    population receptive field (PRF) models and feature \
                    weights for fMRI data.'),
      url='https://github.com/MSchnei/pyprf_feature',
      author='Marian Schneider, Ingo Marquardt',
      author_email='marian.schneider@maastrichtuniversity.nl',
      license='GNU General Public License Version 3',
      install_requires=['numpy', 'scipy', 'nibabel',
                        'cython==0.27.1', 'tensorflow==1.3.0',
                        'scikit-learn==0.19.1'],
      keywords=['pRF', 'fMRI', 'retinotopy', 'feature weights'],
      long_description=long_description,
      packages=['pyprf_feature.analysis'],
      py_modules=['pyprf_feature.analysis'],
      entry_points={
          'console_scripts': [
              'pyprf_feature = pyprf_feature.analysis.__main__:main',
              'pyprf_opt_brute = pyprf_feature.analysis.pyprf_opt_ep:main',
              'pyprf_sim = pyprf_feature.analysis.pyprf_sim_ep:main',
              ]},
      )
