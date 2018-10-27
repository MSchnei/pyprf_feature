[![DOI](https://zenodo.org/badge/78625137.svg)](https://zenodo.org/badge/latestdoi/78625137)

# pyprf_feature
<img src="logo/logo.png" width=200 align="right" />

A free & open source package for finding best-fitting population receptive field (PRF) models and feature weights for fMRI data.

If you are only interested in the spatial properties of the population receptive fields, not preferred features, check out the [pyprf package](https://github.com/ingo-m/pypRF).

## Installation

For installation, follow these steps:

0. (Optional) Create conda environment
```bash
conda create -n env_pyprf_feature python=2.7
source activate env_pyprf_feature
conda install pip
```

1. Clone repository
```bash
git clone https://github.com/MSchnei/pyprf_feature.git
```

2. Install numpy, e.g. by running:
```bash
pip install numpy
```

3. Install pyprf_feature with pip
```bash
pip install /path/to/cloned/pyprf_feature
```

## Dependencies
[**Python 2.7**](https://www.python.org/download/releases/2.7/)

| Package                              | Tested version |
|--------------------------------------|----------------|
| [NumPy](http://www.numpy.org/)       | 1.11.1         |
| [SciPy](http://www.scipy.org/)       | 0.18.0         |
| [NiBabel](http://nipy.org/nibabel/)  | 2.0.2          |

## How to use
### 1. Present stimuli and record fMRI data
The PsychoPy scripts in the Stimulation folder can be used for presenting appropriate visual stimuli.

### 2. Prepare spatial and temporal information for experiment as arrays
1. Run prepro_get_spat_info.py in the prepro folder to obtain an array with the spatial information of the experiment.
   This should result in a 3d numpy array with shape [pixel x pixel x nr of spatial aperture conditions] that represents
   images of the spatial apertures stacked on top of each other.

2. Run prepro_get_temp_info.py in the prepro folder to obtain an array with the temporal information of the experiment.
   This should result in a 2d numpy array with shape [nr of volumes across all runs x 4]. The first column represents
   unique identifiers of spatial aperture conditions. The second column represents onset times and the third durations
   (both in s).The fourth column represents unique feature identifiers.

### 3. Prepare the input data
The input data should be motion-corrected, high-pass filtered and (optionally) distortion-corrected.
If desired, spatial as well as temporal smoothing can be applied.
The PrePro folder contains some auxiliary scripts to perform some of these functions.

### 4. Adjust the csv file
Adjust the information in the config_default.csv file in the Analysis folder, such that the provided information is correct.
It is recommended to make a specific copy of the csv file for every subject.

### 5. Run pyprf_feature
Open a terminal and run
```
pyprf_feature -config path/to/custom_config.csv
```

## References
This application is based on the following work:

* Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. NeuroImage, 39(2), 647–660. https://doi.org/10.1016/j.neuroimage.2007.09.034

* Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu, B., & Gallant, J. L. (2011). Report Reconstructing Visual Experiences from Brain Activity Evoked by Natural Movies, 1641–1646. https://doi.org/10.1016/j.cub.2011.08.031

* St-Yves, G., & Naselaris, T. (2017). The feature-weighted receptive field: An interpretable encoding model for complex feature spaces. NeuroImage, (June), 1–15. https://doi.org/10.1016/j.neuroimage.2017.06.035

## License
The project is licensed under [GNU General Public License Version 3](http://www.gnu.org/licenses/gpl.html).
