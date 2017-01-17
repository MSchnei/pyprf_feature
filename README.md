# py_pRF_motion
A free & open python tool for finding best-fitting pRF models and motion parameters for fMRI data.

## Dependencies

## Dependencies
[**Python 2.7**](https://www.python.org/download/releases/2.7/)

| Package                              | Tested version |
|--------------------------------------|----------------|
| [NumPy](http://www.numpy.org/)       | ?              |
| [SciPy](http://www.scipy.org/)       | ?              |
| [NiBabel](http://nipy.org/nibabel/)  | ?              |

## How to use
Step 1: Record fMRI data
The PsychoPy scripts in the Stimulation folder can be used for presenting appropriate visual stimuli.

Step 2: Prepare the PNG presentation files
The "pRF_createPNGs.py" script in the PrePro takes the msk.npy and Conditions.pickle from the Stimulation folder and converts the presentation files to PNG images. The PNG images contain aperture information for every volume that was presented during the functional runs. Run this script, with the correct inputs, before you run the pRF_main.py script.

Step 3: Prepare the input data
The input data should be motion-corrected, high-pass filtered and demeaned. If desired, distortion correction and temporal as well as spatial smoothing can be applied.
The PrePro folder contains some auxiliary scripts to perfom some of these functions, using either fsl or python functions.

Step 4: Adjust the pRF_config
Adjust the inputs to the "pRF_config.py" file in the Analysis folder, such that the provided information is correct.

Step 5: Run the pRF_main.py script
Open a terminal, navigate to the Analysis folder, containing the "pRF_main.py" script and run "python pRF_main.py". If desired, a custom made pRF_config.py script can additionally be provided by running "python pRF_main.py path/to/custom_config.py". If no custom config script is provided, the pRF_main script will default to the pRF_config.py file in the Analysis folder.

## References
This application is based on the following work:

...


## License

The project is licensed under [GNU Geneal Public License Version 3](http://www.gnu.org/licenses/gpl.html).
