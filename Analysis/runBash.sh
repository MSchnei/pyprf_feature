#!/bin/sh

# directory:
directory="/home/marian/Documents/Git/py_pRF_motion/Analysis"
# Go to input directory
cd ${directory}

echo "----- cross validation -----"
python pRF_main.py /home/marian/Documents/Git/py_pRF_motion/Analysis/pRF_config_xval.py
echo "----- no cross validation -----"
python pRF_main.py /home/marian/Documents/Git/py_pRF_motion/Analysis/pRF_config_noXval.py
