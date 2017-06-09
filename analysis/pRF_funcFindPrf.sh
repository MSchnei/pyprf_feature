#!/bin/sh

# directory:
directory="/media/sf_D_DRIVE/MotionLocaliser/Tools/P02/pRF_Motion"
# Go to input directory
cd ${directory}

echo "----- cross validation AoM-----"
python pRF_main.py /media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/FitResults/6runs_MotionAoMXval/pRF_config.py
echo "----- cross validation DoM-----"
python pRF_main.py /media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/FitResults/6runs_MotionDoMXval/pRF_config.py
