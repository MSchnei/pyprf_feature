#!/bin/sh

# directory:
directory="/media/sf_D_DRIVE/MotionLocaliser/Tools/P02/pRF_Motion"
# Go to input directory
cd ${directory}

echo "----- cross validation -----"
python pRF_calcTstR2Betas_bstMdl.py /media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/FitResults/6runs_MotionAoMNoXval/pRF_config.py
echo "----- no cross validation -----"
python pRF_calcTstR2Betas_bstMdl.py /media/sf_D_DRIVE/MotionLocaliser/Analysis/P02/FitResults/6runs_MotionDoMNoXval/pRF_config.py
