#!/bin/sh

#get pwd
cwd=$(pwd)

# Input directory
input_directory="/media/sf_D_DRIVE/PacMan/Analysis/P3/UndistortedMotCor/01_nii"

# Output directory:
output_directory="/media/sf_D_DRIVE/PacMan/Analysis/P3/UndistortedMotCor/02_highPass"

# Go to input directory
cd ${input_directory}

# Create list with paths to relevant files:
lst_tmp=( $(find . -type f -name '*.nii') )
echo "----- Images included are:"
echo ${lst_tmp[@]}

echo "----- high-pass filter each run:"

# sigmas in volumes, not seconds
# sigma[vol] = filter_width[secs]/(2*TR[secs])
# if cycles longer than 50s should be cut-off, and TR is 3s the following sigma should be input:
# 50/(2*3) = 8.3


# Go through all the runs
for index_1 in "${lst_tmp[@]}"; do
  echo "--------" ${index_1}
  echo "--------get mean"
  fslmaths ${index_1} -Tmean tempMean
  echo "--------filter and add mean back again"
  fslmaths ${index_1} -bptf 8.3 -1 -add tempMean ${index_1%*.nii}_hpf
  imrm tempMean
done

#move all files to new directory
mkdir -p ${output_directory}
find . | grep "r*_hpf*.nii.gz" | xargs mv ${output_directory}

#change back
cd $cwd

echo "----- Done -----"
