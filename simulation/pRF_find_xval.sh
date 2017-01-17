#!/bin/sh

echo "----- Start mskCircleBar 0 -----"
python pRF_find_xval.py mskCircleBar 0 1
echo "----- mskCircleBar 0 done -----"

echo "----- Start mskSquare 0 -----"
python pRF_find_xval.py mskSquare 0 1
echo "----- mskSquare 0 done -----"

echo "----- Start mskBar 0 -----"
python pRF_find_xval.py mskBar 0 1
echo "----- mskBar 0 done -----"

echo "----- Start mskCircle 0 -----"
python pRF_find_xval.py mskCircle 0 1
echo "----- mskCircle 0 done -----"


echo "----- Start mskCircleBar 1 -----"
python pRF_find_xval.py mskCircleBar 1 0
echo "----- mskCircleBar 1 done -----"

echo "----- Start mskSquare 1 -----"
python pRF_find_xval.py mskSquare 1 0
echo "----- mskSquare 1 done -----"

echo "----- Start mskBar 1 -----"
python pRF_find_xval.py mskBar 1 0
echo "----- mskBar 1 done -----"

echo "----- Start mskCircle 1 -----"
python pRF_find_xval.py mskCircle 1 0
echo "----- mskCircle 1 done -----"


echo "----- Start mskCircleBar 2 -----"
python pRF_find_xval.py mskCircleBar 2 0
echo "----- mskCircleBar 2 done -----"

echo "----- Start mskSquare 2 -----"
python pRF_find_xval.py mskSquare 2 0
echo "----- mskSquare 2 done -----"

echo "----- Start mskBar 2 -----"
python pRF_find_xval.py mskBar 2 0
echo "----- mskBar 2 done -----"

echo "----- Start mskCircle 2 -----"
python pRF_find_xval.py mskCircle 2 0
echo "----- mskCircle 2 done -----"

