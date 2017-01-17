#!/bin/sh


#!/bin/sh

echo "----- Start mskBar -----"
python pRF_getBetas.py 0 0
echo "----- mskBar done -----"

echo "----- Start mskSquare -----"
python pRF_getBetas.py 1 0
echo "----- mskSquare done -----"

echo "----- Start mskCircleBar -----"
python pRF_getBetas.py 2 0
echo "----- mskCircleBar done -----"

echo "----- Start mskCircle -----"
python pRF_getBetas.py 3 0
echo "----- mskCircle done -----"


echo "----- Start mskBar -----"
python pRF_getBetas.py 0 1
echo "----- mskBar done -----"

echo "----- Start mskSquare -----"
python pRF_getBetas.py 1 1
echo "----- mskSquare done -----"

echo "----- Start mskCircleBar -----"
python pRF_getBetas.py 2 1
echo "----- mskCircleBar done -----"

echo "----- Start mskCircle -----"
python pRF_getBetas.py 3 1
echo "----- mskCircle done -----"


echo "----- Start mskBar -----"
python pRF_getBetas.py 0 2
echo "----- mskBar done -----"

echo "----- Start mskSquare -----"
python pRF_getBetas.py 1 2
echo "----- mskSquare done -----"

echo "----- Start mskCircleBar -----"
python pRF_getBetas.py 2 2
echo "----- mskCircleBar done -----"

echo "----- Start mskCircle -----"
python pRF_getBetas.py 3 2
echo "----- mskCircle done -----"
