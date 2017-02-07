#!/bin/sh


#!/bin/sh

# simulate
echo "----- Start mskCircleBar -----"
python pRF_simulateTC_xval.py mskCircleBar 1
echo "----- mskCircleBar done -----"

#echo "----- Start mskCircle -----"
#python pRF_simulateTC_xval.py mskCircle 0
#echo "----- mskCircle done -----"

#echo "----- Start mskSquare -----"
#python pRF_simulateTC_xval.py mskSquare 0
#echo "----- mskSquare done -----"

echo "----- Start mskBar -----"
python pRF_simulateTC_xval.py mskBar 0
echo "----- mskBar done -----"

