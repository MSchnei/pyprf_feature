#!/bin/sh

# directory:
directory="/home/marian/Documents/Git/py_pRF_motion/simulation"
# Go to input directory
cd ${directory}

echo "----- circleBar1 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_circleBar1.py
echo "----- Bar1 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Bar1.py
echo "----- circleBar0 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_circleBar0.py
echo "----- Bar0 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Bar0.py
