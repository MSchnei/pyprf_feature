#!/bin/sh

# directory:
directory="/home/marian/Documents/Git/py_pRF_motion/simulation"
# Go to input directory
cd ${directory}


echo "----- circleBar0 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_circleBar0.py
echo "----- Bar0 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Bar0.py
echo "----- Square0 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Square0.py
echo "----- Circle0 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Circle0.py

echo "----- circleBar1 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_circleBar1.py
echo "----- Bar1 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Bar1.py
echo "----- Square1 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Square1.py
echo "----- Circle1 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Circle1.py

echo "----- circleBar2 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_circleBar2.py
echo "----- Bar2 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Bar2.py
echo "----- Square2 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Square2.py
echo "----- Circle2 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/pRF_sim_config_Circle2.py
