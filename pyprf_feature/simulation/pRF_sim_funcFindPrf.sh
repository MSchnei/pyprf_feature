#!/bin/sh

# directory:
directory="/home/marian/Documents/Git/py_pRF_motion/simulation"
# Go to input directory
cd ${directory}


echo "----- circleBar0 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_circleBar0.py
echo "----- Bar0 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_bar0.py
echo "----- Square0 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_square0.py
echo "----- Circle0 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_circle0.py

echo "----- circleBar1 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_circleBar1.py
echo "----- Bar1 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_bar1.py
echo "----- Square1 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_square1.py
echo "----- Circle1 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_circle1.py

echo "----- circleBar2 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_circleBar2.py
echo "----- Bar2 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_bar2.py
echo "----- Square2 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_square2.py
echo "----- Circle2 -----"
python pRF_sim_main.py /home/marian/Documents/Git/py_pRF_motion/simulation/configs/pRF_sim_config_circle2.py
