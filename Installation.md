## Running Deeplearning libs on a Windows laptop
1. Install Cuda.
2. Install CUdnn (you'll have to register on the developer.nvidia.com website for this purpose)
2. Check if your graphic driver is compatible with CUDA (use this link: https://sourceforge.net/projects/cuda-z/files/cuda-z/0.10/CUDA-Z-0.10.251-64bit.exe/download)
3. Run `nvcc --version` to check if everything was installed properly
4. To check the installation path `whih nvcc	`


## Using Python3.5 via conda envs
1. Open GitBash as a "normal" user
2. Run these commands
3. conda create --name py3.5 python=3.5 	(OR conda create --name py3.5 python=3.5 anaconda)
4. source activate py3.5
5. python --version  						(to check the version)
6. conda info --envs  						(this may not work, only seems to work with Admin Git Bash)
7. pip install jupyter
8. pip install --upgrade tensorflow-gpu
9. pip install keras (this might fail and you may havve to install numpy/scipy. Check last point)

## Using python3.5 after installation
1. Open Git Bash as a normal user
2. conda info --envs to check if your created env exists
2. Type "source activate py3.5
3. Type "jupyter notebook" in your directory of interest

## INSTALLING OPEN AI (make sure you are in the evnv by typing - source active {env_name} ) 
1. https://github.com/openai/gym
2. pip install 'gym[all]'
3. How to Install mujoco
4. pip install mujoco-py
5. URL : https://github.com/openai/mujoco-py/
6. Install the Win64 version.
7. Check if you have the env paths defined - echo $MUJOCO_PY_MJKEY_PATH  || echo $MUJOCO_PY_MJPRO_PATH
8. How to import mujoco - import mujoco_py
9. Test the simulate.exe file. It will be unable to export the mjkey.txt file. 
10. Add platname = "win" to platname_targdir.py
11. Use the .dll file in mjcore.py

## To run the Jupyer notebooks
1.  Open command prompt
2.  Navigate to your folder
3.  Type `jupyter notebook` (do "source activate py3.5" if you wish for the python version to be different from system default)

Widgets for Ipython
0. https://github.com/ipython-contrib/jupyter_contrib_nbextensions
1. pip install jupyter_contrib_nbextensions
2. jupyter contrib nbextension install --user


## Removing conda envs
1. conda remove --name py3.5 --all


## NUMPY + MKL problems
1. Sometimes importing scipy.sparse leads to some issues since we are using a conda-env.
2. Simply uninstall (sometimes you have to try this twice as well), 
3. Then reinstall (numoy/scipt) from the UCI website (http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
4. You could also download cv2 from there
