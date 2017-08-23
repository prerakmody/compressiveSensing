## Running Deeplearning libs on a Windows laptop
1. Install Cuda.
2. Install CUdnn (you'll have to register on the developer.nvidia.com website for this purpose)
    i. Use this [link](https://developer.nvidia.com/rdp/cudnn-download)
2. Check if your graphics driver is compatible with CUDA (use this link: https://sourceforge.net/projects/cuda-z/files/cuda-z/0.10/CUDA-Z-0.10.251-64bit.exe/download)
3. Run `nvcc --version` in the command line to check if everything was installed properly
        i. For Ubuntu - sudo apt-get install nvidia-cuda-toolkit
4. To check the installation path `which nvcc`


## Using Python3.5 via conda envs
1. Open Windows GitBash as a "normal" user
2. Run these commands
3. "conda create --name py3.5 python=3.5" 	(OR conda create --name py3.5 python=3.5 anaconda)
4. "source activate py3.5"
5. "python --version"  						(to check the version)
6. "conda info --envs"  						(this may not work, only seems to work with Admin Git Bash)

## Using python3.5 after installation
1. Open Git Bash as a normal user
2. "conda info --envs"                        (to check if your created env exists)
3. "source activate py3.5"

# INSTALLING VARIOUS LIBRARIES
## NUMPY + MKL problems
1. Sometimes importing NUMP/SCIPY leads to some issues since we are using a conda-env.
2. Simply uninstall exiting numpy / scipy installation (sometimes you have to try this twice) 
3. Then reinstall (numpy/scipy) from the UCI website (http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
4. You could also download cv2 from there
    i. For ubuntu following this [link](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/)

## LIBRARIES FOR DEEP LEARNING
1. pip install --upgrade tensorflow-gpu
2. pip install keras
3. pip install jupyter
4. pip install h5py

## Widgets for Ipython
0. https://github.com/ipython-contrib/jupyter_contrib_nbextensions
1. pip install jupyter_contrib_nbextensions
2. jupyter contrib nbextension install --user
3. jupyter notebook --generate-config

## Other libraries
1. pip install joblib

## To run the Jupyer notebooks
1.  Open command prompt
2.  Navigate to your folder
3.  Type `jupyter notebook` (do "source activate py3.5" if you wish for the python version to be different from system default)

## Removing conda envs
1. conda remove --name py3.5 --all

## INSTALLING OPEN AI 
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










