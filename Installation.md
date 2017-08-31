## Installing NVIDIA packages
1. Install Cuda.
    * Linux
        * Refer to this [documentation](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A)
        * Post Installation
            * export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
            * export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}} 
            * export CUDA_HOME=/usr/local/cuda-8.0
            * export CUDA_ROOT=/usr/local/cuda-8.0
        * NVIDIA Proprietary Driver Issues
            * [symlink issues for libEGL.so](https://askubuntu.com/questions/900285/libegl-so-1-is-not-a-symbolic-link)
2. Install CUdnn 
    * Register on the [developer.nvidia.com](https://developer.nvidia.com/rdp/cudnn-download)
3. Run `nvcc --version` in the command line to check if everything was installed properly
    * For Ubuntu
        * `sudo apt-get install nvidia-cuda-toolkit`
        * `dpkg -l | grep -i nvidia`
4. To check the installation path 
    * `which nvcc`
5. A GUI app for Nvidia Card Details
    * [Link](https://sourceforge.net/projects/cuda-z/?source=typ_redirect)
    * Ubuntu
        * sudo apt-get install libxrender1:i386 libxtst6:i386 libxi6:i386
        * apt-get install lib32stdc++6
        * chmod + x CUDA-Z-0.10.251-32bit.run
        * ./CUDA-Z-0.10.251-32bit.run


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
    * For ubuntu following this [link](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/)

## LIBRARIES FOR DEEP LEARNING
1. pip install --upgrade tensorflow-gpu
    * Refer [here](https://www.tensorflow.org/install/install_linux) for Linux
2. pip install theano
    * Ubuntu
      * vim ~/.theanorc
      * ```
        [global]
        device=gpu
        floatX=float32
        
        [nvcc]
        flags=-D_FORCE_INLINES
        ```
         
3. pip install keras
    * Ubuntu
        * vim .keras/keras.json
            * > {"backend": "tensorflow","epsilon": 1e-07,"image_data_format": "channels_last","floatx": "float32"}
            * > {"backend": "tensorflow","epsilon": 1e-07,"image_data_format": "channels_last","floatx": "float32"}
        * While running a NN, run `nvidia-smi` to check processes using GPU

4. pip install jupyter
5. pip install h5py


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










