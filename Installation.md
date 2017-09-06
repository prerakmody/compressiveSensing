# Installing CUDA for NVIDIA
0. **Nvidia Drivers**
    * ***
    * Ubuntu
        * Check if card is available
            * Card : `lspci | grep -i nvidia`
            * Driver Version: 
                * `cat /proc/driver/nvidia/version`
                * `ls /usr/lib | grep nvidia-`
                * `dpkg --get-selections | grep nvidia`
                    * The above three values should match
                    * You should not have multiple Nvidia drivers installed
                * 
            * A process list for your GPU : `nvidia-smi`
            * Check VGA and 3D drivers : `lspci -k | grep -EA2 'VGA|3D'`
                * To turn VGA to Intel GPU
                    * `sudo nvidia-settings`
                    * Go to PRIME Profiles
                    * Select Intel GPU

            * Experimental reset command : 
                * `sudo su`
                * `LD_PRELOAD=/usr/lib/nvidia-367/libnvidia-ml.so`
                * `sudo nvidia-smi --gpu-reset -i 0`
        * If not available 
            * sudo add-apt-repository ppa:graphics-drivers/ppa
            * sudo apt-get update
            * sudo ubuntu-drivers autoinstall (this also helps if the login screen turns into a loop)
            * sudo reboot
            * nvidia-smi
                * You may have to add the following to the command line : `export LD_PRELOAD=/usr/lib/nvidia-375/libnvidia-ml.so`
1. **Install Cuda**
    * ***
    * Linux
        * Refer to this [documentation](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A)
            * use the debian package for installation
        * Post Installation
            * Include these in ~/.profile
            * ```
               export CUDA_HOME=/usr/local/cuda-8.0
               export CUDA_ROOT=/usr/local/cuda-8.0
               export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
               export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/lib/nvidia-375
               export LD_PRELOAD=/usr/lib/nvidia-375/libnvidia-ml.so
            ```
            * To check if driver is detected by CUDA
                * `find /usr/local/cuda-8.0 -name deviceQuery`
                * `./deviceQuery`
                    * you might have to use the command `make` to get an executable    
        * NVIDIA Proprietary Driver Issues
            * [symlink issues for libEGL.so](https://askubuntu.com/questions/900285/libegl-so-1-is-not-a-symbolic-link)
2. **Install CUdnn** 
    * ***
    * Register on the [developer.nvidia.com](https://developer.nvidia.com/rdp/cudnn-download)
    * Download the cudnn files
        * Copy them as follows:
            * ```
               sudo cp -P lib64/libcudnn* /usr/local/cuda-8.0/lib64/
               sudo cp -P include/cudnn.h /usr/local/cuda-8.0/include/
               sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
            ``` 
    * You need to have the following files
        * Check `find /usr | grep libcudnn`
        * Check `find /usr | grep cudnn.h`
        * cudnn.h (`/usr/local/cuda-8.0/include/cudnn.h`)
        * libcudnn.so (`/usr/local/cuda-8.0/lib64/libcudnn.so`)
        * libcudnn.so.6
        * libcudnn.so.6.0.21
        * libcudnn_static.a

3. **More Checks**
    * ***
    * Run `nvcc --version` in the command line to check if everything was installed properly
    * For Ubuntu
        * `dpkg -l | grep -i nvidia`
    * To check the installation path 
        * `which nvcc`
    * A GUI app for Nvidia Card Details
        * [Link](https://sourceforge.net/projects/cuda-z/?source=typ_redirect)
        * Ubuntu
            * sudo apt-get install libxrender1:i386 libxtst6:i386 libxi6:i386
            * apt-get install lib32stdc++6
            * chmod + x CUDA-Z-0.10.251-32bit.run
            * ./CUDA-Z-0.10.251-32bit.run

<br/>
<br/>

# INSTALLING VARIOUS LIBRARIES
<br/>

## NUMPY + MKL problems
1. Sometimes importing NUMP/SCIPY leads to some issues since we are using a conda-env.
2. Simply uninstall exiting numpy / scipy installation (sometimes you have to try this twice) 
3. Then reinstall (numpy/scipy) from the UCI website (http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
4. You could also download cv2 from there
    * For ubuntu following this [link](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/)

## LIBRARIES FOR DEEP LEARNING
1. *Tensorflow* 
    * ***
    * pip install --upgrade tensorflow-gpu
    * Ubuntu
        * Refer [here](https://www.tensorflow.org/install/install_linux)
        * sudo apt-get install libcupti-dev
        * To control logging:
            * `export TF_CPP_MIN_LOG_LEVEL=1` (very verbose)
            * `export TF_CPP_MIN_LOG_LEVEL=2` (only error messages)

2. *Theano*
    * ***
    * conda install theano pygpu
    * Ubuntu
      * vim ~/.theanorc
      * ```
        [global]
        device=gpu
        floatX=float32
        mode=FAST_RUN
        optimizer_including=cudnn (doubtful)
        
        [nvcc]
        flags=-D_FORCE_INLINES

        [dnn]
        enabled = True
        include_path=/usr/local/cuda-8.0/include
        library_path=/usr/local/cuda-8.0/lib64
        ```
      * If any issues with device not found : `sudo service lightdm restart`
      * To clear cache : `theano-cache purge`
      * ```
        sudo apt-get install gcc-4.9 g++-4.9
        sudo ln -s /usr/bin/gcc-4.9 /usr/local/cuda-8.0/bin/gcc
        sudo ln -s /usr/bin/g++-4.9 /usr/local/cuda-8.0/bin/g++
        ```
         
3. *Keras*
    * ***
    * pip install keras
    * Ubuntu
        * vim .keras/keras.json
            * ```{"backend": "tensorflow","epsilon": 1e-07,"image_data_format": "channels_last","floatx": "float32"}```
            * ```{"backend": "tensorflow","epsilon": 1e-07,"image_data_format": "channels_last","floatx": "float32"}```
        * While running a NN, run the following command to check processes using GPU
            * `watch -n 1 nvidia-smi`

4. pip install jupyter
5. pip install h5py

<br/>
<br/>

## Widgets for Ipython
0. https://github.com/ipython-contrib/jupyter_contrib_nbextensions
1. pip install jupyter_contrib_nbextensions
2. jupyter contrib nbextension install --user
3. jupyter notebook --generate-config

## Other libraries
1. pip install joblib
2. pip install opencv-contrib-python

## To run the Jupyer notebooks
1.  Open command prompt
2.  Navigate to your folder
3.  Type `jupyter notebook` (do "source activate py3.5" if you wish for the python version to be different from system default)

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

## Removing conda envs
1. conda remove --name py3.5 --all







