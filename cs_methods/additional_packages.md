Refs : https://gist.github.com/IamAdiSri/a379c36b70044725a85a1216e7ee9a46

## Step1
1. git clone https://github.com/chokkan/liblbfgs.git
2. cd liblbfgs
3. sudo apt-get install libtool automake
4. ./autogen.sh (creates the configure file)
5. ./configure --enable-sse2
6. make
7. make install (sudo make install) --> installs in /usr/local/lib

## Step2
1. git clone https://bitbucket.org/rtaylor/pylbfgs.git
2. sudo apt-get install python3-dev (gives Python.h)

## Step3
1.vim pylbfgs/setup.py
    i. Change Line11 to "include_dirs=['/usr/local/include', '/home/strider/anaconda3/pkgs/numpy-1.11.1-py35_0/lib/python3.5/site-packages/numpy/core/include/numpy/']"
2. vim pylbfgs/pylbfgs.c
    ii. -- Change "/#include <numpy/arrayobject.h>" to "#include <arrayobject.h>"
4. python setup.py install 
    iii. Look out for the -I params for include_dirs

## Step4
    i. python3
    ii. from pylbfgs import owlqn #Simple import test

Step6
pip install cvxpy
pip install Cython
pip install pyrwt
pip3 install scipy
sudo apt-get install python-imaging
pip3 install matplotlib
pip3 install jupyter
pip3 install jupyter_contrib_nbextensions
scp -i alexvpc.pem escher_waterfall.JPG ubuntu@34.202.227.250:/home/ubuntu/Mody/env_cs/code

Step 7 (Final)
jupyter contrib nbextension install --user
jupyter notebook --port 9090 --no-browser --notebook-dirs </home/ubuntu/...>


Step 8 Code:
http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/