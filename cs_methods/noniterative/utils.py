"""
1. Get X and y
2. Split into training and testing data
3. Reshape X -> (samples, channels, x-axis, y-axis)
   Reshape y -> (sampples, x-axis * y-axis)
4. Create ReconNet Units
   Figure out the zero padding post each step to maintain a 33 * 33 filter size 
"""

""" 1. CALLABLE FUNCTION """
def get_data(WIDTH, HEIGHT, backend_type = 'tensorflow', mr_folder='mr40', test_split = 0.0):
    import os
    import numpy as np
    from glob import glob
    from sklearn.cross_validation import train_test_split

    np.random.seed(7)

    url_data_X = '../data/' + mr_folder + '/data_patches_X.gz'
    url_data_y = '../data/' + mr_folder + '/data_patches_y.gz'

    X, y = get_data_backend(url_data_X, url_data_y, WIDTH, HEIGHT, backend_type)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0)
    
    print ('1. X_train:', X_train.shape, ' y_train:', y_train.shape)
    print ('1. X_test:', X_test.shape, ' y_test:', y_test.shape)

    return X_train, y_train, X_test, y_test


""" 2. BACKEND FORMAT """
def get_data_backend(url_data_X, url_data_y, WIDTH, HEIGHT, backend_type):
    import joblib
    import numpy as np
    import matplotlib.pyplot as plt

    with open(url_data_X, 'rb') as handle:
        X = joblib.load(handle)
        
    with open(url_data_y, 'rb') as handle:
        y = joblib.load(handle)
    
    idxs = np.random.randint(0, len(X), 3)
    
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(X[idxs[0]].reshape((WIDTH,HEIGHT)), cmap = plt.cm.gray)
    axarr[1].imshow(X[idxs[1]].reshape((WIDTH,HEIGHT)), cmap = plt.cm.gray)
    axarr[2].imshow(X[idxs[2]].reshape((WIDTH,HEIGHT)), cmap = plt.cm.gray)
    
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(y[idxs[0]].reshape((WIDTH,HEIGHT)), cmap = plt.cm.gray)
    axarr[1].imshow(y[idxs[1]].reshape((WIDTH,HEIGHT)), cmap = plt.cm.gray)
    axarr[2].imshow(y[idxs[2]].reshape((WIDTH,HEIGHT)), cmap = plt.cm.gray)

    if backend_type == 'tensorflow':
        print ('0. --------------------- USING TENSORFLOW FORMAT ---------------------  ')    
        X = X.reshape((X.shape[0], 1, 33, 33))
    else:
        pass

    print ('1. X:', X.shape, ' y:', y.shape)

    return X , y

""" 3. GPU CHECKS """
def check_gpu(verbose = 1):
    import os

    if verbose:
        print ('0. Envs  : CUDA_HOME', os.environ['CUDA_HOME'])
        print ('0. Envs  : CUDA_ROOT',os.environ['CUDA_ROOT'])
        print ('0. Envs  : LD_LIBRARY_PATH:', os.environ['LD_LIBRARY_PATH'])
        print ('0. Envs  : PATH (containing cuda)', [each for each in os.environ['PATH'].split(':') if each.find('cuda') > -1])
        print ('0. CUDnn :', [each.replace('\n', '') for each in os.popen('find /usr/local/cuda-8.0/ -name *dnn*')])

    from keras import backend as K
    K.clear_session()
    
    print ('\n0. Keras backend:', K.backend())
    if K.backend() == 'tensorflow':
        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()
        for device in devices:
            print ('0. TensorFlow Devices:', str(device.name).replace('\n',''))
        print ('\n')
        
        return 1 if len(devices) > 1 else 0
    
    else:
        return 0
