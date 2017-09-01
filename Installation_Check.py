
"""
PYTHON VERSIONS
"""
import sys
print ('\nPython Executable:', sys.executable)
print ('\nPython version:',sys.version)

# import mujoco_py
# import tensorflow as tf
# print('\n---------Tensoflow Version:', tf.__version__)

"""
FOR TENSORFLOW
# """

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
with tf.device('/gpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

# """
# 1. There are sessions and graphs in TensorFlow
# 2. One session can have at a point of time, any graphs run under it.
# """


"""
FOR THEANO
"""
# from theano import function, config, shared, tensor
# import numpy
# import time

# vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
# iters = 1000

# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], tensor.exp(x))
# print(f.maker.fgraph.toposort())
# t0 = time.time()
# for i in range(iters):
#     r = f()
# t1 = time.time()
# print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (r,))
# if numpy.any([isinstance(x.op, tensor.Elemwise) and
#               ('Gpu' not in type(x.op).__name__)
#               for x in f.maker.fgraph.toposort()]):
#     print('Used the cpu')
# else:
#     print('Used the gpu')

# import theano
# from theano.sandbox.cuda.dnn import dnn_available as d; 
# print(d() or d.msg)
## Using gpu device 0: GeForce 940MX (CNMeM is enabled with initial size: 70.0% of memory, cuDNN None)
