
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
"""

# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
# 	print ('\n---------TF session:',sess)

# 	a = tf.random_normal((100,100))
# 	b = tf.random_normal((100,500))
# 	c = tf.matmul(a,b)
# 	sess.run(c)
# 	print ('\n---------matmul:',c.eval())
# 	sess.close()
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


