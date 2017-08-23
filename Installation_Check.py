#!
import sys
print ('\nPython Executable:', sys.executable)
print ('\nPython version:',sys.version)
# import mujoco_py
# import tensorflow as tf
# print('\n---------Tensoflow Version:', tf.__version__)

# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
# 	print ('\n---------TF session:',sess)

# 	a = tf.random_normal((100,100))
# 	b = tf.random_normal((100,500))
# 	c = tf.matmul(a,b)
# 	sess.run(c)
# 	print ('\n---------matmul:',c.eval())
# 	sess.close()

## For Keras
# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'

"""
1. There are sessions and graphs in TensorFlow
2. One session can have at a point of time, any graphs run under it.
"""
