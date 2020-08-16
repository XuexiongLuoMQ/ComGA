# -*- coding: utf-8 -*-
# @Time    : 2018/9/28 11:20
# @Author  : Haoyi
# @Email   : isfanhy@gmail.com
# @File    : __init__.py
# @Software: PyCharm

# import the necessary packages
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
	
tf.compat.v1.Session(config=config)
#set_session(tf.compat.v1.Session(config=config))
