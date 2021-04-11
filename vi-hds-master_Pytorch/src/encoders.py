# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.


import numpy as np
import torch.nn as nn

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def init_u(n_in):
    return tf.Variable(xavier_init(n_in, 1))

def init_w(n_in):
    return tf.Variable(xavier_init(n_in, 1))

def init_b(n_in):
    return tf.Variable(xavier_init(n_in, 1))

def ConditionalEncoder(params):
    n_filters = params["n_filters"] #10
    filter_size = params["filter_size"] #10
    pool_size = params["pool_size"] #5
    n_hidden = params["n_hidden"] #50
    #data_format = params["data_format"] #channel_last by default: 默认最后一位是channel
    #lambda_l2 = params["lambda_l2"] #0.1
    #lambda_l2_hidden = params["lambda_l2_hidden"] #0.1
    #transfer_func = params["transfer_func"] #tanh
    return nn.Sequential(
        nn.Conv1d(4,n_filters, filter_size),  #output最后一维的维度和filter的个数相等 #对该层的权重值进行正则化
        nn.AvgPool1d(pool_size, 1),
        nn.Flatten(),
        nn.Linear(720,n_hidden)) #720最好要泛化