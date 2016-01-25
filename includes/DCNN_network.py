# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:08:37 2016

@author: adityanagarajan
"""

import lasagne

import numpy as np
import os
import theano
from theano import tensor as T

class DCNN_network(object):
    
    def __init__(self,input_shape = (None, 8, 33, 33)):
        self.input_shape = input_shape
    
    def build_DCNN(self,input_var = None):
    
        from lasagne.layers import dnn
        print 'We hit the GPU code!'
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        l_in = lasagne.layers.InputLayer(shape=(None, 8, 33, 33),
                                        input_var=input_var)
    
        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=32,
            filter_size=(11, 11),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv1,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=1,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        return l_out,l_hidden1
    
    def build_CNN(self,input_var = None):
    
        from lasagne.layers import Conv2DLayer
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        l_in = lasagne.layers.InputLayer(shape=(None, 8, 33, 33),
                                        input_var=input_var)
    
        l_conv1 = Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(11, 11),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv1,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=1,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        return l_out,l_hidden1

    def build_CNN_2(self,input_var = None):
    
        from lasagne.layers import Conv2DLayer
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        l_in = lasagne.layers.InputLayer(shape=(None, 8, 33, 33),
                                        input_var=input_var)
    
        l_conv1 = Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(11, 11),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        l_conv2 = Conv2DLayer(
            l_conv1,
            num_filters=48,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=1,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        return l_out,l_hidden1
    
    def build_DCNN_2(self,input_var = None):
    
        from lasagne.layers import dnn
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        l_in = lasagne.layers.InputLayer(shape=(None, 8, 33, 33),
                                        input_var=input_var)
    
        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=32,
            filter_size=(11, 11),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        l_conv2 = dnn.Conv2DDNNLayer(
            l_conv1,
            num_filters=48,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=1,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
        return l_out,l_hidden1
    
    
    
        
    