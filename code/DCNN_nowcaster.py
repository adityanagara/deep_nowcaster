# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:48:23 2016

@author: adityanagarajan
"""

import numpy as np
import os
import theano

from theano import tensor as T
import BuildDataSet
import nowcast

import random

import time

from matplotlib import pyplot as plt
import lasagne
from lasagne.regularization import regularize_layer_params, l2, l1

import csv
import cPickle

import sys

def build_DCNN(input_var = None):
    
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

def save_file(file_name):
    file_path = 'output/' + file_name
    with open(file_path,'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['epoch_number','training_loss','validation_loss'])

def append_file(file_name):
    file_path = 'output/' + file_name
    with open(file_path,'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([1,2,3])

    
def build_CNN(input_var = None):
    
#    from lasagne.layers import dnn
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
    
    

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

                                   

def main(file_name = 'test_file.csv',num_points = 10):
    save_file(file_name)
    # Define number of example points to sample
    
    # Define the threshold as none so that we use actual values of reflectivity
    data_builder = BuildDataSet.dataset(Threshold = None)
    # Sample points and make the IPW and refl frames
    PixelPoints = data_builder.sample_random_pixels()
    train_set = data_builder.make_points_frames(PixelPoints[:num_points])
    #validation_set = data_builder.make_points_frames(PixelPoints[100:200])
    X_train,Y_train = data_builder.arrange_frames(train_set)
    # Define the input tensor
    input_var = T.tensor4('inputs')
    # Define the output tensor (in this case it is a real value or reflectivity values)
    output_var = T.fcol('targets')
    
    network,l_hidden1 = build_CNN(input_var)
    
    train_prediction = lasagne.layers.get_output(network)
    
    loss = T.mean(lasagne.objectives.squared_error(train_prediction,output_var))
    
    l1_penalty = regularize_layer_params(l_hidden1, l1)
    
    loss = loss + l1_penalty
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0000001, momentum=0.9)
            

    
    train_fn = theano.function([input_var, output_var], loss, updates=updates)
    
    
    num_epochs = 100
    # Start training
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, Y_train, 1059, shuffle=False):
            inputs, targets = batch
            print inputs.shape,targets.shape
            train_err += train_fn(inputs, targets)
            train_batches += 1

        print train_batches
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


if __name__ == '__main__':
    if '--help' is sys.argv:
        print 'Help Stuff!!'
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['file_name'] = sys.argv[1]
        else:
            
    file_name = sys.argv[1]
    
    append_file(file_name)
    append_file(file_name)
    print 'Done!'
    

#if __name__ == '__main__':
#    if ('--help' in sys.argv) or ('-h' in sys.argv):
#        print("Trains a neural network on MNIST using Lasagne.")
#        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
#        print()
#        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
#        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
#        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
#        print("       input dropout and DROP_HID hidden dropout,")
#        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
#        print("EPOCHS: number of training epochs to perform (default: 500)")
#    else:
#        kwargs = {}
#        if len(sys.argv) > 1:
#            kwargs['model'] = sys.argv[1]
#        if len(sys.argv) > 2:
#            kwargs['num_epochs'] = int(sys.argv[2])
#        main(**kwargs)
        
#    main()
    
    
        
    








