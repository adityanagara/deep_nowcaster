# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:48:23 2016

@author: adityanagarajan
"""

import numpy as np
import theano

from theano import tensor as T
import BuildDataSet

import time

import lasagne
from lasagne.regularization import regularize_layer_params, l2, l1

import csv
import cPickle

import sys

def build_DCNN(input_var = None):
    
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

def save_file(file_name):
    file_path = 'output/' + file_name
    with open(file_path,'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['epoch_number','training_loss','validation_loss'])

def append_file(file_name,val_1,val_2,val_3):
    file_path = 'output/' + file_name
    with open(file_path,'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([val_1,val_2,val_3])

    
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

def build_CNN_2(input_var = None):
    
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
    
def build_DCNN_2(input_var = None):
    
    from lasagne.layers import dnn
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None, 8, 33, 33),
                                        input_var=input_var)
    
    l_conv1 = dnn.Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(11, 11),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
    l_conv2 = dnn.Conv2DLayer(
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

                                   

def main(num_epochs = 1000,num_points = 10,compute_flag='cpu'):
    results_file_name = 'exp_' + str(num_epochs) + '_' + str(num_points) + '_' + compute_flag + '.csv'
    network_file_name = 'network_' + str(num_epochs) + '_' + str(num_points) + '_' + compute_flag 
    num_points = int(num_points)
    print 'Saving file to: %s' % results_file_name
    print 'Number of points: %d ' % num_points
    print 'Compute Flag: %s ' % compute_flag
    save_file(results_file_name)
    # Define number of example points to sample
    
    # Define the threshold as none so that we use actual values of reflectivity
    data_builder = BuildDataSet.dataset(Threshold = None)
    # Sample 1500 points and make the IPW and refl frames
    PixelPoints = data_builder.sample_random_pixels()
    train_set = data_builder.make_points_frames(PixelPoints[:num_points])
    X_train,Y_train = data_builder.arrange_frames(train_set)
    # Define the input tensor
    input_var = T.tensor4('inputs')
    # Define the output tensor (in this case it is a real value or reflectivity values)
    output_var = T.fcol('targets')
    
    # Build the validation set take the last num_points
    validation_set = data_builder.make_points_frames(PixelPoints[-num_points:])
    X_val,Y_val = data_builder.arrange_frames(validation_set)
    
    # User input to decide which experiment to run, cpu runs were performed
    # to check if the network was working correctly
    if compute_flag =='cpu': 
        network,l_hidden1 = build_CNN(input_var)
    elif compute_flag == 'cpu2':
        network,l_hidden1 = build_CNN_2(input_var)
    elif compute_flag == 'gpu2':
        network,l_hidden1 = build_DCNN_2(input_var)
    else:
        network,l_hidden1 = build_DCNN(input_var)
    
    train_prediction = lasagne.layers.get_output(network)
    test_prediction = lasagne.layers.get_output(network)
    # Define the mean square error objective function
    loss = T.mean(lasagne.objectives.squared_error(train_prediction,output_var))
    
    test_loss = T.mean(lasagne.objectives.squared_error(test_prediction,output_var))
    # Add a l1 regulerization on the fully connected dense layer
    l1_penalty = regularize_layer_params(l_hidden1, l1)
    
    loss = loss + l1_penalty
    
    test_loss = loss + l1_penalty
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0000001, momentum=0.9)
            
    # Define theano function which generates and compiles C code for the optimization problem
    train_fn = theano.function([input_var, output_var], loss, updates=updates)
    
    test_fn = theano.function([input_var, output_var],test_loss)
    
    experiment_start_time = time.time()
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
        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, Y_val, 1059, shuffle=False):
            inputs, targets = batch
            err, acc = test_fn(inputs, targets)
            val_err += err
            val_batches += 1
        
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        append_file(results_file_name,epoch + 1,round(train_err / train_batches,2),val_err / val_batches)
        
        # Dump the network file every 100 epochs
        if epoch + 1 % 100 == 0:
            network_file = file('output/'+ network_file_name + '_' + str(epoch + 1) + '.pkl','wb')
            cPickle.dump(network,network_file,protocol = cPickle.HIGHEST_PROTOCOL)
            network_file.close()
    time_taken = round(time.time() - experiment_start_time,2)
    print('The experiment took {:.3f}s'.format(time.time() - experiment_start_time))
    append_file(results_file_name,'The experiment took',time_taken,0)


if __name__ == '__main__':
    if '--help' is sys.argv:
        print 'Help Stuff!!'
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_points'] = sys.argv[2]
        if len(sys.argv) > 3:
            kwargs['compute_flag'] = sys.argv[3]
    main(**kwargs)
    
    print 'Done!' 

#def main(model='mlp', num_epochs=500):
#    # Load the dataset
#    print("Loading data...")
#    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
#
#    # Prepare Theano variables for inputs and targets
#    input_var = T.tensor4('inputs')
#    target_var = T.ivector('targets')
#
#    # Create neural network model (depending on first command line parameter)
#    print("Building model and compiling functions...")
#    if model == 'mlp':
#        network = build_mlp(input_var)
#    elif model.startswith('custom_mlp:'):
#        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
#        network = build_custom_mlp(input_var, int(depth), int(width),
#                                   float(drop_in), float(drop_hid))
#    elif model == 'cnn':
#        network = build_cnn(input_var)
#    else:
#        print("Unrecognized model type %r." % model)
#
#    # Create a loss expression for training, i.e., a scalar objective we want
#    # to minimize (for our multi-class problem, it is the cross-entropy loss):
#    prediction = lasagne.layers.get_output(network)
#    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
#    loss = loss.mean()
#    # We could add some weight decay as well here, see lasagne.regularization.
#
#    # Create update expressions for training, i.e., how to modify the
#    # parameters at each training step. Here, we'll use Stochastic Gradient
#    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
#    params = lasagne.layers.get_all_params(network, trainable=True)
#    updates = lasagne.updates.nesterov_momentum(
#            loss, params, learning_rate=0.01, momentum=0.9)
#
#    # Create a loss expression for validation/testing. The crucial difference
#    # here is that we do a deterministic forward pass through the network,
#    # disabling dropout layers.
#    test_prediction = lasagne.layers.get_output(network, deterministic=True)
#    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
#                                                            target_var)
#    test_loss = test_loss.mean()
#    # As a bonus, also create an expression for the classification accuracy:
#    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
#                      dtype=theano.config.floatX)
#
#    # Compile a function performing a training step on a mini-batch (by giving
#    # the updates dictionary) and returning the corresponding training loss:
#    train_fn = theano.function([input_var, target_var], loss, updates=updates)
#
#    # Compile a second function computing the validation loss and accuracy:
#    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
#
#    # Finally, launch the training loop.
#    print("Starting training...")
#    # We iterate over epochs:
#    for epoch in range(num_epochs):
#        # In each epoch, we do a full pass over the training data:
#        train_err = 0
#        train_batches = 0
#        start_time = time.time()
#        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
#            inputs, targets = batch
#            train_err += train_fn(inputs, targets)
#            train_batches += 1
#
#        # And a full pass over the validation data:
#        val_err = 0
#        val_acc = 0
#        val_batches = 0
#        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
#            inputs, targets = batch
#            err, acc = val_fn(inputs, targets)
#            val_err += err
#            val_acc += acc
#            val_batches += 1
#
#        # Then we print the results for this epoch:
#        print("Epoch {} of {} took {:.3f}s".format(
#            epoch + 1, num_epochs, time.time() - start_time))
#        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
#        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
#        print("  validation accuracy:\t\t{:.2f} %".format(
#            val_acc / val_batches * 100))
#
#    # After training, we compute and print the test error:
#    test_err = 0
#    test_acc = 0
#    test_batches = 0
#    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
#        inputs, targets = batch
#        err, acc = val_fn(inputs, targets)
#        test_err += err
#        test_acc += acc
#        test_batches += 1
#    print("Final results:")
#    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
#    print("  test accuracy:\t\t{:.2f} %".format(
#        test_acc / test_batches * 100))

    
        
    








