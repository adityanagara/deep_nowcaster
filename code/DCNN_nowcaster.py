# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:48:23 2016

@author: adityanagarajan
"""




# Custom package in ./includes
import BuildDataSet
# Numerical package
import numpy as np
# Deep learning packages 
import theano
from theano import tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params, l2, l1

# Misc. packages 
import time
import csv
import cPickle
import sys
import os


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
# 3 convolutional layers and 1 fully connected layer
def build_CNN_3(input_var = None):
    
    from lasagne.layers import Conv2DLayer
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None, 8, 33, 33),
                                        input_var=input_var)
    
    l_conv1 = Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
    l_conv2 = Conv2DLayer(
            l_conv1,
            num_filters=64,
            filter_size=(3, 3),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
    l_conv3 = Conv2DLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
    l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=2048,
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
# 3 convolutional layers and 1 fully connected layer on GPU
def build_DCNN_3(input_var = None):
    
    from lasagne.layers import dnn
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None, 8, 33, 33),
                                        input_var=input_var)
    
    l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=32,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
    l_conv2 = dnn.Conv2DDNNLayer(
            l_conv1,
            num_filters=64,
            filter_size=(3, 3),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
    l_conv3 = dnn.Conv2DDNNLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
    l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=2048,
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

def main(num_epochs = 100,num_points = 1200,compute_flag='cpu'):
    # Arguments passed as string need to be converted to int    
    num_epochs = int(num_epochs)
    num_points = int(num_points)
    # Define name of output files
    results_file_name = 'exp_' + str(num_epochs) + '_' + str(num_points) + '_' + compute_flag + '.csv'
    network_file_name = 'network_' + str(num_epochs) + '_' + str(num_points) + '_' + compute_flag 
    print 'Saving file to: %s' % results_file_name
    print 'Number of points: %d ' % num_points
    print 'Compute Flag: %s ' % compute_flag
    save_file(results_file_name)    
    # Define the input tensor
    input_var = T.tensor4('inputs')
    # Define the output tensor (in this case it is a real value or reflectivity)
    output_var = T.fcol('targets')
    # User input to decide which experiment to run, cpu runs were performed
    # to check if the network was working correctly
    if compute_flag =='cpu': 
        network,l_hidden1 = build_CNN(input_var)
    elif compute_flag == 'cpu2':
        network,l_hidden1 = build_CNN_2(input_var)
    elif compute_flag == 'cpu3':
        network,l_hidden1 = build_CNN_3(input_var)
    elif compute_flag == 'gpu2':
        print('gpu2 experiment')
        network,l_hidden1 = build_DCNN_2(input_var)
    else:
        network,l_hidden1 = build_DCNN(input_var)
    
    # Define the threshold as none so that we use actual values of reflectivity
    data_builder = BuildDataSet.dataset(Threshold = None)
    # Sample 1500 points and make the IPW and refl frames
    PixelPoints = data_builder.sample_random_pixels()
    # reverse the list for validation set
    rev_PixelPoints = PixelPoints[::-1]
    
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
    
    test_fn = theano.function([input_var, output_var],test_loss, updates=updates)
    
    base_path = '/home/an67a/deep_nowcaster/data/dataset2/'
    training_set_list = os.listdir(base_path)
    training_set_list = filter(lambda x: x[-4:] == '.pkl' and 'val' not in x,training_set_list)
    validation_set_list = os.listdir(base_path)
    validation_set_list = filter(lambda x: x[-4:] == '.pkl' and 'val' in x,validation_set_list)
    experiment_start_time = time.time()
    # Start training
    # Load each point from disc to avoid memory error for > 50 points
    # Pass through all points in the training data
    # Pass through all points in validation set
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        val_batches = 0
        val_err = 0
        start_time = time.time()
        for file_name in training_set_list:
            print('Loading 40 points onto memory...')
            temp_file = file(base_path + file_name,'rb')
            X_train,Y_train = cPickle.load(temp_file)
            temp_file.close()
            for batch in iterate_minibatches(X_train, Y_train, 1059, shuffle=False):
                inputs, targets = batch
                print inputs.shape,targets.shape
                train_err += train_fn(inputs, targets)
                train_batches += 1
        for val_file in validation_set_list:
            val_temp_file = file(base_path + val_file)
            X_val,Y_val = cPickle.load(val_temp_file)
            val_temp_file.close()
            for batch in iterate_minibatches(X_val, Y_val, 1059, shuffle=False):
                inputs, targets = batch
                err = test_fn(inputs, targets)
                val_err += err
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        append_file(results_file_name,epoch + 1,round(train_err / train_batches,2),round(val_err / val_batches,2))
        
        # Dump the network file every 100 epochs
        if (epoch + 1) % 100 == 0:
            print('creating network file')
            network_file = file('/home/an67a/deep_nowcaster/output/'+ network_file_name + '_' + str(epoch + 1) + '.pkl','wb')
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


    
        
    








