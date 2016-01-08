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
import cPickle

from matplotlib import pyplot as plt
import lasagne

# Define the threshold as none so that we use actual values of reflectivity
data_builder = BuildDataSet.dataset(Threshold = None)

PixelPoints = data_builder.sample_random_pixels()

data_builder.plot_domain(PixelPoints)

train_set = data_builder.make_points_frames(PixelPoints[:10])

#validation_set = data_builder.make_points_frames(PixelPoints[100:200])

X_train,Y_train = data_builder.arrange_frames(train_set)






#now_caster = nowcast.BuildNowcaster()
#data_builder = BuildDataSet.dataset()
#
## We are going to choose a sub domain in our total 300x300 domain in DFW
#fill_domain = (range(17,83),range(17,83))
##PixelPoints = plot_domains(fill_domain[0],fill_domain[1],region_dict['region1'][2])
#
#PixelX = fill_domain[0]
#PixelY = fill_domain[1]
#
## Pull the central chunk of points out
#central_chunk = (range(46,54),range(46,54))
#
#central_chunk_points = [(x_,y_) for x_ in central_chunk[0] for y_ in central_chunk[1]]
#
## Pair up the pixel points
#PixelPoints = [(x,y) for x in PixelX for y in PixelY]
#
## Remove all central points from the pair
#PixelPoints = [pairs for pairs in PixelPoints if pairs not in central_chunk_points]
#
## Randomely sample 1500 pairs of points
#random.seed(12345)   
#PixelPoints = [PixelPoints[x] for x in random.sample(range(4292),1500)]
#
#PixelPoints = np.array(PixelPoints)
#
## Plot the domain to check the points
##data_builder.plot_domain(PixelPoints)
#
#sorted_days = data_builder.club_days()
#
#days_in_sorted = sorted_days.keys()
#
#days_in_sorted.sort()
#
#print days_in_sorted
#
## Let the size of one batch be 500 points all days
## This function takes the fields surrounding each point dynamically
## We can use this if we do not have a large disc space
#def make_mini_batches(PixelPoints,days_in_sorted):
#    IPW_Refl_points = []
#    save_ctr = 1
#    for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
#        print 'Building data set for point: (%d,%d)'%(x_,y_)
#        for set_ in days_in_sorted:
#            temp_ipw_file_list = filter(lambda x: x[7:10] in sorted_days[set_],data_builder.IPWfiles)
#            temp_radar_file_list = filter(lambda x: x[9:12] in sorted_days[set_],data_builder.Radarfiles)
#            tmp_array = data_builder.build_features_and_truth(temp_ipw_file_list,temp_radar_file_list,x_,y_)
#            IPW_Refl_points.append(tmp_array)
#        if save_ctr % 100 == 0:
#            print 'Batch Done %d'%save_ctr
#            IPW_Refl_points = []
#        save_ctr+=1
#
#def load_data_set(set_no = 3):
#    # Temporarely route the data from summer
#    file_list = os.listdir('data/TrainTest/RandomPoints/')
#    file_list = filter(lambda x: x[:3] == 'IPW',file_list)
#    random_points_file = file('data/TrainTest/RandomPoints/' + file_list[set_no],'rb')
#    data_set = cPickle.load(random_points_file)
#    random_points_file.close()
#    return data_set
#
#
#def build_data_set(set_no):
#    # Load entire 1500 points
#    data = load_data_set(set_no = set_no)
#    
#    IPWFeatures = np.concatenate(map(lambda x: x[0].astype('float32'),data))
#    
#    Y_train = IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)
#    
#    IPWFeatures = IPWFeatures[:,:-1].reshape(IPWFeatures.shape[0],6,33,33)
#    
#    IPWFeatures = IPWFeatures[:,2:,:,:]
#    
#    ReflFeatures = np.concatenate(map(lambda x: x[1].astype('float32'),data))
#    
#    ReflFeatures = ReflFeatures.reshape(IPWFeatures.shape[0],6,33,33)
#    
#    ReflFeatures = ReflFeatures[:,2:,:,:]
#    
#    X_train = np.concatenate((IPWFeatures,ReflFeatures),axis=1)
#    
#    print X_train.shape,Y_train.shape
#    
#    return X_train,Y_train

def build_DCNN(input_var = None):
    
    from lasagne.layers import dnn
    
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None, 33, 33, 8),
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
    
    return l_out
    
                                        

#def main():
#    # Define the input tensor
#    input_var = T.tensor4('inputs')
#    # Define the output tensor (in this case it is a real value or reflectivity values)
#    output_var = T.fscalar('targets')
#    
#    network = build_DCNN(input_var)
#    
#    train_prediction = lasagne.layers.get_output(network)
#    
#    loss = lasagne.objectives.squared_error(train_prediction,output_var)
#    
#    loss = loss.mean()
#    
#    params = lasagne.layers.get_all_params(network, trainable=True)
#    
#    updates = lasagne.updates.sgd(loss, params, 0.001)
#    
#    test_prediction = lasagne.layers.get_output(network, deterministic=True)
#    
#    test_loss = lasagne.objectives.squared_error(test_prediction,
#                                                            output_var)
#    
#    train_fn = theano.function([input_var, output_var], loss, updates=updates)
#    
#    val_fn = theano.function([input_var, output_var], [test_loss])
#    
#    num_epochs = 100
#    
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

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


        
    
    
    
    
    
    
    
#train_set = build_data_set(2)
#
#test_set = build_data_set(3)
#
#
#PixelPoints = PixelPoints[:100,:]
#
#make_mini_batches(PixelPoints,days_in_sorted)
#
#print 'Done!'
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
#
#    # Optionally, you could now dump the network weights to a file like this:
#    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
#    #
#    # And load them again later on like this:
#    # with np.load('model.npz') as f:
#    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#    # lasagne.layers.set_all_param_values(network, param_values)
#
#
#
#def build_cnn(input_var=None):
#    # As a third model, we'll create a CNN of two convolution + pooling stages
#    # and a fully-connected hidden layer in front of the output layer.
#
#    # Input layer, as usual:
#    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
#                                        input_var=input_var)
#    # This time we do not apply input dropout, as it tends to work less well
#    # for convolutional layers.
#
#    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
#    # convolutions are supported as well; see the docstring.
#    network = lasagne.layers.Conv2DLayer(
#            network, num_filters=32, filter_size=(5, 5),
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.GlorotUniform())
#    # Expert note: Lasagne provides alternative convolutional layers that
#    # override Theano's choice of which implementation to use; for details
#    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
#
#    # Max-pooling layer of factor 2 in both dimensions:
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
#
#    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
#    network = lasagne.layers.Conv2DLayer(
#            network, num_filters=32, filter_size=(5, 5),
#            nonlinearity=lasagne.nonlinearities.rectify)
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
#
#    # A fully-connected layer of 256 units with 50% dropout on its inputs:
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.5),
#            num_units=256,
#            nonlinearity=lasagne.nonlinearities.rectify)
#
#    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.5),
#            num_units=10,
#            nonlinearity=lasagne.nonlinearities.softmax)
#
#    return network

        
        
    








