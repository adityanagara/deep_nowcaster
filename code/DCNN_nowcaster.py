# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:48:23 2016

@author: adityanagarajan
"""




# Custom package in ../includes
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
import DCNN_network

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

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
    Deep_learner = DCNN_network.DCNN_network()
    # Define the input tensor
    input_var = T.tensor4('inputs')
    # Define the output tensor (in this case it is a real value or reflectivity)
    if compute_flag == 'gpu3_softmax':
        output_var = T.ivector('targets')
    else:
        output_var = T.fcol('targets')
    # User input to decide which experiment to run, cpu runs were performed
    # to check if the network was working correctly
    if compute_flag =='cpu': 
        network,l_hidden1 = Deep_learner.build_CNN(input_var)
    elif compute_flag == 'cpu2':
        network,l_hidden1 = Deep_learner.build_CNN_2(input_var)
    elif compute_flag == 'cpu3':
        network,l_hidden1 = Deep_learner.build_CNN_3(input_var)
    elif compute_flag == 'gpu2':
        print('gpu2 experiment')
        network,l_hidden1 = Deep_learner.build_DCNN_2(input_var)
    elif compute_flag == 'gpu3':
        print('gpu3 experiment')
        network,l_hidden1 = Deep_learner.build_DCNN_3(input_var)
    elif compute_flag == 'deep':
        network,l_hidden1 = Deep_learner.build_DCNN_deep(input_var)
    elif compute_flag == 'gpu3_softmax':
        network,l_hidden1 = Deep_learner.build_DCNN_3_softmax(input_var)
    else:
        network,l_hidden1 = Deep_learner.build_DCNN(input_var)
    
    train_prediction = lasagne.layers.get_output(network)
    test_prediction = lasagne.layers.get_output(network)
    if compute_flag == 'gpu3_softmax':
        loss = lasagne.objectives.categorical_crossentropy(train_prediction, output_var)
        loss = loss.mean()
    else:
    
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
    
    train_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), output_var),
                      dtype=theano.config.floatX) 
    # Define theano function which generates and compiles C code for the optimization problem
    train_fn = theano.function([input_var, output_var], [loss,train_acc], updates=updates)
    

    
#    test_fn = theano.function([input_var, output_var],test_loss, updates=updates)
    
    base_path = '/home/an67a/deep_nowcaster/data/dataset2/'
    training_set_list = os.listdir(base_path)
    training_set_list = filter(lambda x: x[-4:] == '.pkl' and 'val' not in x,training_set_list)
    validation_set_list = os.listdir(base_path)
    validation_set_list = filter(lambda x: x[-4:] == '.pkl' and 'val' in x,validation_set_list)
    experiment_start_time = time.time()
    # Load Data Set
    DataSet = []
    print('Loading data set...')
    for file_name in training_set_list[:3]:
        print file_name
        temp_file = file(base_path + file_name,'rb')
        X_train,Y_train = cPickle.load(temp_file)
        temp_file.close()
        Y_train = Y_train.reshape(-1,).astype('uint8')
        DataSet.append((X_train,Y_train))
    
    print('Start training...')
    for epoch in range(num_epochs):
        print('Epoch number : %d '%epoch)
        train_err = 0
        train_batches = 0
        train_acc = 0
        start_time = time.time()
        for data in DataSet:
#        for file_name in training_set_list:
#            print file_name
#            temp_file = file(base_path + file_name,'rb')
#            X_train,Y_train = cPickle.load(temp_file)
#            Y_train = Y_train.astype('uint8')
#            temp_file.close()
            for batch in iterate_minibatches(data[0], data[1], 1059, shuffle=False):
                inputs, targets = batch
                err,acc = train_fn(inputs, targets)
                train_err += err
                train_acc += acc
                train_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))
        append_file(results_file_name,epoch + 1,round(train_err / train_batches,2),round((train_acc / train_batches) * 100,2))
        
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


    
        
    








