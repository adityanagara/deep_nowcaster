# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:56:40 2016

@author: adityanagarajan
"""

import numpy as np
import BuildDataSet

import theano
from theano import tensor as T
import lasagne
import re
import os
from lasagne.layers import dnn

from lasagne.regularization import regularize_layer_params, l2, l1

import cPickle as pkl

np.random.seed(12345)
yr_mon = {14: [5,6,7,8],
              15: [5,6],
              16: [5,6,7]}

blocks = [str(yr) + '_' + str(mon) for yr in [14,15,16] for mon in yr_mon[yr]]

def get_val_doys(storm_dates):
    '''May: 121-151, June: 152 - 181 July: 182 - 212 August: 213 - 243'''
    val_blocks = []
    train_blocks = []
    for bl in blocks:
        yr,mon = re.findall('\d+',bl)
        val_doys = storm_dates[np.logical_and(storm_dates[:,1] == int(yr),storm_dates[:,2] == int(mon))]
        train_doys = storm_dates[np.logical_or(storm_dates[:,1] != int(yr),storm_dates[:,2] != int(mon))]
        val_blocks.append(val_doys)
        train_blocks.append(train_doys)
    return train_blocks,val_blocks


def build_training_validation_sets(data_builder):
    '''Training and validation split: We have a total of 7 months 4 in 2014
    and 2 in 2015. In 2015 we dont have any storm dates from August. Thus we 
    will have 7 of these blocks, train for 6 months and test on the last one'''
    storm_dates_all = {}
    for yr in [14,15,16]:
        storm_dates_all[yr] = data_builder.load_storm_days(yr)    
    storm_dates_all = np.concatenate((storm_dates_all[14],storm_dates_all[15],storm_dates_all[16]))
    train,val = get_val_doys(storm_dates_all)
    return train,val

def arrange_training_validation(tr,val):
    '''return a liat of tuples containing 7 elements where each tuple represents
    the files needed to be loaded for the training and validation sets'''
    file_list = []
    for yr in [14,15,16]:
        temp_file_list = os.listdir('../data/TrainTest/20' + str(yr) + os.sep)
        temp_file_list = filter(lambda x: 'frames' in x,temp_file_list)
        file_list.extend(temp_file_list)
    
    file_list = filter(lambda x: x[-4:] == '.pkl',file_list)
    train_files = []
    val_files= []
    for t,v in zip(tr,val):
        train_list = map(lambda x: (x[0],x[1]),t)
        val_list = map(lambda x: (x[0],x[1]),v)
        train_files.append(filter(lambda x: (int(re.findall('\d+',x)[1]),int(re.findall('\d+',x)[0])) in train_list,file_list))
        val_files.append(filter(lambda x: (int(re.findall('\d+',x)[1]),int(re.findall('\d+',x)[0])) in val_list,file_list))
    return train_files,val_files

def make_dataset_NN_2(data_builder):
    '''Builds the 4 frames of ipw and 4 frames of reflectivity. The way this
    is built is we make num_points = 500 files where each file contains all storm
    dates for a pixel point. So in minibatch SGD we train on each point'''
    storm_dates_all = {}
    PixelPoints = data_builder.sample_random_pixels()
    print PixelPoints.shape
    
    for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
        print 'Building data set for point (%d,%d)'%(x_,y_)
        points_array = []
        for yr in [14,15,16]:
            storm_dates_all[yr] = data_builder.load_storm_days(yr)
            # load the dictionary which gives us the days which are 
            # clubbed together. 
            doy_strings = data_builder.club_days(storm_dates_all[yr])
            days_in_sorted = doy_strings.keys()
            days_in_sorted.sort()
        
            ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(yr)
            
            for set_ in days_in_sorted:
#                print 'Building data set for year: %d and string of days %s'%(yr,set_)
                
                # Get the required files only
                temp_ipw_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],ipw_files)
                temp_refl_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],refl_files)
                temp_ipw_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_ipw_files)
                temp_refl_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_refl_files)
            
                temp_array = data_builder.build_features_and_truth_imgs(temp_ipw_files,temp_refl_files,x_,y_)
                
                ipw_refl_tensors = data_builder.arrange_frames_single(temp_array)
                
                points_array.append((ipw_refl_tensors))
                
                
        # Save each string of days to a pkl file
        print '../data/TrainTest/points_regression/' + 'IPW_refl_frames{0}_{1}.pkl'.format(x_,y_)
        temp_file = file('../data/TrainTest/points_regression/' + 'IPW_refl_frames{0}_{1}.pkl'.format(x_,y_),'wb')
        pkl.dump(points_array,temp_file,protocol = pkl.HIGHEST_PROTOCOL)
        temp_file.close()

def determine_indices(file_name,val_block):
    arr = np.load(file_name)
    val_indices = []
    date_blocks = data_builder.club_days(val_block)
    date_blocks = sorted(date_blocks.keys())
    for i in range(len(arr)):
        if np.all(np.in1d(arr[i][0][0,:2],np.array((val_block[:,1],val_block[:,0])).T)):
            val_indices.append(i)
    return val_indices

def build_DCNN_softmax_cpu(input_var = None):
    
        print('Training the softmax network!!')
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        lasagne.layers.Conv2DLayer
        l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
    
        l_conv1 = lasagne.layers.Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
        
#        l_maxpool = dnn.MaxPool2DDNNLayer(l_conv1)
    
        l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_conv1,p=0.3),
            num_units=2000,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
        return network,l_hidden1

def build_2DCNN_softmax_special(input_var_ipw = None,input_var_refl = None):
    
    print('Single layer conv net')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in_ipw = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_ipw)
    
    l_in_refl = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_refl)
    
    
    l_conv1_ipw = dnn.Conv2DDNNLayer(
            l_in_ipw,
            num_filters=8,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_conv2_ipw = dnn.Conv2DDNNLayer(
            l_conv1_ipw,
            num_filters=8,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'valid'
        )

    conv_shape1 = lasagne.layers.get_output_shape(l_conv2_ipw)
    
    print conv_shape1
    
    l_conv1_refl = dnn.Conv2DDNNLayer(
            l_in_refl,
            num_filters=8,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_conv2_refl = dnn.Conv2DDNNLayer(
            l_conv1_refl,
            num_filters=8,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'valid'
        )
    
    conv_shape2 = lasagne.layers.get_output_shape(l_conv1_refl)
    
    print conv_shape2
    
    l_concat = lasagne.layers.concat([l_conv2_ipw,l_conv2_refl],axis = 1)
    
    concat_shape = lasagne.layers.get_output_shape(l_concat)
    
    print concat_shape
    
    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_concat,p=0.2),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
    network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return network,l_hidden1



def build_2DCNN_softmax_special_refl(input_var_refl = None):
    
    print('2 CNN refl special')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
#    l_in_ipw = lasagne.layers.InputLayer(shape = (None,4,33,33),
#                                        input_var = input_var_ipw)
    
    l_in_refl = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_refl)
    
    
    
    l_conv1_refl = dnn.Conv2DDNNLayer(
            l_in_refl,
            num_filters=8,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_conv2_refl = dnn.Conv2DDNNLayer(
            l_conv1_refl,
            num_filters=8,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'valid'
        )
    
    
    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_conv2_refl,p=0.2),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
    network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return network,l_hidden1

def build_2DCNN_softmax_mod_special(input_var_ipw = None,input_var_refl = None):
    
    print('2 CNN and a maxpool mod')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in_ipw = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_ipw)
    
    l_in_refl = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_refl)
    
    
    l_conv1_ipw = dnn.Conv2DDNNLayer(
            l_in_ipw,
            num_filters=8,
            filter_size=(7, 7),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_conv2_ipw = dnn.Conv2DDNNLayer(
            l_conv1_ipw,
            num_filters=16,
            filter_size=(7, 7),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'valid'
        )
    
    l_maxpool_ipw = dnn.MaxPool2DDNNLayer(l_conv2_ipw,(2,2))
    
    conv_shape1 = lasagne.layers.get_output_shape(l_maxpool_ipw)
    
    print conv_shape1
    
    l_conv1_refl = dnn.Conv2DDNNLayer(
            l_in_refl,
            num_filters=8,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_conv2_refl = dnn.Conv2DDNNLayer(
            l_conv1_refl,
            num_filters=16,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'valid'
        )
    
    l_maxpool_refl = dnn.MaxPool2DDNNLayer(l_conv2_refl,(2,2))
    
    conv_shape2 = lasagne.layers.get_output_shape(l_maxpool_refl)
    
    print conv_shape2
    
    l_concat = lasagne.layers.concat([lasagne.layers.reshape(l_maxpool_ipw,([0],-1)),
                                      lasagne.layers.reshape(l_maxpool_refl,([0],-1))],
                                    axis = 1)
    
    concat_shape = lasagne.layers.get_output_shape(l_concat)
    
    print concat_shape
    
    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_concat,p=0.2),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
    network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return network,l_hidden1

def build_DCNN_softmax_special(input_var_ipw = None,input_var_refl = None):
    
    print('Single layer conv net')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in_ipw = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_ipw)
    
    l_in_refl = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_refl)
    
    
    l_conv1_ipw = dnn.Conv2DDNNLayer(
            l_in_ipw,
            num_filters=8,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    conv_shape1 = lasagne.layers.get_output_shape(l_conv1_ipw)
    print conv_shape1
    
    l_conv1_refl = dnn.Conv2DDNNLayer(
            l_in_refl,
            num_filters=8,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    conv_shape2 = lasagne.layers.get_output_shape(l_conv1_refl)
    
    print conv_shape2
    
    l_concat = lasagne.layers.concat([l_conv1_ipw,l_conv1_refl],axis = 1)
    concat_shape = lasagne.layers.get_output_shape(l_concat)
    
    print concat_shape
    
    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_concat,p=0.2),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
    network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return network,l_hidden1

def build_DCNN_softmax_mod_special(input_var_ipw = None,input_var_refl = None):
    
    print('Single Convolution layer with different spatial extent')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in_ipw = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_ipw)
    
    l_in_refl = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_refl)
    
    
    l_conv1_ipw = dnn.Conv2DDNNLayer(
            l_in_ipw,
            num_filters=8,
            filter_size = (7, 7),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    conv_shape1 = lasagne.layers.get_output_shape(l_conv1_ipw)
    print conv_shape1
    
    l_conv1_refl = dnn.Conv2DDNNLayer(
            l_in_refl,
            num_filters=8,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    conv_shape2 = lasagne.layers.get_output_shape(l_conv1_refl)
    
    print conv_shape2
    
    
    l_concat = lasagne.layers.concat([lasagne.layers.reshape(l_conv1_ipw,([0],-1)),
                                      lasagne.layers.reshape(l_conv1_refl,([0],-1))],
                                        axis = 1)
    
    concat_shape = lasagne.layers.get_output_shape(l_concat)
    
    print concat_shape
    
    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_concat,p=0.2),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
    network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return network,l_hidden1


def build_DCNN_softmax(input_var = None):
    
    print('Single layer conv net')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None,4,33,33),
                                        input_var=input_var)
    
    l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=16,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
        
    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_conv1,p=0.2),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
    network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return network,l_hidden1

def build_DCNN_softmax_refl(input_var_refl = None):
    
    print('Single layer conv net')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None,4,33,33),
                                        input_var=input_var_refl)
    
    l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=8,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
        
    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_conv1,p=0.2),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
    network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return network,l_hidden1

def simple_neural_network(input_var = None):
    
    print 'Training a simple neural network...'
    
    l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
    
    l_hidden1 = lasagne.layers.DenseLayer(
            l_in,
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
    
    
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hidden1,0.4),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return network,l_hidden1
    
def build_DCNN_maxpool_softmax(input_var = None):
    
        print('Training 1 CNN max pool network!!')
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
    
        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=16,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
        
        l_maxpool = dnn.MaxPool2DDNNLayer(l_conv1,(2,2))
    
        l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_maxpool,p=0.2),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
        return network,l_hidden1

def build_2layer_cnn_maxpool(input_var = None):
    
#    from lasagne.layers import Conv2DLayer, MaxPool2DLayer
    print('Training 2 layer CNN-max pool network!!')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None,4,33,33),
                                        input_var=input_var)
                                        
    l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=16,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    l_maxpool1 = dnn.MaxPool2DDNNLayer(l_conv1,(2,2))
        
    l_conv2 = dnn.Conv2DDNNLayer(
            l_maxpool1,
            num_filters=16,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_maxpool2 = dnn.MaxPool2DDNNLayer(l_conv2,(2,2))
        
    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_maxpool2,0.4),
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
    network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return network,l_hidden1

def build_2layer_cnn_maxpool_2(input_var = None):
    
#    from lasagne.layers import Conv2DLayer, MaxPool2DLayer
    print('Training 2 layer CNN max-pool network!!')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None,4,33,33),
                                        input_var=input_var)
                                        
    l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=16,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    l_maxpool1 = dnn.MaxPool2DDNNLayer(l_conv1,(2,2))
        
    l_conv2 = dnn.Conv2DDNNLayer(
            l_maxpool1,
            num_filters=32,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_maxpool2 = dnn.MaxPool2DDNNLayer(l_conv2,(2,2))
        
    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_maxpool2,p=0.4),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
    network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    return network,l_hidden1

def build_CNN_softmax(input_var = None):
    
        from lasagne.layers import Conv2DLayer
        print('Training the softmax network!!')
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
    
        l_conv1 = Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv1,
            num_units=2000,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )
        
        network = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
        return network,l_hidden1

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

def convert_gpu_cpu(net):
    from lasagne.layers import dnn
    print('Starting to convert gpu network to cpu network')
    params = lasagne.layers.get_all_param_values(net)
    return params
#    cpu_n_file = file('../output/' + 'CPU_' + network_file,'wb')
#    pkl.dump(params,cpu_n_file,protocol = pkl.HIGHEST_PROTOCOL)
#    cpu_n_file.close()
#    print('Done!')

def load_data_to_memory(point_files,val_indices):
    '''Caution!! this routine loads the entire data set to memory,
    not sure if this is the ay to go but definately speeds up the 
    process
    '''
    base_path = '../data/TrainTest/points/'
    val_batches = []
    X_train_full_data = []

    for ea_point in point_files:
        temp_matrix = np.load(base_path + ea_point)
        # Add this index for reflectivity features alone [:,4:,...]
        X_train = np.vstack(map(lambda x: temp_matrix[x][1],[i for i in range(len(temp_matrix)) if i not in val_indices])).astype('float')
        Y_train = np.vstack(map(lambda x: temp_matrix[x][2],[i for i in range(len(temp_matrix)) if i not in val_indices]))
        X_mean = X_train.mean(axis = 0)
        X_train -= X_mean
        X_val = np.vstack(map(lambda x: temp_matrix[x][1],val_indices)).astype('float')
        X_val -= X_mean
        Y_val = np.vstack(map(lambda x: temp_matrix[x][2],val_indices))
        val_batches.append((X_val,Y_val))
        # over here we are going to kill 1 sample at random and divide the 
        # training examples to 4 batches
        X_train_full_data.append((X_train,Y_train))
    
    x_val = np.vstack(map(lambda x: x[0] ,val_batches))
    y_val = np.vstack(map(lambda x: x[1],val_batches))
    
#    X_means = map(lambda x: x[2],val_batches)
    
    print 'Shape of validation set (%d,%d) '%(x_val.shape[0],x_val.shape[1])
    print 'Shape of validation set ground truth %d '%(y_val.shape[0])
    
    x_train = np.vstack(map(lambda x: x[0] ,X_train_full_data))
    y_train = np.vstack(map(lambda x: x[1] ,X_train_full_data))
    
    print 'Shape of training set (%d,%d) '%(x_train.shape[0],x_train.shape[1])
    print 'Shape of training set ground truth %d '%y_train.shape[0]
    
    print 'Size of training set %d bytes '%x_train.nbytes
    print 'Size of validation set %d bytes bytes '%x_val.nbytes
    
#    print 'Mean list %d '%len(X_means)
    
    return x_train, y_train, x_val, y_val
    
def conv_net(tr_block,val_block,num_epochs,exp_no,load_model_weights = False,model_file_name = ''):
    #------------------------------------------
    # Model
    
    input_var_ipw = T.tensor4('inputs')
    
    input_var_refl = T.tensor4('inputs')
    
    target_var = T.ivector('targets')
    
    net,l1_hidden = build_DCNN_softmax_refl(input_var_refl)
    
#    net,l1_hidden = build_2DCNN_softmax_special(input_var_ipw,input_var_refl)
    
    l2_penelty = regularize_layer_params(l1_hidden,l2)
    
    prediction = lasagne.layers.get_output(net)
    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    
    loss = loss.mean() + l2_penelty
    
    if load_model_weights:
        print 'Loading existing model parameters...'
        model_file_name = '../data/1CNNneural_network_refl_' + str(exp_no) + '_100.pkl'
        model_file = file(model_file_name,'rb')
        model_weights = pkl.load(model_file)
        model_file.close()
        lasagne.layers.set_all_param_values(net,model_weights)
        
    
    params = lasagne.layers.get_all_params(net, trainable=True)
    
#    updates = lasagne.updates.adadelta(loss, params)
    
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0001, momentum=0.9)

    test_prediction = lasagne.layers.get_output(net, deterministic=True)
    
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    
    test_loss = test_loss.mean()
    
    test_accuracy = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
                
    train_fn = theano.function([input_var_refl, target_var], loss, updates=updates)
    
    val_fn = theano.function([input_var_refl, target_var], [test_accuracy,test_loss])
    #------------------------------------------
    performance_metrics = {}
    base_path = '../data/TrainTest/points/'
    point_files = filter(lambda x: x[-4:] == '.pkl',os.listdir(base_path))
    val_indices = determine_indices(base_path + point_files[8],val_block)
    first_pass = True
#    val_batches = []
#    X_train_full_data = []
    for ep in range(num_epochs):
        performance_metrics[ep + 1] = []
        print 'Train Model for epoch: %d'%(ep)
        print '-'*50
        train_err = 0.
        train_batches = 0
        if first_pass:
            print 'Loading the entire data to memory!!!!'
            X_train, Y_train, X_val, Y_val = load_data_to_memory(point_files,val_indices)
            first_pass = False
            
#        X_mean_list = []
#        for ea_point in point_files[:2]:
#            temp_matrix = np.load(base_path + ea_point)
#            # Add this index for reflectivity features alone [:,4:,...]
#            X_train = np.vstack(map(lambda x: temp_matrix[x][1][:,4:,...],[i for i in range(len(temp_matrix)) if i not in val_indices])).astype('float')
#            Y_train = np.vstack(map(lambda x: temp_matrix[x][2],[i for i in range(len(temp_matrix)) if i not in val_indices]))
#            X_mean = X_train.mean(axis = 0)
#            X_train -= X_mean
#            if not first_pass:
#                X_val = np.vstack(map(lambda x: temp_matrix[x][1][:,4:,...],val_indices))
#                Y_val = np.vstack(map(lambda x: temp_matrix[x][2],val_indices))
#                val_batches.append((X_val,Y_val,X_mean))
#            # over here we are going to kill 1 sample at random and divide the 
#            # training examples to 4 batches
#            X_train_full_data.append((X_train,Y_train))
        
        
        for X_,Y_ in iterate_minibatches(X_train,Y_train, 250, shuffle = True):
            train_err += train_fn(X_[:,4:,...].astype('float32'),Y_.reshape(-1,))
            train_batches += 1
        print 'Number of batches %d '%train_batches
        print 'Training loss = %.6f'%(train_err/train_batches)
    
        val_acc = 0.
        val_loss = 0.
        
        val_batches_ctr = 0
#        if not first_pass:
#            
#            x_batch = np.vstack(map(lambda x: x[0] ,val_batches))
#            y_batch = np.vstack(map(lambda x: x[1],val_batches))
##            x_batch_means = np.vstack(map(lambda x: x[2],val_batches))
#            del val_batches
#        x_batch -= X_mean
        
        for x_val_batch,y_val_batch in iterate_minibatches(X_val,Y_val,250,shuffle = True):
#            x_val_batch = x_val_batch.astype('float')
            temp = val_fn(x_val_batch[:,4:,...].astype('float32'),y_val_batch.reshape(-1,))
            val_acc += temp[0]
            val_loss += temp[1]
            val_batches_ctr+=1

        print 'Number of validation batches %d'%val_batches_ctr
        
        print 'Validation accuracy for epoch %d = %.6f'%(ep,val_acc/val_batches_ctr)
        print 'Validation loss for epoch %d = %.6f'%(ep,val_loss/val_batches_ctr)
        
        performance_metrics[ep + 1].append({'val_loss': val_loss / val_batches_ctr,
                                            'val_acc' : val_acc / val_batches_ctr,
                                            'train_loss' : train_err / train_batches})
        
        if (ep+ 1) % 10 == 0:
            params = convert_gpu_cpu(net)
            network_file_name = '1CNN_0maxpool_2048neural_network_p20_refl8'
            network_file = file('../output/'+ network_file_name + '_' + str(exp_no) +'_' + str(ep + 1) + '.pkl','wb')
            pkl.dump(params,network_file,protocol = pkl.HIGHEST_PROTOCOL)
            network_file.close()
            f1 = file('../output/performance_metrics_' + network_file_name + '_' + str(exp_no) + '.pkl','wb')
            pkl.dump(performance_metrics,f1,protocol = pkl.HIGHEST_PROTOCOL)
            f1.close()

def main(data_builder,make_data_set = True):
    
    training_blocks,validation_blocks = build_training_validation_sets(data_builder)
    if make_data_set:
        make_dataset_NN_2(data_builder)
    for i in range(4):
        print '-'*50
        print 'Validation year = %s Month = %s'%tuple(blocks[i].split('_'))
        conv_net(training_blocks[i],validation_blocks[i],200,i)

if __name__ == '__main__':
    data_builder = BuildDataSet.dataset(num_points = 500)
    main(data_builder)    


