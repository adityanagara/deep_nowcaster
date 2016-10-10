# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:49:57 2016

@author: adityanagarajan
"""

import numpy as np
import os
import cPickle as pkl
import BuildDataSet
import lasagne
from theano import tensor as T
import theano
import ModelMetrics
from matplotlib import pyplot as plt

import re

yr_mon = {14: [5,6,7,8],
              15: [5,6],
              16: [5,6,7]}


blocks = [str(yr) + '_' + str(mon) for yr in [14,15,16] for mon in yr_mon[yr]]

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


def build_DCNN_softmax(input_var = None,ipw_refl = False):
    
        print('Single layer conv net')
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        if not ipw_refl:
            l_in = lasagne.layers.InputLayer(shape=(None,4,33,33),
                                        input_var=input_var)
        else:
            l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
    
        l_conv1 = lasagne.layers.Conv2DLayer(
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

def build_DCNN_softmax_special(input_var_ipw = None,input_var_refl = None):
    
    print('Single layer conv net')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in_ipw = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_ipw)
    
    l_in_refl = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_refl)
    
    
    l_conv1_ipw = lasagne.layers.Conv2DLayer(
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
    
    l_conv1_refl = lasagne.layers.Conv2DLayer(
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

def build_2DCNN_softmax_special_refl(input_var_refl = None):
    
    print('2 CNN refl special')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
#    l_in_ipw = lasagne.layers.InputLayer(shape = (None,4,33,33),
#                                        input_var = input_var_ipw)
    
    l_in_refl = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_refl)
    
    
    
    l_conv1_refl = lasagne.layers.Conv2DLayer(
            l_in_refl,
            num_filters=8,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_conv2_refl = lasagne.layers.Conv2DLayer(
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

def build_2DCNN_softmax_special(input_var_ipw = None,input_var_refl = None):
    
    print('Single layer conv net')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in_ipw = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_ipw)
    
    l_in_refl = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_refl)
    
    
    l_conv1_ipw = lasagne.layers.Conv2DLayer(
            l_in_ipw,
            num_filters=8,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_conv2_ipw = lasagne.layers.Conv2DLayer(
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
    
    l_conv1_refl = lasagne.layers.Conv2DLayer(
            l_in_refl,
            num_filters=8,
            filter_size=(5, 5),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_conv2_refl = lasagne.layers.Conv2DLayer(
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

def build_DCNN_maxpool_softmax(input_var = None,temp_file = ''):
    
        print('Training the CNN maxpool network!!')
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        if 'refl' in temp_file:
            l_in = lasagne.layers.InputLayer(shape=(None,4,33,33),
                                        input_var=input_var)
        else:
            l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
    
    
        l_conv1 = lasagne.layers.Conv2DLayer(
            l_in,
            num_filters=16,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
        
        l_maxpool = lasagne.layers.MaxPool2DLayer(incoming = l_conv1,stride = (2,2))
    
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


def build_2layer_cnn_maxpool_2(input_var = None,temp_file = ''):
    
#    from lasagne.layers import Conv2DLayer, MaxPool2DLayer
    print('Training 2 layer CNN max-pool network!!')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    if 'refl' in temp_file:
        l_in = lasagne.layers.InputLayer(shape=(None,4,33,33),
                                        input_var=input_var)
    else:
        l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
                                        
    l_conv1 = lasagne.layers.Conv2DLayer(
            l_in,
            num_filters=16,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    l_maxpool1 = lasagne.layers.MaxPool2DLayer(l_conv1,(2,2))
        
    l_conv2 = lasagne.layers.Conv2DLayer(
            l_maxpool1,
            num_filters=32,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_maxpool2 = lasagne.layers.MaxPool2DLayer(l_conv2,(2,2))
        
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

'''
for k, layer in net:
    if k.startswith('conv'):
        layer.W.set_value(np.asarray(layer.W.get_value())[:,:,::-1,::-1])
'''
def determine_indices(data_builder,file_name,val_block):
    arr = np.load(file_name)
    val_indices = []
    date_blocks = data_builder.club_days(val_block)
    date_blocks = sorted(date_blocks.keys())
    for i in range(len(arr)):
        if np.all(np.in1d(arr[i][0][0,:2],np.array((val_block[:,1],val_block[:,0])).T)):
            val_indices.append(i)
    return val_indices

def determine_indices_array(data_builder,arr,val_block):
    val_indices = []
    date_blocks = data_builder.club_days(val_block)
    date_blocks = sorted(date_blocks.keys())
    for i in range(len(arr)):
        if np.all(np.in1d(arr[i][0][0,:2],np.array((val_block[:,1],val_block[:,0])).T)):
            val_indices.append(i)
    return val_indices

def load_network(network_path):
    
    f1 = file(network_path,'rb')
    load_network = pkl.load(f1)
    f1.close()
    temp_file = network_path.split('/')[-1]
    
    num_conv_layers,num_max_pool_layers,num_units,ipw_refl = parse_file(temp_file)
    
    input_var = T.tensor4('inputs')
    
    if num_conv_layers == 1:
        input_var_ipw = T.tensor4('inputs')
        input_var_refl = T.tensor4('inputs')
        
        if 'special' in temp_file and 'mod' in temp_file:
            net,hidden = build_DCNN_softmax_mod_special(input_var_ipw,input_var_refl)
            load_network[0] = load_network[0][:,:,::-1,::-1]
            load_network[2] = load_network[2][:,:,::-1,::-1]
        elif 'special' in temp_file:
            net,hidden = build_DCNN_softmax_special(input_var_ipw,input_var_refl)
            load_network[0] = load_network[0][:,:,::-1,::-1]
            load_network[2] = load_network[2][:,:,::-1,::-1]
        else:
            net,hidden = build_DCNN_softmax(input_var_refl,ipw_refl)
            load_network[0] = load_network[0][:,:,::-1,::-1]
    elif num_conv_layers == 2:
        if 'special' in temp_file or 'refl' in temp_file:
            input_var_ipw = T.tensor4('inputs')
            input_var_refl = T.tensor4('inputs')
            if ipw_refl:
                net,hidden = build_2DCNN_softmax_special(input_var_ipw,input_var_refl)
                load_network[0] = load_network[0][:,:,::-1,::-1]
                load_network[2] = load_network[2][:,:,::-1,::-1]
                load_network[4] = load_network[4][:,:,::-1,::-1]
                load_network[6] = load_network[6][:,:,::-1,::-1]
            else:
                net,hidden = build_2DCNN_softmax_special_refl(input_var_refl)
                load_network[0] = load_network[0][:,:,::-1,::-1]
                load_network[2] = load_network[2][:,:,::-1,::-1]
        else:
            net,hidden = build_2layer_cnn_maxpool_2(input_var,temp_file[-1])
    
#    if 'special' in temp_file:
#        load_network[0] = load_network[0][:,:,::-1,::-1]
#        load_network[2] = load_network[2][:,:,::-1,::-1]
#    else:
#        load_network[0] = load_network[0][:,:,::-1,::-1]
    
#    if temp_file[-1][:4] == '2CNN':
#        load_network[2] = load_network[2][:,:,::-1,::-1]

    lasagne.layers.set_all_param_values(net, load_network)
    
    prediction = lasagne.layers.get_output(net,deterministic=True)
    
    if 'special' in temp_file or 'refl' in temp_file:
        if ipw_refl:
            fn = theano.function([input_var_ipw, input_var_refl],prediction)
        else:
            fn = theano.function([input_var_refl],prediction)
    else:
        fn = theano.function([input_var],prediction)
    
    return fn

def parse_file(temp_file):
    
    num_conv_layers = int(re.findall('\d+CNN',temp_file)[0].strip('CNN'))
    
    num_max_pool_layers = int(re.findall('\d+maxpool',temp_file)[0].strip('maxpool'))
    
    num_units = int(re.findall('\d+neural_network',temp_file)[0].strip('neural_network'))
    
    ipw_refl = not 'refl' in temp_file
    
    return num_conv_layers, num_max_pool_layers, num_units, ipw_refl


def get_frames_points(tr_block,val_block,data_builder,network_path):
    
    temp_file = network_path.split('/')[-1]

    prediction_fn = load_network(network_path)
    
    base_path = '../data/TrainTest/points/'
    point_files = filter(lambda x: x[-4:] == '.pkl',os.listdir(base_path))
    val_indices = determine_indices(data_builder,base_path + point_files[8],val_block)
    prediction_list = []
    for ea_point in point_files:
#        print 'Predicting ' + ea_point
        temp_matrix = np.load(base_path + ea_point)
        # Add this index for reflectivity features alone [:,4:,...]
        if 'refl' in temp_file:
            X_train = np.vstack(map(lambda x: temp_matrix[x][1][:,4:,...],[i for i in range(len(temp_matrix)) if i not in val_indices])).astype('float')
        else: 
            X_train = np.vstack(map(lambda x: temp_matrix[x][1],[i for i in range(len(temp_matrix)) if i not in val_indices])).astype('float')
        
#        Y_train = np.vstack(map(lambda x: temp_matrix[x][2],[i for i in range(len(temp_matrix)) if i not in val_indices]))
        X_mean = X_train.mean(axis = 0)
#        X_train -= X_mean
        if 'refl' in temp_file:
            
            X_val = np.vstack(map(lambda x: temp_matrix[x][1][:,4:,...],val_indices))
        else:
            X_val = np.vstack(map(lambda x: temp_matrix[x][1],val_indices))
        Y_val = np.vstack(map(lambda x: temp_matrix[x][2],val_indices))
        X_val_batch = X_val.astype('float') - X_mean
        if 'special' in temp_file:
            Y_preds = prediction_fn(X_val_batch[:,:4,...].astype('float32'),X_val_batch[:,4:,...].astype('float32'))
        else:
            Y_preds = prediction_fn(X_val_batch.astype('float32'))
#        print Y_preds
#        print Y_preds.shape
        prediction_list.append((Y_preds,Y_val))
    
    Y_prediction = np.vstack(map(lambda x: x[0], prediction_list))
    Y_true = np.vstack(map(lambda x: x[1], prediction_list))
    
    performance = ModelMetrics.NOWCAST_performance((np.argmax(Y_prediction,axis = 1),Y_true.reshape(-1,),Y_prediction[:,1]))
    
    print 'Precision = %.2f'%performance.p_score
    print 'Recall = %.2f'%performance.r_score
    print 'F1 score = %.2f'%performance.f1
    print 'Area under the curve = %.2f'%performance.average_precision
    
    print 'POD = %.2f'%performance.POD
    print 'FAR = %.2f'%performance.FAR
    print 'CSI = %.2f'%performance.CSI
    
    return performance

    
def get_frames(PixelPoints,data_builder,validation_set):
#    data_builder = BuildDataSet.dataset(num_points = 500)
#    PixelPoints = data_builder.sample_random_pixels()
    
    yr,mon = validation_set.split('_')
    points_array = []
    for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
#        print 'Building data set for point (%d,%d)'%(x_,y_)
        storm_dates = data_builder.load_storm_days(int(yr))
        storm_dates = storm_dates[storm_dates[:,2] == int(mon)]
        # load the dictionary which gives us the days which are 
        # clubbed together. 
        doy_strings = data_builder.club_days(storm_dates)
        days_in_sorted = doy_strings.keys()
        days_in_sorted.sort()
        
        ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(yr)
            
        for set_ in days_in_sorted:
                
            # Get the required files only
            temp_ipw_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],ipw_files)
            temp_refl_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],refl_files)
            temp_ipw_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_ipw_files)
            temp_refl_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_refl_files)
            
            temp_array = data_builder.build_features_and_truth_imgs(temp_ipw_files,temp_refl_files,x_,y_)
                
            ipw_refl_tensors = data_builder.arrange_frames_single(temp_array)
                
            points_array.append((ipw_refl_tensors))
    
    X_val = np.vstack(map(lambda x: x[1],points_array))
    Y_val = np.vstack(map(lambda x: x[2],points_array))
    return X_val,Y_val
    

    
#    data_builder = BuildDataSet.dataset(num_points = 500)
#    PixelPoints = data_builder.sample_random_pixels()
#    performance_list = []
#    for points in iterate_minibatches(PixelPoints,100):
##        print points
#        print '-'*50
#        X,Y_true = get_frames(points,data_builder,'14_5') 
#        # for refl [:,4:,...]
#        Y_preds = fn(X)
##        out_array.append((Y_preds,Y_true))    
##        Y_prediction = np.vstack(map(lambda x: x[0],out_array))
##        Y_true = np.vstack(map(lambda x: x[1],out_array))
#        performance = ModelMetrics.NOWCAST_performance((np.argmax(Y_preds,axis = 1),Y_true.reshape(-1,),Y_preds[:,1]))
#    
#        print 'Precision = %.2f'%performance.p_score
#        print 'Recall = %.2f'%performance.r_score
#        print 'F1 score = %.2f'%performance.f1
#        print 'Area under the curve = %.2f'%performance.average_precision
#    
#        print 'POD = %.2f'%performance.POD
#        print 'FAR = %.2f'%performance.FAR
#        print 'CSI = %.2f'%performance.CSI
#        
#        performance_list.append(performance)
    
#    average_POD = sum(map(lambda x: x.POD, performance_list)) / len(performance_list)
#    average_FAR = sum(map(lambda x: x.FAR, performance_list)) / len(performance_list)
#    average_CSI = sum(map(lambda x: x.CSI, performance_list)) / len(performance_list)
#    average_AUC = sum(map(lambda x: x.average_precision, performance_list)) / len(performance_list)
#    
#    print 'Average POD = %.4f '%average_POD
#    print 'Average FAR = %.4f '%average_FAR
#    print 'Average CSI = %.4f '%average_CSI
#    print 'Average AUC = %.4f '%average_AUC
    
#    f1 = file('performance_list_ipw_refl' + val_set + '.pkl','wb')
#    pkl.dump(performance_list,f1,protocol = pkl.HIGHEST_PROTOCOL)
#    f1.close()
    
    

def calc_metrics(network_path):
    
    temp_file = network_path.split('/')[-1]
    
    data_builder = BuildDataSet.dataset(num_points = 500)
    
    val_set = int(re.findall('\d+',temp_file)[-2])
    
    
#    network_params['num_conv_layers'] = num_conv_layers
    
    training_blocks,validation_blocks = build_training_validation_sets(data_builder)
    
    performance = get_frames_points(training_blocks[val_set],validation_blocks[val_set],data_builder,network_path)
        
    return (performance.POD,performance.FAR,performance.CSI,performance.average_precision)
    
    
'''
Model after 100 epochs 1 CNN no maxpool ipw + refl
1CNNneural_network_0_100.pkl

Model after 100 epochs 1 CNN no maxpool refl
1CNNneural_network_refl_0_100.pkl

Model after 100 epochs for 2 CNN 2 max pool refl only
2CNN_2maxpool_neural_network_0_100.pkl

Model after 100 epochs for 2 CNN 2 max pool refl only
2CNN_2maxpool_neural_network_refl_run2_0_10.pkl
'''

def get_network_files():
    
    cnn1_templet_ipw_refl = {'ipw_refl' : ['1CNNneural_network_0_{}.pkl'.format(x) for x in range(10,110,10)],
                            'refl' : ['1CNNneural_network_refl_0_{}.pkl'.format(x) for x in range(10,110,10)]}
    
    cnn2_templet_ipw_refl = {'ipw_refl' : ['2CNN_2maxpool_neural_network_0_{}.pkl'.format(x) for x in range(10,210,10)],
                              'refl' : ['2CNN_2maxpool_neural_network_refl_0_{}.pkl'.format(x) for x in range(10,100,10)]}

    
    cnn2_templet_ipw_refl['refl'].extend(['2CNN_2maxpool_neural_network_refl_run2_0_{}.pkl'.format(x) for x in range(10,120,10)])
    
    
    return cnn1_templet_ipw_refl,cnn2_templet_ipw_refl
    
'''
Here are some files for you to look at 
1. Rest of the 3 months in 2014
2. 1CNN maxpool?? re you sure this came with a 0.001 training rate?
1CNN_maxpool_neural_network_0_100.pkl
'''
def get_2014CV_files():
    cnn1_templet_ipw_refl = []
    for i in range(4):
        cnn1_templet_ipw_refl.append({'ipw_refl' : ['1CNNneural_network_{0}_{1}.pkl'.format(i,x) for x in range(100,110,10)],
                                        'refl' : ['1CNNneural_network_refl_{0}_{1}.pkl'.format(i,x) for x in range(100,110,10)]})
    
    
    return cnn1_templet_ipw_refl

def get_2014CV_files_convention():
    # 1CNN_0maxpool_2048neural_network_p20_special_2_50.pkl
    # 1CNN_0maxpool_2048neural_network_p20_0_10.pkl
    cnn1_templet_ipw_refl = []
    for i in range(3):
        cnn1_templet_ipw_refl.append({'ipw_refl' : ['2CNN_0maxpool_2048neural_network_p20_special_{0}_{1}.pkl'.format(i,x) for x in range(10,210,10)],
                                        'refl' : ['1CNN_0maxpool_2048neural_network_p20_special_{0}_{1}.pkl'.format(i,x) for x in range(10,210,10)]})
    
    
    return cnn1_templet_ipw_refl

def get_1CNN_experiment_networks():
    # 1CNN_0maxpool_2048neural_network_p20_special_0_10.pkl
    # 1CNN_0maxpool_2048neural_network_p20_refl_0_10.pkl
    cnn1_templet_ipw_refl = []
    for i in range(4):
        cnn1_templet_ipw_refl.append({'ipw_refl' : ['1CNN_0maxpool_2048neural_network_p20_special_{0}_{1}.pkl'.format(i,x) for x in range(10,210,10)],
                                        'refl' : ['1CNN_0maxpool_2048neural_network_p20_refl_{0}_{1}.pkl'.format(i,x) for x in range(10,210,10)]})
    
    return cnn1_templet_ipw_refl

def get_2CNN_experiment_networks():
    # 2CNN_0maxpool_2048neural_network_p20_special_3_10.pkl
    # 2CNN_0maxpool_2048neural_network_p20_refl_3_199.pkl
    cnn2_templet_ipw_refl = []
    for i in range(4):
        cnn2_templet_ipw_refl.append({'ipw_refl' : ['2CNN_0maxpool_2048neural_network_p20_special_{0}_{1}.pkl'.format(i,x) for x in range(10,210,10)],
                                        'refl' : ['2CNN_0maxpool_2048neural_network_p20_refl_{0}_{1}.pkl'.format(i,x) for x in range(9,209,10)]})
    
    return cnn2_templet_ipw_refl

def get_1CNN_mod_networks():
    # 2CNN_0maxpool_2048neural_network_p20_special_3_10.pkl
    # 2CNN_0maxpool_2048neural_network_p20_refl_3_199.pkl
    # 1CNN_0maxpool_2048neural_network_p20_mod_special_0_10.pkl
    cnn2_templet_ipw_refl = []
    for i in range(4):
        cnn2_templet_ipw_refl.append({'ipw_refl' : ['1CNN_0maxpool_2048neural_network_p20_special_{0}_{1}.pkl'.format(i,x) for x in range(10,210,10)],
                                        'refl' : ['1CNN_0maxpool_2048neural_network_p20_mod_special_{0}_{1}.pkl'.format(i,x) for x in range(10,210,10)]})
    
    return cnn2_templet_ipw_refl

def build_DCNN_softmax_mod_special(input_var_ipw = None,input_var_refl = None):
    
    print('Single Convolution layer with different spatial extent')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in_ipw = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_ipw)
    
    l_in_refl = lasagne.layers.InputLayer(shape = (None,4,33,33),
                                        input_var = input_var_refl)
    
    
    l_conv1_ipw = lasagne.layers.Conv2DLayer(
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
    
    l_conv1_refl = lasagne.layers.Conv2DLayer(
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

def main():
    base_path = '../output/thesis_results/'    
#    file_list_1,file_list_2 = get_network_files()
#    list_of_files = get_2014CV_files_convention()
    list_of_files = get_1CNN_mod_networks()    
    file_performance = []    
    for f_list in list_of_files:
        performance_list = []
        for i,r in zip(f_list['ipw_refl'],f_list['refl']):
            print i
            ipw_refl_performance = calc_metrics(base_path + i)
            print ipw_refl_performance
            print r
            refl_performance = calc_metrics(base_path + r)
            print refl_performance
            performance_list.append({'ipw_refl' : ipw_refl_performance,
                                     'refl' : refl_performance})
        file_performance.append(performance_list)
    f1 = file('1CNN_mod_experiment_results.pkl','wb')
    pkl.dump(file_performance,f1)
    f1.close()
               

def plot_curves():
    # '1CNN_experiment_results.pkl' : Contains 1CNN IPW + refl where refl consists of 16 filters 
    # '1CNN_mod_experiment_results.pkl' : Comparison of regular 5 x 5 window for both variables and 5 x 5 and 7 x 7 
    # where the 'refl' key has the values for the modified
    # '2CNN_experiment_results.pkl' : 2CNN architecture 
    f1 = file('2CNN_experiment_results.pkl','rb')
    arr = pkl.load(f1)
    f1.close()
    print len(arr)
    for i in range(4):
        x_axis = range(10,len(arr[i]) * 10 + 10,10)
        CSIs_ipw_refl = map(lambda x: x['ipw_refl'][2],arr[i])
        CSIs_refl = map(lambda x: x['refl'][2],arr[i])
        AUC_ipw_refl = map(lambda x: x['ipw_refl'][3],arr[i])
        AUC_refl = map(lambda x: x['refl'][3],arr[i])
        optimal_epoch_ipw_refl = np.argmax(CSIs_ipw_refl)
        optimal_epoch_refl = np.argmax(CSIs_refl)
        print 'The maximum CSI for validation set %d IPW + refl is %.4f at epoch %d'%(i,max(CSIs_ipw_refl),optimal_epoch_ipw_refl)
        print 'The maximum CSI for validation set %d reflectivity only is %.4f at epoch %d'%(i,max(CSIs_refl),optimal_epoch_refl)
        plt.figure()
        plt.title('CSI plot for every 10 epochs')
        plt.plot(x_axis,CSIs_ipw_refl,label = 'CSI IPW + refl')
        plt.plot(x_axis,CSIs_refl, label = 'CSI refl')
        plt.ylim((0.10,0.60))
        plt.grid()
        plt.legend(loc = 'lower right')
        plt.xlabel('epochs')
        plt.ylabel('CSI score')
        plt.xlim((10,len(arr[i]) * 10 + 10))
        plt.figure()
        plt.title('AUC plot for every 10 epochs')
        plt.plot(x_axis,AUC_ipw_refl, label = 'AUC IPW + refl')
        plt.plot(x_axis,AUC_refl, label = 'AUC refl')
        plt.legend(loc = 'lower right')
        plt.grid()
        plt.ylim((0.20,0.80))
        plt.xlabel('epochs')
        plt.ylabel('AUC of Precision Recall curve')
        plt.xlim((10,len(arr[i]) * 10 + 10))
        print 'Optimal values for IPW + reflectivity'
        print arr[i][optimal_epoch_ipw_refl]['ipw_refl']
        print 'Optimal values for reflectivity'
        print arr[i][optimal_epoch_refl]['refl']

#def determine_optimal_epoch(file_name):
#    f1 = file(file_name,'rb')
#    arr = pkl.load(f1)
#    f1.close()
    

def nowcast_storm(data_builder,val_block,network_path):
    '''Builds the 4 frames of ipw and 4 frames of reflectivity. The way this
    is built is we make num_points = 500 files where each file contains all storm
    dates for a pixel point. So in minibatch SGD we train on each point'''
    storm_dates_all = {}
    network_temp_file = network_path.split('/')[-1]
    prediction_fn = load_network(network_path)
    # all points GG wp
    domain_points = (range(17,83),range(17,83))
    
    PixelPoints = [(x,y) for x in domain_points[0]  
                    for y in domain_points[1]]
    
    PixelPoints = np.array(PixelPoints)
    
    print PixelPoints.shape
    prediction_list = []
    for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
#        print 'Building data set for point (%d,%d)'%(x_,y_)
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
                
        # we now reach this point where we have all the points in this weird
        # data structure that we have developed WTF???
        # time to normalize and take only refl buddy you so fucked!!
        
        # Save each string of days to a pkl file
        val_indices = determine_indices_array(data_builder,points_array,val_block)
        
        temp_matrix = points_array
        # Add this index for reflectivity features alone [:,4:,...]
        if 'refl' in network_temp_file:
            X_train = np.vstack(map(lambda x: temp_matrix[x][1][:,4:,...],[i for i in range(len(temp_matrix)) if i not in val_indices])).astype('float')
        else: 
            X_train = np.vstack(map(lambda x: temp_matrix[x][1],[i for i in range(len(temp_matrix)) if i not in val_indices])).astype('float')
        
#        Y_train = np.vstack(map(lambda x: temp_matrix[x][2],[i for i in range(len(temp_matrix)) if i not in val_indices]))
        X_mean = X_train.mean(axis = 0)
        if 'refl' in network_temp_file:
            X_val = np.vstack(map(lambda x: temp_matrix[x][1][:,4:,...],val_indices))
        
        else:
            X_val = np.vstack(map(lambda x: temp_matrix[x][1],val_indices))
        
        Y_val = np.vstack(map(lambda x: temp_matrix[x][2],val_indices))
        X_val_batch = X_val.astype('float') - X_mean
        # X_val_batch[:,:4,...].astype('float32'),X_val_batch[:,4:,...].astype('float32')
        
        if 'refl' in network_temp_file:
            Y_preds = prediction_fn(X_val_batch.astype('float32'))
        else:
            Y_preds = prediction_fn(X_val_batch[:,:4,...].astype('float32'),X_val_batch[:,4:,...].astype('float32'))
        
#        print Y_preds.shape,Y_val.shape
        
        prediction_list.append((Y_val,Y_preds))
        
#        Y_prediction = np.vstack(map(lambda x: x[0], prediction_list))
#        Y_true = np.vstack(map(lambda x: x[1], prediction_list))
    
    return prediction_list

def call_nowcast_storm():
    # Based of the optimal epoch measured against CSI
    # 2CNN_0maxpool_2048neural_network_p20_special_0_40.pkl
    # 2CNN_0maxpool_2048neural_network_p20_special_1_50.pkl
    # ------------------------------------------------------# 
    # 2CNN Network list for IPW + refl
#    network_list = ['2CNN_0maxpool_2048neural_network_p20_special_0_40.pkl',
#                    '2CNN_0maxpool_2048neural_network_p20_special_1_50.pkl',
#                    '2CNN_0maxpool_2048neural_network_p20_special_2_140.pkl',
#                    '2CNN_0maxpool_2048neural_network_p20_special_3_110.pkl']
    # ------------------------------------------------------# 
    # 2CNN Network list for refl
    network_list = ['2CNN_0maxpool_2048neural_network_p20_refl_0_79.pkl',
                    '2CNN_0maxpool_2048neural_network_p20_refl_1_99.pkl',
                    '2CNN_0maxpool_2048neural_network_p20_refl_2_39.pkl',
                    '2CNN_0maxpool_2048neural_network_p20_refl_3_19.pkl']
    # ------------------------------------------------------#
    data_builder = BuildDataSet.dataset(num_points = 500)
    base_path = '../output/thesis_results/'
    training_blocks,validation_blocks = build_training_validation_sets(data_builder)
    for network_file in network_list:
        network_path = base_path + network_file 
        temp_file = network_path.split('/')[-1]
        val_set = int(re.findall('\d+',temp_file)[-2])
        print 'Running Validation Set %d'%val_set
        prediction_list = nowcast_storm(data_builder,validation_blocks[val_set],network_path)
        f1 = file('2CNN_prediction_file_refl_' + str(val_set) + '.pkl','wb')
        pkl.dump(prediction_list,f1,protocol = pkl.HIGHEST_PROTOCOL)
        f1.close()    
    
if __name__ == '__main__':
    # performance_metrics_1CNN_0maxpool_2048neural_network_p20_0.pkl
    generate_results = False
    generate_predictions = True
    if generate_results:
        main()
    elif generate_predictions:
        call_nowcast_storm()
#    plot_curves()
#    plt.show()


    