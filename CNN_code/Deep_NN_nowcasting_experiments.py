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
#from lasagne.layers import dnn

from lasagne.regularization import regularize_layer_params, l2, l1

import cPickle as pkl

def get_val_doys(storm_dates):
    '''May: 121-151, June: 152 - 181 July: 182 - 212 August: 213 - 243'''
    yr_mon = {14: [5,6,7,8],
              15: [5,6,7,8]}
    blocks = [str(yr) + '_' + str(mon) for yr in yr_mon.keys() for mon in yr_mon[yr]]
    val_blocks = []
    train_blocks = []
    for bl in blocks[:-1]:
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
    for yr in [14,15]:
        storm_dates_all[yr] = data_builder.load_storm_days(yr)    
    storm_dates_all = np.concatenate((storm_dates_all[14],storm_dates_all[15]))
    train,val = get_val_doys(storm_dates_all)
    return train,val

def arrange_training_validation(tr,val):
    '''return a liat of tuples containing 7 elements where each tuple represents
    the files needed to be loaded for the training and validation sets'''
    file_list = []
    for yr in [14,15]:
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
    '''Builds the four frames of ipw and 4 frames of reflectivity. The way this
    is built is we make num_points = 500 files where each file contains all storm
    dates for a pixel point'''
    storm_dates_all = {}
    PixelPoints = data_builder.sample_random_pixels()
    print PixelPoints.shape
    
    for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
        print 'Building data set for point (%d,%d)'%(x_,y_)
        points_array = []
        for yr in [14,15]:
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
        print '../data/TrainTest/points/' + 'IPW_refl_frames{0}_{1}.pkl'.format(x_,y_)
        temp_file = file('../data/TrainTest/points/' + 'IPW_refl_frames{0}_{1}.pkl'.format(x_,y_),'wb')
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

def build_DCNN_softmax(input_var = None):
    
        print('Training the softmax network!!')
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
    
        l_conv1 = dnn.Conv2DDNNLayer(
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

def build_DCNN_maxpool_softmax(input_var = None):
    
        print('Training the maxpool network!!')
        # Define the input variable which is 4 frames of IPW fields and 4 frames of 
        # reflectivity fields
        l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
    
        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=32,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
        
        l_maxpool = dnn.MaxPool2DDNNLayer(l_conv1,(2,2))
    
        l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_maxpool,p=0.3),
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

def build_2layer_cnn_maxpool(input_var = None):
    
#    from lasagne.layers import Conv2DLayer, MaxPool2DLayer
    print('Training 2 layer CNN-max pool network!!')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
                                        
    l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=8,
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
            num_filters=16,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
    
    l_maxpool2 = dnn.MaxPool2DDNNLayer(l_conv2,(2,2))
        
    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_maxpool2,p=0.3),
            num_units=100,
            nonlinearity=lasagne.nonlinearities.sigmoid,
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
    print('Training 2 layer CNN-max pool network!!')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
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
            num_units=100,
            nonlinearity=lasagne.nonlinearities.sigmoid,
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

    
def conv_net(tr_block,val_block,num_epochs,exp_no):
    #------------------------------------------
    # Model
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    net,l1_hidden = build_DCNN_softmax_cpu(input_var)
    l2_penelty = regularize_layer_params(l1_hidden,l2)
    prediction = lasagne.layers.get_output(net)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + l2_penelty
    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0001, momentum=0.9)

    test_prediction = lasagne.layers.get_output(net, deterministic=True)
    
    prediction = T.argmax(test_prediction, axis=1)
    
    val_accuracy = T.mean(T.eq(prediction, target_var),
                      dtype=theano.config.floatX)
    
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    
    
    
    hits = T.sum(T.and_(T.eq(prediction,target_var),T.eq(target_var,1))) + 1e-16
    misses = T.sum(T.and_(T.neq(prediction,target_var),T.eq(target_var,1))) + 1e-16
    false_alarms = T.sum(T.and_(T.neq(prediction,target_var),T.eq(target_var,0)))
    
    POD = hits / (hits + misses)
    FAR = false_alarms / (false_alarms + hits)
    CSI = hits / (hits + misses + false_alarms)
    
    val_fn = theano.function([input_var, target_var], [val_accuracy,POD,FAR,CSI])
    #------------------------------------------
    performance_metrics = {}
    base_path = '../data/TrainTest/points/'
    point_files = filter(lambda x: x[-4:] == '.pkl',os.listdir(base_path))
    val_indices = determine_indices(base_path + point_files[8],val_block)
    print val_indices
    first_pass = False
    val_batches = []
    for ep in range(num_epochs):
        performance_metrics[ep] = []
        print 'Train Model for epoch: %d'%(ep+1)
        print '-'*50
        train_err = 0.
        train_batches = 0
        for ea_point in point_files:
            temp_matrix = np.load(base_path + ea_point)
            X_train = np.vstack(map(lambda x: temp_matrix[x][1],[i for i in range(29) if i not in val_indices]))
            Y_train = np.vstack(map(lambda x: temp_matrix[x][2],[i for i in range(29) if i not in val_indices]))
            if not first_pass:
                X_val = np.vstack(map(lambda x: temp_matrix[x][1],val_indices))
                Y_val = np.vstack(map(lambda x: temp_matrix[x][2],val_indices))
                val_batches.append((X_val,Y_val))
            train_err += train_fn(X_train,Y_train.reshape(-1,))
            train_batches+=1
        print 'Training loss = %.6f'%(train_err/train_batches)
        
        first_pass = True
            
        val_acc = 0.
        val_POD = 0.
        val_FAR = 0.
        val_CSI = 0.
        
        val_batches_ctr = 0
        
        for batch in val_batches:
            x_batch,y_batch = batch
            temp = val_fn(x_batch,y_batch.reshape(-1,))
            val_acc += temp[0]
            val_POD += temp[1]
            val_FAR += temp[2]
            val_CSI += temp[3]
            val_batches_ctr+=1
        
        
        print 'Validation accuracy for epoch %d = %.6f'%(ep,val_acc/val_batches_ctr)
        print 'Probability of detection for epoch %d = %.6f'%(ep,val_POD/val_batches_ctr)
        print 'False alarm rate for epoch %d = %.6f '%(ep,val_FAR/val_batches_ctr)
        print 'CSI for epoch %d = %.6f'%(ep,val_CSI/val_batches_ctr)
        
        performance_metrics[ep].append([val_acc / val_batches_ctr,val_POD / val_batches_ctr,val_FAR / val_batches_ctr,val_CSI / val_batches_ctr])
        if (ep+ 1) % 10 == 0:
            network_file_name = '2_CNN_layer_max_pool_2'
            network_file = file('../output/'+ network_file_name + '_' + str(exp_no) +'_' + str(ep + 1) + '.pkl','wb')
            pkl.dump(net,network_file,protocol = pkl.HIGHEST_PROTOCOL)
            network_file.close()
    
            f1 = file('../output/performance_metrics_2layer_maxpool_2_' + str(exp_no) + '.pkl','wb')
            pkl.dump(performance_metrics,f1,protocol = pkl.HIGHEST_PROTOCOL)
            f1.close()

def main(make_data_set = False):
    training_blocks,validation_blocks = build_training_validation_sets(data_builder)
    if make_data_set:
        make_dataset_NN_2(data_builder)
    for i in range(7):
        conv_net(training_blocks[i],validation_blocks[i],100,i)

if __name__ == '__main__':
    data_builder = BuildDataSet.dataset(num_points = 500)
    main()    


