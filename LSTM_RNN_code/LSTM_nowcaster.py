# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:39:42 2016

@author: adityanagarajan
"""

import numpy as np
import re
import BuildDataSet
import os
import cPickle as pkl

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.optimizers import RMSprop
data_builder = BuildDataSet.dataset(num_points = 500)

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


def process_dataset(ea_point,block):
    base_path = '../data/TrainTest/points/'
    indices = determine_indices(base_path + ea_point,block)
    temp_matrix = np.load(base_path + ea_point)
    X_train = np.vstack(map(lambda x: temp_matrix[x][1],[i for i in range(29) if i not in indices]))
    Y_train = np.vstack(map(lambda x: temp_matrix[x][2],[i for i in range(29) if i not in indices]))
    return X_train.reshape((-1,4,2,33,33),order = 'F'),Y_train

def process_dataset_validation(ea_point,block):
    base_path = '../data/TrainTest/points/'
    indices = determine_indices(base_path + ea_point,block)
    temp_matrix = np.load(base_path + ea_point)
    X_val = np.vstack(map(lambda x: temp_matrix[x][1],[i for i in range(29) if i in indices]))
    Y_val = np.vstack(map(lambda x: temp_matrix[x][2],[i for i in range(29) if i in indices]))
    return X_val.reshape((-1,4,2,33,33),order = 'F'),Y_val
    
    
def generate_arrays_from_file(val_block):
    base_path = '../data/TrainTest/points/'
    while 1:
        point_files = filter(lambda x: x[-4:] == '.pkl',os.listdir(base_path))
        for i,ea_point in enumerate(point_files):
            # create numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_dataset(ea_point,val_block)
            
            yield (x.reshape(-1,4,2*33*33), y)

def generate_arrays_from_file_validation(val_block):
    base_path = '../data/TrainTest/points/'
    while 1:
        point_files = filter(lambda x: x[-4:] == '.pkl',os.listdir(base_path))
        for i,ea_point in enumerate(point_files):
            # create numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_dataset_validation(ea_point,val_block)
            
            yield (x.reshape(-1,4,2*33*33), y)



def lstm_model(val_block):
    points = 500
#    for a in generate_arrays_from_file(val_block):
#        print a[0].shape,a[1].shape
    
#    for b in generate_arrays_from_file_validation(val_block):
#        print b[0].shape,b[1].shape
    
    model = Sequential()
    model.add(LSTM(1024, input_shape = (4,2*33*33),return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # try using different optimizers and different optimizer configs
    optimizer = RMSprop(lr=0.0001)
    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    
    mdl_fit = model.fit_generator(generate_arrays_from_file(val_block),
                                  samples_per_epoch = 2424*points,
                                  nb_epoch = 100, verbose=1,
                                  validation_data=generate_arrays_from_file_validation(val_block),
                                    nb_val_samples = 551*points, 
                                    show_accuracy=True)
    
#    print mdl_fit.history
    
    model.save_weights('my_model_weights.h5')
#    model.load_weights('my_model_weights.h5')
    f1 = file('results.pkl','w+')
    pkl.dump(mdl_fit.history,f1)
    f1.close()
    

def main():
    data_builder = BuildDataSet.dataset(num_points = 500)
    training_blocks,validation_blocks = build_training_validation_sets(data_builder)
#    make_dataset_NN_2(data_builder)
    lstm_model(validation_blocks[0])
    
if __name__ == '__main__':
    main()

'''
def conv_net(tr_block,val_block,num_epochs,exp_no):
    #------------------------------------------
    # Model
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    net,l1_hidden = build_2layer_cnn_maxpool_2(input_var)
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

'''

