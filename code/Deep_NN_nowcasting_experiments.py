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
                print 'Building data set for year: %d and string of days %s'%(yr,set_)
                
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
#        print excerpt
        yield inputs[excerpt], targets[excerpt]

def conv_net(tr_block,val_block):
    print tr_block,val_block
    
def main(make_data_set = False):
    data_builder = BuildDataSet.dataset(num_points = 500)
    training_blocks,validation_blocks = build_training_validation_sets(data_builder)
    if make_data_set:
        make_dataset_NN_2(data_builder)
    conv_net(training_blocks,validation_blocks)
    
#    train,validation = arrange_training_validation(training_blocks,validation_blocks)
    
    
#    if convert_to_images:
##        convert_fields_to_images(data_builder)
#        make_dataset_NN(data_builder)
#    train,validation = arrange_training_validation(training_blocks,validation_blocks)
#    
#    exp_no = 0
#    for tr,val in zip(train,validation):
#        print 'Train Test Split %d '%(exp_no + 1)
#        neural_network_model_new(tr,val,exp_no+1)
##    convolution_neural_network_model(train[0],validation[0], exp_no + 1,100)
##        empty_network(tr,val,exp_no+1,1)
#        exp_no+=1
        

if __name__ == '__main__':
    main()    


