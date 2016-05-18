# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:11:21 2016

@author: adityanagarajan
"""

import numpy as np
import BuildDataSet
import re
import os
import cPickle as pkl
import tensorflow as tf

#import ModelMetrics

def weight_variable(shape):
    '''Helper function to create a weight variable initialized with
    a normal distribution

    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    '''Helper function to create a bias variable initialized with
    a constant value.

    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)



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

def convert_fields_to_images(data_builder):
    '''Convert the ipw and reflectivity field arrays to gray scale image 
    arrays'''
    storm_dates_all = {}
    for yr in [14,15]:
        storm_dates_all[yr] = data_builder.load_storm_days(yr)
        doy_strings = data_builder.club_days(storm_dates_all[yr])
        days_in_sorted = doy_strings.keys()
        days_in_sorted.sort()
        ipw_files,refl_files = data_builder.sort_IPW_refl_files(yr)
        ipw_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,ipw_files)
        refl_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,refl_files)
        for i,r in zip(ipw_files,refl_files):
            ipw_array = np.load(i)
            refl_array = np.load(r)
            ipw_img_array = data_builder.convert_IPW_img(ipw_array)
            refl_img_array = data_builder.convert_reflectivity_img(refl_array)
            
            print ipw_img_array.shape
            print refl_img_array.shape
            
            ipw_img_file = i.split('/')[-1].split('.')[0] + '_img' + '.npy'
            refl_img_file = r.split('/')[-1].split('.')[0] + '_img' + '.npy'
            
            print '../data/dataset/20' + str(yr) + os.sep + ipw_img_file
            print '../data/dataset/20' + str(yr) + os.sep + refl_img_file
            
            np.save('../data/dataset/20' + str(yr) + os.sep + ipw_img_file,ipw_img_array)
            np.save('../data/dataset/20' + str(yr) + os.sep + refl_img_file,refl_img_array)

def make_dataset_NN(data_builder):
    '''Builds the four frames of ipw and 4 frames of reflectivity'''
    storm_dates_all = {}
    for yr in [14,15]:
        storm_dates_all[yr] = data_builder.load_storm_days(yr)
        # load the dictionary which gives us the days which are 
        # clubbed together. 
        doy_strings = data_builder.club_days(storm_dates_all[yr])
        days_in_sorted = doy_strings.keys()
        days_in_sorted.sort()
        
        ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(yr)
            
        PixelPoints = data_builder.sample_random_pixels()
        print PixelPoints.shape
        for set_ in days_in_sorted:
            print 'Building data set for year: %d and string of days %s'%(yr,set_)
            points_array = []
            # Get the required files only
            temp_ipw_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],ipw_files)
            temp_refl_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],refl_files)
            temp_ipw_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_ipw_files)
            temp_refl_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_refl_files)
            for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
                temp_array = data_builder.build_features_and_truth_imgs(temp_ipw_files,temp_refl_files,x_,y_)
                
                ipw_refl_tensors = data_builder.arrange_frames_single(temp_array)
                points_array.append((ipw_refl_tensors))
                
            print len(points_array)
            # Save each string of days to a pkl file
            print '../data/TrainTest/20' + str(yr) + os.sep + 'IPW_refl_frames{0}_{1}.pkl'.format(yr,set_)
            temp_file = file('../data/TrainTest/20' + str(yr) + os.sep + 'IPW_refl_frames{0}_{1}.pkl'.format(yr,set_),'wb')
            pkl.dump(points_array,temp_file,protocol = pkl.HIGHEST_PROTOCOL)
            temp_file.close()

def make_dataset_NN_2(data_builder):
    '''Builds the four frames of ipw and 4 frames of reflectivity. The way this
    is built is we make num_points = 500 files where each file contains all storm
    dates for a pixel point'''
    storm_dates_all = {}
    PixelPoints = data_builder.sample_random_pixels()
    print PixelPoints.shape
    
    for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
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
        print len(points_array)
        print '../data/TrainTest/20' + str(yr) + os.sep + 'IPW_refl_frames{0}_{1}.pkl'.format(yr,set_,x_,y_)
#                temp_file = file('../data/TrainTest/20' + str(yr) + os.sep + 'IPW_refl_frames{0}_{1}.pkl'.format(yr,set_),'wb')
#                pkl.dump(points_array,temp_file,protocol = pkl.HIGHEST_PROTOCOL)
#                temp_file.close()


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

def neural_network_model(x_train,y_train,x_val,y_val):
    # we will have 8712 features because 33*33*8
    x = tf.placeholder(tf.float32, [None, 33,33,8])
    # For now binary one hot encoding 
    y = tf.placeholder(tf.float32, [None, 2])
    nn_input = tf.reshape(x, [-1, 33 * 33 * 8])
    
    
    # Build a single layer neural network
    n_fc = 4000
    
    W_fc1 = weight_variable([33 * 33 * 8, n_fc])
    
    b_fc1 = bias_variable([n_fc])
    
    h_fc1 = tf.nn.relu(tf.matmul(nn_input, W_fc1) + b_fc1)
    
    W_fc2 = weight_variable([n_fc, 2])
    
    b_fc2 = bias_variable([2])
    
    prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
    cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
    
    optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
#    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    print 'Training Model...'
    print '-'*50
    print 'Training set'
    print x_train.shape,y_train.shape
    print '-'*50
    print 'Validation set'
    print x_val.shape,y_val.shape
    print '-'*50
    performance_metrics = {}
    for ep in range(1000):
        performance_metrics[ep] = []
        print 'Train Model for epoch: %d'%(ep+1)
        for batch in iterate_minibatches(x_train,y_train,500):
            
            x_batch,y_batch = batch
            x_batch = x_batch / 255.
            y_batch = y_batch.astype('float32')
            
            
            sess.run(optimizer, feed_dict = {
                x: x_batch, y: y_batch})
            
        ctr_val = 0.
        val_acc = 0.
        
        print 'Validating model for epoch: %d'%(ep+1)
        for val_batch in iterate_minibatches(x_val,y_val,500):
            x_val_batch,y_val_batch = val_batch
            
            x_val_batch = x_val_batch / 255.
            y_val_batch = y_val_batch.astype('float32')
            
            temp_accuracy = sess.run(accuracy,
                   feed_dict={
                       x: x_val_batch,
                       y: y_val_batch
                   })
        
            ctr_val+=1
            
            val_acc += temp_accuracy
        
        print 'Final Accuracy = %.6f'%(val_acc / ctr_val)

def convolution_neural_network_model(train_files,val_files,exp_no,num_epochs = 100):
    # we will have 8712 features because 33*33*8
    x = tf.placeholder(tf.float32, [None, 33,33,8])
    # For now binary one hot encoding 
    y = tf.placeholder(tf.float32, [None, 2])
    # Convolutiolal model with 7x7 window size 32 filters and stride 2x2
    filter_size = 7
    n_filters_1 = 32
    input_channels = 8
    # Initialize the weights
    W_conv1 = weight_variable([filter_size, filter_size, input_channels, n_filters_1])
    b_conv1 = bias_variable([n_filters_1])
    # VALID
    # out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    # out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
    
    # SAME
    # out_height = ceil(float(in_height) / float(strides[1]))
    # out_width  = ceil(float(in_width) / float(strides[2]))
    h_conv1 = tf.nn.relu(
    tf.nn.conv2d(input=x,
                 filter=W_conv1,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv1)
    n_filters_2 = 16
    # Initialize the weights
    W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
    b_conv2 = bias_variable([n_filters_2])

    h_conv2 = tf.nn.relu(
    tf.nn.conv2d(input=h_conv1,
                 filter=W_conv2,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv2)


    nn_input = tf.reshape(h_conv2, [-1, 9 * 9 * n_filters_2])
    
    # Build a single layer neural network
    n_fc = 1000
    
    W_fc1 = weight_variable([9 * 9 * n_filters_2, n_fc])
    
    b_fc1 = bias_variable([n_fc])
    
    h_fc1 = tf.nn.relu(tf.matmul(nn_input, W_fc1) + b_fc1)
    
    W_fc2 = weight_variable([n_fc, 2])
    
    b_fc2 = bias_variable([2])
    
    prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
    pred = tf.argmax(prediction, 1)
    truth = tf.argmax(y, 1)
#    correct_prediction = tf.equal(pred, truth)

    cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
    
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    
    correct_prediction = tf.equal(pred, truth)
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
    # Define nowcasting metrics which we will measure after every epoch
    hits = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(truth, pred),tf.equal(truth,1)),'float')) + 1e-16
    misses = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(truth, pred),tf.equal(truth,1)),'float')) + 1e-16
    false_alarms = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(truth, pred),tf.equal(truth,0)),'float'))
    
    POD = tf.div(hits,tf.add(hits,misses))
    FAR = tf.div(false_alarms,tf.add(false_alarms,hits))
    CSI = tf.div(hits,tf.add_n([hits,misses,false_alarms]))
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    performance_metrics = {}
    for ep in range(num_epochs):
        performance_metrics[ep] = []
        print 'Train Model for epoch: %d'%(ep+1)
        print '-'*50
        for tr_file in train_files:
            print tr_file
            if re.findall('\d+',tr_file)[0] == '14':
                base_path = '../data/TrainTest/2014/'
            else:
                base_path = '../data/TrainTest/2015/'
            
            temp_matrix = np.load(base_path + tr_file)
            
            x_train = []
            y_train = []
            for pt in xrange(len(temp_matrix)):         
                x_train.append(temp_matrix[pt][0])
                y_train.append(temp_matrix[pt][1])
            
            x_train = np.vstack(x_train)
            y_train = np.vstack(y_train)

            y_train = y_train.reshape(-1,)
            one_hot = np.zeros((y_train.shape[0], 2),dtype='uint8')
            one_hot[np.arange(y_train.shape[0]),y_train] = 1
            y_train = one_hot
            x_train = x_train.transpose(0,2,3,1)
            print x_train.shape,y_train.shape
            for batch in iterate_minibatches(x_train,y_train,500):
            
                x_batch,y_batch = batch                
                x_batch = x_batch.astype('float32') / 255.          
                sess.run(optimizer, feed_dict = {
                    x: x_batch, y: y_batch.astype('float32')})
            
        
        
        
        print 'Validating model for epoch: %d'%(ep+1)
        print '-'*50
        
        val_acc = 0.
        val_pod = 0.
        val_far = 0.
        val_csi = 0.
        ctr_val = 0.

        for val_file in val_files:
            print val_file
            if re.findall('\d+',val_file)[0] == '14':
                base_path = '../data/TrainTest/2014/'
            else:
                base_path = '../data/TrainTest/2015/'
            
            temp_matrix = np.load(base_path + val_file)
            
            x_val = []
            y_val = []
            for pt in xrange(len(temp_matrix)):         
                x_val.append(temp_matrix[pt][0])
                y_val.append(temp_matrix[pt][1])
            
            x_val = np.vstack(x_val)
            y_val = np.vstack(y_val)

            y_val = y_val.reshape(-1,)
            one_hot = np.zeros((y_val.shape[0], 2),dtype='uint8')
            one_hot[np.arange(y_val.shape[0]),y_val] = 1
            y_val = one_hot
            x_val = x_val.transpose(0,2,3,1)
            print x_val.shape,y_val.shape
            for val_batch in iterate_minibatches(x_val,y_val,500):
                x_val_batch,y_val_batch = val_batch                
                x_val_batch = x_val_batch / 255.
                temp_accuracy = sess.run([accuracy,POD,FAR,CSI],
                   feed_dict={
                       x: x_val_batch,
                       y: y_val_batch.astype('float32')
                       })
                
#                y_hats = sess.run(prediction,
#                                  feed_dict = {
#                                  x : x_val_batch})
#                
#                print y_hats
                
                ctr_val+=1
            
                val_acc += temp_accuracy[0]
                val_pod += temp_accuracy[1]
                val_far += temp_accuracy[2]
                val_csi += temp_accuracy[3]
                
        
        print 'Accuracy for epoch %d = %.6f'%(ep,val_acc / ctr_val)
        print 'Probability of detection for epoch %d = %.6f'%(ep,val_pod / ctr_val)
        print 'False alarm rate for epoch %d = %.6f'%(ep,val_far / ctr_val)
        print 'CSI for epoch %d = %.6f'%(ep,val_csi / ctr_val)
        performance_metrics[ep].append([val_acc / ctr_val,val_pod / ctr_val,val_far / ctr_val,val_csi / ctr_val])
        
    
    saver.save(sess,'../output/model_patameters_' + str(exp_no) + '.ckpt')
    
    f1 = file('../output/performance_metrics_' + str(exp_no) + '.pkl','wb')
    pkl.dump(performance_metrics,f1,protocol = pkl.HIGHEST_PROTOCOL)
    f1.close()
    sess.close()
    # save the model
    


def empty_network(train_files,val_files,exp_no,num_epochs = 100):

    performance_metrics = {}
    for ep in range(num_epochs):
        performance_metrics[ep] = []
        print 'Train Model for epoch: %d'%(ep+1)
        print '-'*50
        for tr_file in train_files:
            print tr_file
            if re.findall('\d+',tr_file)[0] == '14':
                base_path = '../data/TrainTest/2014/'
            else:
                base_path = '../data/TrainTest/2015/'
            
            temp_matrix = np.load(base_path + tr_file)
            
            x_train = []
            y_train = []
            for pt in xrange(len(temp_matrix)):         
                x_train.append(temp_matrix[pt][0])
                y_train.append(temp_matrix[pt][1])
            
            x_train = np.vstack(x_train)
            y_train = np.vstack(y_train)

            y_train = y_train.reshape(-1,)
            one_hot = np.zeros((y_train.shape[0], 2),dtype='uint8')
            one_hot[np.arange(y_train.shape[0]),y_train] = 1
            y_train = one_hot
            x_train = x_train.transpose(0,2,3,1)
            print x_train.shape,y_train.shape
            for batch in iterate_minibatches(x_train,y_train,500):
            
                x_batch,y_batch = batch                
                x_batch = x_batch.astype('float32') / 255.                      
                # train function foes here
                
        print 'Validating model for epoch: %d'%(ep+1)
        print '-'*50
        
        val_acc = 0.
        val_pod = 0.
        val_far = 0.
        val_csi = 0.
        ctr_val = 0.

        for val_file in val_files:
            print val_file
            if re.findall('\d+',val_file)[0] == '14':
                base_path = '../data/TrainTest/2014/'
            else:
                base_path = '../data/TrainTest/2015/'
            
            temp_matrix = np.load(base_path + val_file)
            
            x_val = []
            y_val = []
            for pt in xrange(len(temp_matrix)):         
                x_val.append(temp_matrix[pt][0])
                y_val.append(temp_matrix[pt][1])
            
            x_val = np.vstack(x_val)
            y_val = np.vstack(y_val)

            y_val = y_val.reshape(-1,)
            one_hot = np.zeros((y_val.shape[0], 2),dtype='uint8')
            one_hot[np.arange(y_val.shape[0]),y_val] = 1
            y_val = one_hot
            x_val = x_val.transpose(0,2,3,1)
            print x_val.shape,y_val.shape
            for val_batch in iterate_minibatches(x_val,y_val,500):
                x_val_batch,y_val_batch = val_batch                
                x_val_batch = x_val_batch / 255.
                
                
                ctr_val+=1
            
#                val_acc += temp_accuracy[0]
#                val_pod += temp_accuracy[1]
#                val_far += temp_accuracy[2]
#                val_csi += temp_accuracy[3]
                
        
#        print 'Accuracy for epoch %d = %.6f'%(ep,val_acc / ctr_val)
#        print 'Probability of detection for epoch %d = %.6f'%(ep,val_pod / ctr_val)
#        print 'False alarm rate for epoch %d = %.6f'%(ep,val_far / ctr_val)
#        print 'CSI for epoch %d = %.6f'%(ep,val_csi / ctr_val)
#        performance_metrics[ep].append([val_acc / ctr_val,val_pod / ctr_val,val_far / ctr_val,val_csi / ctr_val])
#        
#        
#    sess.close()
#    f1 = file('../output/performance_metrics_' + str(exp_no) + '.pkl','wb')
#    pkl.dump(performance_metrics,f1,protocol = pkl.HIGHEST_PROTOCOL)
#    f1.close()
#    # save the model
#    saver.save(sess,'../output/model_patameters_' + str(exp_no) + '.ckpt')
   
     

def neural_network_model_new(train_files,val_files,exp_no):
    # we will have 8712 features because 33*33*8
    x = tf.placeholder(tf.float32, [None, 33,33,8])
    # For now binary one hot encoding 
    y = tf.placeholder(tf.float32, [None, 2])
    nn_input = tf.reshape(x, [-1, 33 * 33 * 8])
    
    
    # Build a single layer neural network
    n_fc = 2000
    
    W_fc1 = weight_variable([33 * 33 * 8, n_fc])
    
    b_fc1 = bias_variable([n_fc])
    
    h_fc1 = tf.nn.relu(tf.matmul(nn_input, W_fc1) + b_fc1)
    
    W_fc2 = weight_variable([n_fc, 2])
    
    b_fc2 = bias_variable([2])
    
    prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
    cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
    
    optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
    
    pred = tf.argmax(prediction, 1)
    truth = tf.argmax(y, 1)
    correct_prediction = tf.equal(pred, truth)
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
    # Define nowcasting metrics which we will measure after every epoch
    hits = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(truth, pred),tf.equal(truth,1)),'float'))
    misses = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(truth, pred),tf.equal(truth,1)),'float'))
    false_alarms = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(truth, pred),tf.equal(truth,0)),'float'))
    
    hits = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(truth, pred),tf.equal(truth,1)),'float')) + 1e-16
    misses = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(truth, pred),tf.equal(truth,1)),'float')) + 1e-16
    false_alarms = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(truth, pred),tf.equal(truth,0)),'float'))
    
    POD = tf.div(hits,tf.add(hits,misses))
    FAR = tf.div(false_alarms,tf.add(false_alarms,hits))
    CSI = tf.div(hits,tf.add_n([hits,misses,false_alarms]))
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    performance_metrics = {}
    for ep in range(100):
        performance_metrics[ep] = []
        print 'Train Model for epoch: %d'%(ep+1)
        print '-'*50
        for tr_file in train_files:
#            print tr_file
            if re.findall('\d+',tr_file)[0] == '14':
                base_path = '../data/TrainTest/2014/'
            else:
                base_path = '../data/TrainTest/2015/'
            
            temp_matrix = np.load(base_path + tr_file)
            
            x_train = []
            y_train = []
            for pt in xrange(len(temp_matrix)):         
                x_train.append(temp_matrix[pt][0])
                y_train.append(temp_matrix[pt][1])
            
            x_train = np.vstack(x_train)
            y_train = np.vstack(y_train)

            y_train = y_train.reshape(-1,)
            one_hot = np.zeros((y_train.shape[0], 2),dtype='uint8')
            one_hot[np.arange(y_train.shape[0]),y_train] = 1
            y_train = one_hot
            x_train = x_train.transpose(0,2,3,1)
            x_train.shape,y_train.shape
            for batch in iterate_minibatches(x_train,y_train,500):
            
                x_batch,y_batch = batch                
                x_batch = x_batch.astype('float32') / 255.          
                sess.run(optimizer, feed_dict = {
                    x: x_batch, y: y_batch.astype('float32')})
        
        val_acc = 0.
        val_pod = 0.
        val_far = 0.
        val_csi = 0.
        ctr_val = 0.
        
        for val_file in val_files:
            if re.findall('\d+',val_file)[0] == '14':
                base_path = '../data/TrainTest/2014/'
            else:
                base_path = '../data/TrainTest/2015/'
            
            temp_matrix = np.load(base_path + val_file)
            
            x_val = []
            y_val = []
            for pt in xrange(len(temp_matrix)):         
                x_val.append(temp_matrix[pt][0])
                y_val.append(temp_matrix[pt][1])
            
            x_val = np.vstack(x_val)
            y_val = np.vstack(y_val)

            y_val = y_val.reshape(-1,)
            one_hot = np.zeros((y_val.shape[0], 2),dtype='uint8')
            one_hot[np.arange(y_val.shape[0]),y_val] = 1
            y_val = one_hot
            x_val = x_val.transpose(0,2,3,1)
            print x_val.shape,y_val.shape
            print '-'*50
            for val_batch in iterate_minibatches(x_val,y_val,500):
                x_val_batch,y_val_batch = val_batch                
                x_val_batch = x_val_batch / 255.
                temp_accuracy = sess.run([accuracy,POD,FAR,CSI],
                   feed_dict={
                       x: x_val_batch,
                       y: y_val_batch.astype('float32')
                       })
            
                ctr_val+=1
            
                val_acc += temp_accuracy[0]
                val_pod += temp_accuracy[1]
                val_far += temp_accuracy[2]
                val_csi += temp_accuracy[3]
                
        
        print 'Accuracy for epoch %d = %.6f'%(ep,val_acc / ctr_val)
        print 'Probability of detection for epoch %d = %.6f'%(ep,val_pod / ctr_val)
        print 'False alarm rate for epoch %d = %.6f'%(ep,val_far / ctr_val)
        print 'CSI for epoch %d = %.6f'%(ep,val_csi / ctr_val)
        performance_metrics[ep].append([val_acc / ctr_val,val_pod / ctr_val,val_far / ctr_val,val_csi / ctr_val])

    f1 = file('../output/performance_metrics_' + str(exp_no) + '.pkl','wb')
    pkl.dump(performance_metrics,f1,protocol = pkl.HIGHEST_PROTOCOL)
    f1.close()
    sess.close()

def train_neural_net(train_files,validation_files):
    '''We do a total of 7 runs, train on 6 months and test on the 7th
    month thus loop thru the 7 teain,validation file blocks'''
    yr_mon = {14: [5,6,7,8],
              15: [5,6,7,8]}
    blocks = [str(yr) + '_' + str(mon) for yr in yr_mon.keys() for mon in yr_mon[yr]]
    ctr = 0
    for tr,val in zip(train_files,validation_files):
        x_train = []
        y_train = []
        # Loop thru all training files and store into an array with
        # 8 bit data type
        print '-'*50
        print 'Validation year = %s Month = %s'%tuple(blocks[ctr].split('_'))
        print 'Building training set...'
        for t in tr:
            if re.findall('\d+',t)[0] == '14':
                base_path = '../data/TrainTest/2014/'
            else:
                base_path = '../data/TrainTest/2015/'
            # we are going to load 500 points from a string of days
            # at each call of this load function
            temp_matrix = np.load(base_path + t)
            for pt in xrange(len(temp_matrix)):         
                x_train.append(temp_matrix[pt][0])
                y_train.append(temp_matrix[pt][1])
        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train)
        y_train = y_train.reshape(-1,)
        # One hot encode the output variable which is the preffered 
        # representation for tensor flow
        one_hot = np.zeros((y_train.shape[0], 2),dtype='uint8')
        one_hot[np.arange(y_train.shape[0]),y_train] = 1
        print 'Shape of training set'
        print x_train.shape
        x_train = x_train.transpose(0,2,3,1)
        y_train = one_hot
        print y_train.shape
        # Loop thru validation files and store it into an array
        print '-'*50
        print 'Building validation set...'
        x_val = []
        y_val = []
        for v in val:
            if re.findall('\d+',v)[0] == '14':
                base_path = '../data/TrainTest/2014/'
            else:
                base_path = '../data/TrainTest/2015/'
            temp_matrix = np.load(base_path + v)
            for pt in xrange(len(temp_matrix)):
                x_val.append(temp_matrix[pt][0])
                y_val.append(temp_matrix[pt][1])
        x_val = np.vstack(x_val)
        y_val = np.vstack(y_val)
        y_val = y_val.reshape(-1,)
        one_hot = np.zeros((y_val.shape[0],2),dtype='uint8')
        one_hot[np.arange(y_val.shape[0]),y_val] = 1
        y_val = one_hot
        print 'Shape of validation set'
        print x_val.shape
        print y_val.shape
        x_val = x_val.transpose(0,2,3,1)
        
        neural_network_model(x_train,y_train,x_val,y_val)
        ctr+=1
        
def main(convert_to_images = False):
    data_builder = BuildDataSet.dataset(num_points = 500)
    training_blocks,validation_blocks = build_training_validation_sets(data_builder)
#    make_dataset_NN_2(data_builder)
    if convert_to_images:
#        convert_fields_to_images(data_builder)
        make_dataset_NN(data_builder)
    train,validation = arrange_training_validation(training_blocks,validation_blocks)
    
    exp_no = 0
    for tr,val in zip(train,validation):
        print 'Train Test Split %d '%(exp_no + 1)
        neural_network_model_new(tr,val,exp_no+1)
#    convolution_neural_network_model(train[0],validation[0], exp_no + 1,100)
#        empty_network(tr,val,exp_no+1,1)
        exp_no+=1
#    train_neural_net(train,validation)
    
if __name__ == '__main__':
    main()