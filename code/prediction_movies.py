# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:33:27 2016

@author: adityanagarajan
"""

import os
import cPickle as pkl
from matplotlib import pyplot as plt
from theano import tensor as T
import numpy as np
import lasagne
import theano
import BuildDataSet
import re
plt.ioff()


def build_DCNN_maxpool_softmax(input_var = None):
    from lasagne.layers import Conv2DLayer, MaxPool2DLayer
    
    print('Training the maxpool network!!')
    # Define the input variable which is 4 frames of IPW fields and 4 frames of 
    # reflectivity fields
    l_in = lasagne.layers.InputLayer(shape=(None,8,33,33),
                                        input_var=input_var)
    
    l_conv1 = Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(5, 5),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            pad = 'full'
        )
        
    l_maxpool = MaxPool2DLayer(l_conv1,(2,2))
    
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


def make_predictions(yr = 14):
    base_path = '../output/1_CNN_experiments/'
    # /Users/adityanagarajan/projects/nowcaster/output/1_CNN_experiments
    print os.path.exists(base_path + 'CPU_1_CNN_layer_max_pool_0_200.pkl')
    params = np.load(base_path + 'CPU_1_CNN_layer_max_pool_0_200.pkl')
    print len(params)
    
    input_var = T.tensor4('inputs')
    
    network,hidden_1 = build_DCNN_maxpool_softmax(input_var)
    
    lasagne.layers.set_all_param_values(network, params)
    
    prediction_ = lasagne.layers.get_output(network, deterministic=True)

    predict_function = theano.function([input_var], prediction_)
    
    data_builder = BuildDataSet.dataset(num_points = 1000)
    
    # This is the range of points in the central domain which can 
    # have a 33x33 around it.
    num_points = 4356
    
    domain_points = (range(17,83),range(17,83))
    
    PixelPoints = [(x,y) for x in domain_points[0]  
                    for y in domain_points[1]]
                        
    
    PixelPoints = np.array(PixelPoints)
    
    storm_dates_all = data_builder.load_storm_days(yr)
    
    doy_strings = data_builder.club_days(storm_dates_all)
    
    days_in_sorted = doy_strings.keys()
    
    days_in_sorted.sort()
    
    print doy_strings
    
    ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(yr)
    
    temp_ipw_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings['129'],ipw_files)
#    
    temp_refl_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings['129'],refl_files)
    
    temp_ipw_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_ipw_files)
    temp_refl_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_refl_files)       
    # We define an array which contains the ground truth in column 1
    # and the prediction probabilities in column 2 and 3
    real_predictions = np.zeros((num_points,91,3))
    
    pt_ctr = 0
    
    for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
        
        print 'Predicting for point: (%d,%d)'%(x_,y_)
        
        temp_array = data_builder.build_features_and_truth_imgs(temp_ipw_files,temp_refl_files,x_,y_)
        
        ipw_refl_tensors = data_builder.arrange_frames_single(temp_array)
        
        X_test = ipw_refl_tensors[1]
        
        Y_test = ipw_refl_tensors[2]
        
        Y_pred = predict_function(X_test)
        
        real_predictions[pt_ctr,:,0] = Y_test.reshape(91,)
        
#        print Y_test.reshape(91,)
#        print np.argmax(Y_pred,axis=1)
        
        real_predictions[pt_ctr,:,1:] = Y_pred
        
        real_predictions.shape
        
        pt_ctr+=1
        
    return real_predictions
    
def plot_movies():
    real_pred = np.load('../output/CNN_real_pred_array.npy')
    real_pred =  real_pred.reshape(66,66,91,3)
    
    gridX = np.arange(-150.0,151.0,300.0/(100-1))
    gridY = np.arange(-150.0,151.0,300.0/(100-1))
    
    # center of the domain
    x_ = 49.0
    y_ = 49.0

    x_range_start = gridX[x_] - 33.0*(300.0/99.0)
    y_range_start = gridY[y_] - 33.0*(300.0/99.0)

    x_range_end = gridX[x_] + 33.0*(300.0/99.0)
    y_range_end = gridY[y_] + 33.0*(300.0/99.0)

    gridX_ = np.arange(x_range_start,x_range_end,300./99.)
    gridY_ = np.arange(y_range_start,y_range_end,300./99.)
    
#    i_start = x_ -33
#    i_end = x_ + 33
#    j_start = y_ - 33
#    j_end = y_ + 33
#    
#    time_index = ['{0}{1}'.format(str(x).zfill(2),str(y).zfill(2)) 
#                        for x in range(24) for y in [0,30]]
    num_time_steps = 91
    
    
    for i in range(num_time_steps):
        plt.figure()
        
        pred_ = np.argmax(real_pred[:,:,i,1:],axis=2)
        
        pred_ = np.ma.array(pred_, mask = pred_ == 0)
        
        real_ = np.ma.array(real_pred[:,:,i,0], mask = real_pred[:,:,i,0] == 0.)
        plt.subplot(121)
        plt.pcolor(gridX_,gridY_,real_.T,cmap='RdGy',vmin=0.0,vmax=1.0)
        
        plt.xlim((-150.0,150.0))
        
        plt.ylim((-150.0,150.0))
        
        plt.grid()
        
        plt.subplot(122)
        plt.pcolor(gridX_,gridY_,pred_.T,cmap='RdGy',vmin=0.0,vmax=1.0)
        
        plt.xlim((-150.0,150.0))
        
        plt.ylim((-150.0,150.0))
        
        plt.grid()
        
        plt.savefig('../output/prediction_movies_thesis/Plot_' + str(i) + '.png')
        

if __name__ == '__main__':
#    real_pred = make_predictions()
#    print real_pred.shape
#    np.save('../output/CNN_real_pred_array.npy',real_pred)
    plot_movies()
    
'''
for x_ in range(num_time_steps):
#    plt.figure()
    plt.subplots(1,4)
    plt.subplot(141)
    real_ = np.ma.array(movie_points_1[:,:,x_,0], mask=movie_points_1[:,:,x_,0] == 0.)
    plt.pcolor(gridX_,gridY_,real_.T,cmap='RdGy', vmin=0.0, vmax=1.0)
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))
    plt.grid()
    plt.title('Actual')
    plt.subplot(142)
    predicted_1 = np.ma.array(movie_points_1[:,:,x_,1], mask=movie_points_1[:,:,x_,1] == 0.)
    plt.pcolor(gridX_,gridY_,predicted_1.T,cmap='RdGy', vmin=0.0, vmax=1.0)
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))
    plt.grid()
    plt.title('IPW')
    plt.subplot(143)
    predicted_2 = np.ma.array(movie_points_3[:,:,x_,1], mask=movie_points_3[:,:,x_,1] == 0.)
    plt.pcolor(gridX_,gridY_,predicted_2.T,cmap='RdGy', vmin=0.0, vmax=1.0)
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))
    plt.grid()
    plt.title('Reflectivity')
    plt.subplot(144)
    predicted_3 = np.ma.array(movie_points_2[:,:,x_,1], mask=movie_points_2[:,:,x_,1] == 0.)
    plt.pcolor(gridX_,gridY_,predicted_3.T,cmap='RdGy', vmin=0.0, vmax=1.0)
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))
    plt.grid()
    plt.title('Reflectivity + IPW')
    plt.suptitle('Actual and predicted fields storm May 8-10 2015 ') #+ time_index[x_ - 43] + ' UTC')    
    plt.savefig('output/prediction_movies_vaisala/RF/Plot_' + str(x_) + '.png')
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

# Load the data builder object
data_builder = BuildDataSet.dataset(Threshold = None)

# This is the range of points in the central domain which can have a 33x33 around it.
num_points = 4356
domain_points = (range(17,83),range(17,83))

# Arrange all pixel points to a list
PixelPoints = [(x,y) for x in domain_points[0] for y in domain_points[1]]

# Call the function to club consecutive days together
sorted_days = data_builder.club_days()

# Get the end of each consecutive string of days
days_in_sorted = sorted_days.keys()

# Sort the days
days_in_sorted.sort()

PixelPoints = np.array(PixelPoints)

# Get all the IPW and reflectivity files for the clubbed days
temp_ipw_file_list = filter(lambda x: x[7:10] in sorted_days['129'],data_builder.IPWfiles)
temp_radar_file_list = filter(lambda x: x[9:12] in sorted_days['129'],data_builder.Radarfiles)

#PixelPoints = PixelPoints[:10]
print PixelPoints.shape

real_predictions = np.zeros((num_points,91,2))

pt_ctr = 0

for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
    print 'Building data set for point: (%d,%d)'%(x_,y_)
    test_set = data_builder.build_features_and_truth(temp_ipw_file_list,temp_radar_file_list,x_,y_)
    
    X_test,Y_test = data_builder.arrange_frames_single(test_set) 
    Y_pred = predict_function(X_test)
    real_predictions[pt_ctr,:,0] = Y_test.reshape(91,)
    real_predictions[pt_ctr,:,1] = Y_pred[0].reshape(91,)
    pt_ctr+=1
#np.save('test_predictions.npy',real_predictions)

np.argmax()
'''

