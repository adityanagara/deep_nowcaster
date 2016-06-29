# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 14:52:25 2016

@author: adityanagarajan
"""

import os
import BuildDataSet
import DCNN_network
import numpy as np
from theano import tensor as T
import lasagne
import cPickle
import theano
import sys

# Load the parameters of a trained neural network
file_name = 'CPU_network_1000_1200_gpu3_1000.pkl' #sys.argv[1]
network_file = open('output/' + file_name,'rb')
params = cPickle.load(network_file)
network_file.close()

# Initialize a 4D theano tensor variable
input_var = T.tensor4('inputs')

# Load the deep neural network module
neural_nets = DCNN_network.DCNN_network()

# Build the deep neural net (see the package in includes/)
network = neural_nets.build_CNN_3_softmax(input_var)
lasagne.layers.set_all_param_values(network, params)

# Build theano function which inturn generates and compiles C code to run the network
prediction_ = lasagne.layers.get_output(network, deterministic=True)

predict_function = theano.function([input_var], prediction_)

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

    





