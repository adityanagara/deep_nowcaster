# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:55:01 2015

@author: adityanagarajan
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import cPickle


# Define the full path of the data set
raw_data_base_dir = '/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/data/TrainTest/'


#movie_points = np.load('data/NB_real_predictions.npy')
#file_name = file('output/output_array.pkl','rb')
#movie_points = cPickle.load(file_name)
movie_points = np.load('output/real_prediction_array_2015.npy')

#file_name.close()

#for m in range(len(movie_points)):
#    movie_points[m,...] = movie_points[m,...].reshape(4356,91,2)
#    movie_points[m,...] = movie_points[m,...].reshape(66,66,91,2)

num_time_steps = 139
movie_points_1 = movie_points[0,...].reshape(4356,num_time_steps,2)
movie_points_2 = movie_points[1,...].reshape(4356,num_time_steps,2)
movie_points_3 = movie_points[2,...].reshape(4356,num_time_steps,2)

movie_points_1 = movie_points[0,...].reshape(66,66,num_time_steps,2)
movie_points_2 = movie_points[1,...].reshape(66,66,num_time_steps,2)
movie_points_3 = movie_points[2,...].reshape(66,66,num_time_steps,2)

gridX = np.arange(-150.0,151.0,300.0/(100-1))
gridY = np.arange(-150.0,151.0,300.0/(100-1))

out_point = np.zeros((100,100))


x_ = 49.0
y_ = 49.0

x_range_start = gridX[x_] - 33.0*(300.0/99.0)
y_range_start = gridY[y_] - 33.0*(300.0/99.0)

x_range_end = gridX[x_] + 33.0*(300.0/99.0)
y_range_end = gridY[y_] + 33.0*(300.0/99.0)

gridX_ = np.arange(x_range_start,x_range_end,300./99.)
gridY_ = np.arange(y_range_start,y_range_end,300./99.)

plt.figure()

i_start = x_ -33
i_end = x_ + 33
j_start = y_ - 33
j_end = y_ + 33

time_index = ['{0}{1}'.format(str(x).zfill(2),str(y).zfill(2)) for x in range(24) for y in [0,30]]

# file_list = ['RF_ipw_random_points_avg.pkl','RF_ipw_refl_random_points_avg.pkl','RF_refl_random_points_avg.pkl']

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