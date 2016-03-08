# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:19:51 2016

@author: adityanagarajan
This file visually verifies the slices
"""

import BuildDataSet
import cPickle
import os
import sys
import numpy as np

data_builder = BuildDataSet.dataset(Threshold = 'binary')
PixelPoints = data_builder.sample_random_pixels()

#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

def build_points_fields():
    num_points = 80
    prev_pt_tr = 0
    for tr_pt in range(0,num_points,40):
        print 'Building set %d '%tr_pt
        train_set = data_builder.make_points_frames(PixelPoints[prev_pt_tr:tr_pt + 40])
        print 'Done making training set'
        X_Y_train = data_builder.arrange_frames(train_set)
        temp_file = open('data/dataset2/points_' + str(prev_pt_tr) + '_'+ str(tr_pt + 40) + '.pkl','wb')
        cPickle.dump((X_Y_train,PixelPoints[prev_pt_tr:tr_pt + 40]),temp_file,protocol=cPickle.HIGHEST_PROTOCOL)
        temp_file.close()
#        training_points.append(X_Y_train,PixelPoints[10:20])
        prev_pt_tr += 40

def load_points_fields():
    base_path = 'data/dataset2/'
    file_list = os.listdir(base_path)
    file_list = filter(lambda x: x[-4:] == '.pkl',file_list)
    massive_data = []
    print(file_list)
    for f in file_list:
        print f
        open_file = open(base_path + f,'rb')
        load_file = cPickle.load(open_file)
        open_file.close()
        massive_data.append(load_file)
#        print('Done loading file %s '%f)
    for data in massive_data:
        print(data[0].shape,data[1].shape)
    print('Successfully Loaded 50gigs of data!! on da cluster')

temp_file = file('data/dataset2/points_40_80.pkl')
arr = cPickle.load(temp_file)
temp_file.close()

import nowcast
nowcaster = nowcast.BuildNowcaster()

# This function is for verificarion of the data set we created where we sliced the frames
# It then arranges files as a list where each element has 4 files in it.
def arrange_files():
#    doy_didata_builder.sorted_days
    doys = data_builder.days_in_sorted
    doys.sort()
    # This list needs to be of size 1059 where each entry is a set of 4 files
    final_file_list_ipw = []
    final_file_list_refl = []
    for d in doys:
        temp_ipw_file_list = filter(lambda x: x[9:12] in data_builder.sorted_days[d],data_builder.IPWfiles)
        temp_radar_file_list = filter(lambda x: x[11:14] in data_builder.sorted_days[d],data_builder.Radarfiles)
        for i in range(len(temp_ipw_file_list)):
            if i > 4:
                final_file_list_ipw.append(temp_ipw_file_list[i-5:i-1])
                final_file_list_refl.append(temp_radar_file_list[i-5:i-1])
    return final_file_list_ipw,final_file_list_refl
    
def plot_original_fields(grid_point,time_step):
    base_path = 'data/TrainTest/'
    ipw,refl = arrange_files()
    temp_file_list_ipws = ipw[time_step]
    temp_file_list_radars = refl[time_step]
    for i,r in zip(temp_file_list_ipws,temp_file_list_radars):
        print i,r
        nowcaster.plot_fields_side(base_path + i,base_path + r,grid_point)

def map_back(ipws,refls):
    ipws_back = np.zeros((4,33,33))
    refls_back = np.zeros((4,33,33))
    ipw_range = np.linspace(-5,5,256)
    refl_range = np.linspace(0,90,256)
    for i in range(ipws.shape[0]):
        for j in range(ipws.shape[1]):
            for k in range(ipws.shape[2]):
                ipws_back[i,j,k] = ipw_range[ipws[i,j,k]]
                refls_back[i,j,k] = refl_range[refls[i,j,k]]
    return ipws_back,refls_back

# Determine the point in the index you want to verify
point = 20
#Find out what to slice based on the point you just picked
start_block = (point + 1)*1059 - 1059
end_block = 1059 + point*1059
point_block = slice(start_block,end_block)
# Make the array with respect to that point only with all the files in the training set
point_1 = arr[0][0][point_block]
# Define a time step which you want to verify
time_step = 24
ipws = point_1[time_step,:4,...]
refls = point_1[time_step,4:,...]
# map the images back to fields for verification purposes
ipw_field,refl_field = map_back(ipws,refls)
# assign the grid point that we are verifying to a variable
grid_point = arr[1][point]
# Plot the 6 ipw and reflectivity fields
plot_original_fields(grid_point,time_step)
# plot the sliced fields
#refl_field[refl_field < 20.0] = 0.0
refl_field = np.ma.array(refl_field,mask = refl_field == 0.)
nowcaster.plot_field_slices(ipws,refls,ipw_field,refl_field,grid_point)


#build_points_fields()
#temp_file = open('data/dataset2/visua_verification_img.pkl','wb')
#cPickle.dump(visual_verification,temp_file,protocol=cPickle.HIGHEST_PROTOCOL)
#temp_file.close()
#        temp_file = open('data/dataset2/points_' + str(prev_pt_tr) + '_'+ str(tr_pt + 40) + '.pkl','wb')
#        cPickle.dump(X_Y_train,temp_file,protocol=cPickle.HIGHEST_PROTOCOL)
#        temp_file.close()
#
#load_points_fields()

    
    