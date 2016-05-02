# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:35:54 2015

@author: adityanagarajan

Notes: This file pulls data from IPW and reflectivity fields cropes 33x33 and maps 
to central point and then makes predictions using the averages
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import random
import cPickle

#import BuildDataSet

class dataset(object):
    """This class builds the data set given the pixel points
    """
    def __init__(self,Thrashold = 24.0):
        self.TrainTestdir = '/Users/adityanagarajan/projects/nowcaster/data/TrainTest/'
        self.IPWfiles, self.Radarfiles = self.sort_IPW_refl_files()
        self.Thrashold = Thrashold
        
    def sort_IPW_refl_files(self):
        
        files = os.listdir(self.TrainTestdir)

        IPWfiles = filter(lambda x: x[:7] == 'IPWdata' ,files)

        Radarfiles = filter(lambda x: x[:9] == 'RadarRefl',files)

        #Sort files DOY.hh
        IPWfiles.sort(key = lambda x: float(x[9:12]) + float(x[x.index('_') + 1: x.index('.')]) * 0.01)

        
        Radarfiles.sort(key = lambda x: float(x[11:14]) + float(x[x.index('_') + 1: x.index('.')])* 0.01)
        
        # Pull out june data for now
#        IPWfiles = filter(lambda x: int(x[7:10]) < 152 or int(x[7:10]) > 181,IPWfiles)
#        
#        Radarfiles = filter(lambda x: int(x[9:12]) < 152 or int(x[9:12]) > 181,Radarfiles)
        
        # Pull out 205 = 07/24
#        IPWfiles = filter(lambda x: int(x[7:10]) != 205, IPWfiles)
#        
#        Radarfiles = filter(lambda x: int(x[9:12]) != 205,Radarfiles)
                
        return IPWfiles,Radarfiles

    ## Define list of days in this experiment
    def doy_list(self):
        temp_list = []
        for f in self.IPWfiles:
            if f[9:12] not in temp_list:
                temp_list.append(f[9:12])
        return temp_list
    
    def club_days(self):
        temp_list = self.doy_list()
        club_list = {}
        l = 0
        while l < len(temp_list):
            temp_num = int(temp_list[l])
            ctr2 = 1
            while ctr2 < 10:
                if str(temp_num +ctr2) in temp_list:
                    ctr2+=1
                    l+=1
                else:
                    club_list[temp_list[l]] = [str(x) for x in reversed(range(int(temp_list[l]),int(temp_list[l]) - ctr2,-1))]
                    l+=1
                    break
        return club_list
    
    def build_features_and_truth(self,temp_ipw_file_list,temp_radar_file_list,x_,y_):
        '''The structure of the out array is as follows:
        return tuple(ipw_array,reflectivity_array)
        ipw_array -> shape = n_examples x 6534 (6 x 1089(33x33 stretched vector) + 1)
        t,t - 30,t - 60,t - 90,t - 120,t - 150, rainfall_flag
        reflectivity_array -> shape = n_examples x (6 x 1089)
        t, t - 30, t - 60,t - 90,t - 120, t - 150
        '''
        Thrashold = self.Thrashold
        # Initialize array (4536 for 4 time steps of IPW fields, and one for ground truth)
        out_matrixIPW = np.zeros((len(temp_ipw_file_list),1089*6 + 1))
        out_matrixRadar = np.zeros((len(temp_ipw_file_list),1089*6))
        i_start = x_ -16
        i_end = x_ + 17
        j_start = y_ -16
        j_end = y_ + 17

        out_matrixIPW[:] = np.nan
        matrix_ctr = 0
    
        for i_file,r_file in zip(temp_ipw_file_list,temp_radar_file_list):
            RadarMatrix = np.load(self.TrainTestdir + r_file)
    
            RadarMatrix[np.isnan(RadarMatrix)] = 0.0
            RadarMatrixFeature = RadarMatrix.copy()
    
            RadarMatrix[RadarMatrix < Thrashold] = 0.0
    
            RadarMatrix[RadarMatrix >= Thrashold] = 1.0
        
            pointXY = RadarMatrix[y_,x_]
            out_matrixIPW[matrix_ctr,-1] = pointXY
            # Current field time stem in is the first 1089 features
            IPWMatrix = np.load(self.TrainTestdir + i_file)
            out_matrixIPW[matrix_ctr,:1089] = IPWMatrix[j_start:j_end,i_start:i_end].reshape(-1,)
            out_matrixRadar[matrix_ctr,:1089] = RadarMatrixFeature[j_start:j_end,i_start:i_end].reshape(-1,)
            # Start with the first point 
            if temp_ipw_file_list.index(i_file) == 0:
                pass
            # If second point then we can accomodate fields from 2 time steps (current and previous)
            elif temp_ipw_file_list.index(i_file) == 1:
                out_matrixIPW[matrix_ctr,1089:2178] = out_matrixIPW[matrix_ctr -1,:1089]
                out_matrixRadar[matrix_ctr,1089:2178] = out_matrixRadar[matrix_ctr - 1,:1089]
            # Third time step 3 fields
            elif temp_ipw_file_list.index(i_file) == 2:
                out_matrixIPW[matrix_ctr,1089:2178] = out_matrixIPW[matrix_ctr -1,:1089]
                out_matrixIPW[matrix_ctr,2178:3267] = out_matrixIPW[matrix_ctr -2,:1089]
                # dbz fields
                out_matrixRadar[matrix_ctr,1089:2178] = out_matrixRadar[matrix_ctr - 1,:1089]
                out_matrixRadar[matrix_ctr,2178:3267] = out_matrixRadar[matrix_ctr - 2,:1089]
            # Fourth time step 4 fields including current one
            elif temp_ipw_file_list.index(i_file) == 3:
                out_matrixIPW[matrix_ctr,1089:2178] = out_matrixIPW[matrix_ctr -1,:1089]
                out_matrixIPW[matrix_ctr,2178:3267] = out_matrixIPW[matrix_ctr -2,:1089]
                out_matrixIPW[matrix_ctr,3267:4356] = out_matrixIPW[matrix_ctr -3,:1089]
                # dbz fields
                out_matrixRadar[matrix_ctr,1089:2178] = out_matrixRadar[matrix_ctr - 1,:1089]
                out_matrixRadar[matrix_ctr,2178:3267] = out_matrixRadar[matrix_ctr - 2,:1089]
                out_matrixRadar[matrix_ctr,3267:4356] = out_matrixRadar[matrix_ctr - 3,:1089]
            # Fifth time step 5 fields including current one (Most of the time we will go in here)
            elif temp_ipw_file_list.index(i_file) == 4:
                # Fifth time step 
                out_matrixIPW[matrix_ctr,1089:2178] = out_matrixIPW[matrix_ctr -1,:1089]
                out_matrixIPW[matrix_ctr,2178:3267] = out_matrixIPW[matrix_ctr -2,:1089]
                out_matrixIPW[matrix_ctr,3267:4356] = out_matrixIPW[matrix_ctr -3,:1089]
                out_matrixIPW[matrix_ctr,4356:5445] = out_matrixIPW[matrix_ctr -4,:1089]
                # dbz firlds
                out_matrixRadar[matrix_ctr,1089:2178] = out_matrixRadar[matrix_ctr - 1,:1089]
                out_matrixRadar[matrix_ctr,2178:3267] = out_matrixRadar[matrix_ctr - 2,:1089]
                out_matrixRadar[matrix_ctr,3267:4356] = out_matrixRadar[matrix_ctr - 3,:1089]
                out_matrixRadar[matrix_ctr,4356:5445] = out_matrixRadar[matrix_ctr - 4,:1089]
            else:
                # Fully loaded vector
                out_matrixIPW[matrix_ctr,1089:2178] = out_matrixIPW[matrix_ctr -1,:1089]
                out_matrixIPW[matrix_ctr,2178:3267] = out_matrixIPW[matrix_ctr -2,:1089]
                out_matrixIPW[matrix_ctr,3267:4356] = out_matrixIPW[matrix_ctr -3,:1089]
                out_matrixIPW[matrix_ctr,4356:5445] = out_matrixIPW[matrix_ctr -4,:1089]
                out_matrixIPW[matrix_ctr,5445:6534] = out_matrixIPW[matrix_ctr -5,:1089]
                # dbz firlds
                out_matrixRadar[matrix_ctr,1089:2178] = out_matrixRadar[matrix_ctr - 1,:1089]
                out_matrixRadar[matrix_ctr,2178:3267] = out_matrixRadar[matrix_ctr - 2,:1089]
                out_matrixRadar[matrix_ctr,3267:4356] = out_matrixRadar[matrix_ctr - 3,:1089]
                out_matrixRadar[matrix_ctr,4356:5445] = out_matrixRadar[matrix_ctr - 4,:1089]
                out_matrixRadar[matrix_ctr,5445:6534] = out_matrixRadar[matrix_ctr - 5,:1089]

            # increment counter
            matrix_ctr+=1
        return out_matrixIPW,out_matrixRadar
        
    def squish_to_averages(self,random_points):
        '''This function takes IPW and refl. foelds at each time step
        to generate average fields at (33x33) --> 1'''
        indices_i = [(x*1089,(x+1)*1089)for x in range(4) ]
        indices_r = [(x*1089,(x+1)*1089)for x in range(4,8) ]
            
#        random_points = (ipw_matrix,refl_matrix)
            
#        IPWFeatures = np.concatenate(map(lambda x: x[0].astype('float32'),random_points))
            
#        ReflFeatures = np.concatenate(map(lambda x: x[1].astype('float32'),random_points))
        
        IPWFeatures = random_points[0].astype('float32')
        
        ReflFeatures = random_points[1].astype('float32')
            
        # 2178 -> use the fields from a whole time step back
        data = np.hstack((IPWFeatures[:,2178:-1],ReflFeatures[:,2178:],IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)))

        data = data[~np.any(np.isnan(data),axis = 1),:]

        ipw_avg_fields = np.zeros((data.shape[0],9))


        ipw_avg_fields[:,-1] = data[:,-1]
            
#        average_fields = []
            
        for ix in range(len(indices_i)):
            ipw_avg_fields[:,ix] = np.average(data[:,indices_i[ix][0]:indices_i[ix][1]],axis = 1)
    
            ipw_avg_fields[:,ix + 4] = np.average(data[:,indices_r[ix][0]:indices_r[ix][1]],axis = 1)

#        average_fields.append(ipw_avg_fields)
        
        return ipw_avg_fields

data_builder = dataset()

# This is the range of points in the central domain which can have a 33x33
# around it.
domain_points = (range(17,83),range(17,83))

PixelPoints = [(x,y) for x in domain_points[0] for y in domain_points[1]]

sorted_days = data_builder.club_days()

days_in_sorted = sorted_days.keys()

days_in_sorted.sort()

print days_in_sorted

PixelPoints = np.array(PixelPoints)

temp_ipw_file_list = filter(lambda x: x[9:12] in sorted_days['130'],data_builder.IPWfiles)

temp_radar_file_list = filter(lambda x: x[11:14] in sorted_days['130'],data_builder.Radarfiles)

points_data = []

# 4356,43,9; points = 66 x 66

num_points = 4356

X_test_set = np.zeros((num_points,139,9),dtype='float32')
ctr = 0
for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):

    temp_array_tuple = data_builder.build_features_and_truth(temp_ipw_file_list,temp_radar_file_list,x_,y_)
    
    X_test_points = data_builder.squish_to_averages(temp_array_tuple)
    
    X_test_set[ctr, ...] = X_test_points
    ctr +=1


X_test_set = X_test_set.reshape(num_points*139,9).astype('float64')

real_predictions = np.zeros((3,num_points*139,2))
'''
RF_ipw_random_points_avg.pkl
RF_ipw_refl_random_points_avg.pkl
RF_refl_random_points_avg.pkl
'''
file_list = ['RF_ipw_random_points_avg.pkl','RF_ipw_refl_random_points_avg.pkl','RF_refl_random_points_avg.pkl']
file_slices = [slice(4),slice(8),slice(4,8)]
real_predictions[0,:,0] = X_test_set[:,-1]

file_path = '/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/model_metrics/'

predictions_ = []

print 'Starting predictions'

ctr = 0
for f1,s1 in zip(file_list,file_slices):
    f = file(file_path + f1)
    RF_models = cPickle.load(f)
    f.close()
    real_predictions[ctr,:,1] = RF_models[0][-1].predict(X_test_set[:,s1])
    ctr+=1

np.save('output/real_prediction_array_2015.npy',real_predictions)

#save_predictions = file('output/output_array.pkl','wb')
#cPickle.dump(predictions_,save_predictions,protocol = cPickle.HIGHEST_PROTOCOL)
#save_predictions.close()
#np.save('data/NB_real_predictions.npy',real_predictions)

print 'Done!'
 


    
    
    

  