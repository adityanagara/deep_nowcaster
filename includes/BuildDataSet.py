# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:23:41 2015

@author: adityanagarajan
"""

import os
import numpy as np
import random

Basefile = '/home/an67a/deep_nowcaster/data/dataset/'
'''
This package deals with building data sets for training a machine learning
based nowcasting system. This means we can build 3 frames of IPW and 4 frames
of reflectivity for each sampled point. 
'''

class dataset(object):
    """This class builds the data set given the pixel points
    """
    def __init__(self,Threshold = 24.0,num_points = 1500):
        self.TrainTestdir = '/home/an67a/deep_nowcaster/data/dataset/'
        self.IPWfiles, self.Radarfiles = self._sort_IPW_refl_files()
        self.Threshold = Threshold
        self.num_points = num_points 
        self.sorted_days = self.club_days()
        self.days_in_sorted = self.sorted_days.keys()
        
    def _sort_IPW_refl_files(self):
        
        files = os.listdir(self.TrainTestdir)

        IPWfiles = filter(lambda x: x[:7] == 'IPWdata' ,files)

        Radarfiles = filter(lambda x: x[:9] == 'RadarRefl',files)

        #Sort files DOY.hh
        IPWfiles.sort(key = lambda x: float(x[7:10]) + float(x[x.index('_') + 1: x.index('.')]) * 0.01)

        
        Radarfiles.sort(key = lambda x: float(x[9:12]) + float(x[x.index('_') + 1: x.index('.')])* 0.01)
        
        # Pull out june data for now
        IPWfiles = filter(lambda x: int(x[7:10]) < 152 or int(x[7:10]) > 181,IPWfiles)
        
        Radarfiles = filter(lambda x: int(x[9:12]) < 152 or int(x[9:12]) > 181,Radarfiles)
        
        # Pull out 205 = 07/24
        IPWfiles = filter(lambda x: int(x[7:10]) != 205, IPWfiles)
        
        Radarfiles = filter(lambda x: int(x[9:12]) != 205,Radarfiles)
                
        return IPWfiles,Radarfiles

    ## Define list of days in this experiment
    def doy_list(self):
        temp_list = []
        for f in self.IPWfiles:
            if f[7:10] not in temp_list:
                temp_list.append(f[7:10])
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
        
        '''
        Threshold = self.Threshold
        # Define IPW array (6534 for 6 time steps of IPW fields, and one for ground truth)
        out_matrixIPW = np.zeros((len(temp_ipw_file_list),1089*6 + 1),dtype='float32')
        # Define reflectivity array same as IPW array        
        out_matrixRadar = np.zeros((len(temp_radar_file_list),1089*6),dtype='float32')
        i_start = x_ -16
        i_end = x_ + 17
        j_start = y_ -16
        j_end = y_ + 17
        # Initialize both the array to nan, this will make it easier to drop examples 
        # which do not have all the time frames as an example will need 2 hours of prior data
        # ex. of a case with no data will be start of new UTC day from the storm dates picked
        out_matrixIPW[:] = np.nan
        out_matrixRadar[:] = np.nan
        
        matrix_ctr = 0
    
        for i_file,r_file in zip(temp_ipw_file_list,temp_radar_file_list):
            RadarMatrix = np.load(Basefile + r_file).astype('float32')
    
            RadarMatrix[np.isnan(RadarMatrix)] = 0.0
            RadarMatrixFeature = RadarMatrix.copy()
            
            if Threshold:
                
                RadarMatrix[RadarMatrix < Threshold] = 0.0
    
                RadarMatrix[RadarMatrix >= Threshold] = 1.0
        
            pointXY = RadarMatrix[y_,x_]
            out_matrixIPW[matrix_ctr,-1] = pointXY
            # Current field time stem in is the first 1089 features
            IPWMatrix = np.load(Basefile + i_file).astype('float32')
            
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
    
    def sample_random_pixels(self):
        
        # We are going to choose a sub domain in our total 300x300 domain in DFW
        fill_domain = (range(17,83),range(17,83))

        PixelX = fill_domain[0]
        PixelY = fill_domain[1]

        # Pull the central chunk of points out
        central_chunk = (range(46,54),range(46,54))

        central_chunk_points = [(x_,y_) for x_ in central_chunk[0] for y_ in central_chunk[1]]

        # Pair up the pixel points
        PixelPoints = [(x,y) for x in PixelX for y in PixelY]

        # Remove all central points from the pair
        PixelPoints = [pairs for pairs in PixelPoints if pairs not in central_chunk_points]

        # Randomely sample 1500 pairs of points
        random.seed(12345)
        
        PixelPoints = [PixelPoints[x] for x in random.sample(range(4292),self.num_points)]

        PixelPoints = np.array(PixelPoints)
        
        return PixelPoints
    
    def make_points_frames(self,PixelPoints):
        self.days_in_sorted.sort()
        IPW_Refl_points = []
        for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
            print 'Building data set for point: (%d,%d)'%(x_,y_)
            for set_ in self.days_in_sorted:
                temp_ipw_file_list = filter(lambda x: x[7:10] in self.sorted_days[set_],self.IPWfiles)
                temp_radar_file_list = filter(lambda x: x[9:12] in self.sorted_days[set_],self.Radarfiles)
                tmp_array = self.build_features_and_truth(temp_ipw_file_list,temp_radar_file_list,x_,y_)
                IPW_Refl_points.append(tmp_array)

        return IPW_Refl_points
    
    # This function is for arranging frames with list of tuples tmp_array = (out_matrixIPW,out_matrixRadar)
    def arrange_frames(self,IPW_Refl_points):
        # Load IPW frames
        IPWFeatures = np.concatenate(map(lambda x: x[0].astype('float32'),IPW_Refl_points))      
        # Drop all time stamps which do not have the last 4 frames or any row that has a nan value        
        IPWFeatures = IPWFeatures[~np.any(np.isnan(IPWFeatures),axis = 1),:]
        # Load ground truth
        Y = IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)
        # Load Refl. frames
        ReflFeatures = np.concatenate(map(lambda x: x[1].astype('float32'),IPW_Refl_points))
        # Drop all time stamps which do not have the last 4 frames        
        ReflFeatures = ReflFeatures[~np.any(np.isnan(ReflFeatures),axis = 1),:]
        # Stack frames into volumes
        IPWFeatures = IPWFeatures[:,:-1].reshape(IPWFeatures.shape[0],6,33,33)
        ReflFeatures = ReflFeatures.reshape(ReflFeatures.shape[0],6,33,33)
        # Use frames from one hour ago, this will drip the current frame and frame at t -30
        IPWFeatures = IPWFeatures[:,2:,:,:]
        ReflFeatures = ReflFeatures[:,2:,:,:]
        # Merge IPW and reflectivity to create volume of shape number of examples x 8 x 33 x 33
        X = np.concatenate((IPWFeatures,ReflFeatures),axis=1)
        # return data sets
        return X,Y
    
    # This function is to arrange frames with single days or string of days
    def arrange_frames_single(self,IPW_Refl_points):
        # Load IPW frames
        IPWFeatures = IPW_Refl_points[0]
        # Drop all time stamps which do not have the last 4 frames or any row that has a nan value        
        IPWFeatures = IPWFeatures[~np.any(np.isnan(IPWFeatures),axis = 1),:]
        # Load ground truth
        Y = IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)
        # Load Refl. frames
        ReflFeatures = IPW_Refl_points[1]
        # Drop all time stamps which do not have the last 4 frames        
        ReflFeatures = ReflFeatures[~np.any(np.isnan(ReflFeatures),axis = 1),:]
        # Stack frames into volumes
        IPWFeatures = IPWFeatures[:,:-1].reshape(IPWFeatures.shape[0],6,33,33)
        ReflFeatures = ReflFeatures.reshape(ReflFeatures.shape[0],6,33,33)
        # Use frames from one hour ago, this will drip the current frame and frame at t -30
        IPWFeatures = IPWFeatures[:,2:,:,:]
        ReflFeatures = ReflFeatures[:,2:,:,:]
        # Merge IPW and reflectivity to create volume of shape number of examples x 8 x 33 x 33
        X = np.concatenate((IPWFeatures,ReflFeatures),axis=1)
        # return data sets
        return X,Y
        
#def load_data_set(set_no = 3):
#    # Temporarely route the data from summer
#    file_list = os.listdir('data/TrainTest/RandomPoints/')
#    file_list = filter(lambda x: x[:3] == 'IPW',file_list)
#    random_points_file = file('data/TrainTest/RandomPoints/' + file_list[set_no],'rb')
#    data_set = cPickle.load(random_points_file)
#    random_points_file.close()
#    return data_set
    
#def build_data_set(set_no):
#    # Load entire 1500 points
#    data = load_data_set(set_no = set_no)
#    
#    IPWFeatures = np.concatenate(map(lambda x: x[0].astype('float32'),data))
#    
#    Y_train = IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)
#    
#    IPWFeatures = IPWFeatures[:,:-1].reshape(IPWFeatures.shape[0],6,33,33)
#    
#    IPWFeatures = IPWFeatures[:,2:,:,:]
#    
#    ReflFeatures = np.concatenate(map(lambda x: x[1].astype('float32'),data))
#    
#    ReflFeatures = ReflFeatures.reshape(IPWFeatures.shape[0],6,33,33)
#    
#    ReflFeatures = ReflFeatures[:,2:,:,:]
#    
#    X_train = np.concatenate((IPWFeatures,ReflFeatures),axis=1)
#    
#    print X_train.shape,Y_train.shape
#    
#    return X_train,Y_train
        



















