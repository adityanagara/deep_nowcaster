# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:23:41 2015

@author: adityanagarajan
"""

import os
import numpy as np
import random
import getpass
import re 
import ftplib
import subprocess
from matplotlib import pyplot as plt

'''
This package deals with building data sets for training a machine learning
based nowcasting system. This means we can build 4 frames of IPW and 4 frames
of reflectivity for each sampled point. 
'''
def cart2pol(x,y):
    r = np.sqrt(np.power(x,2) + np.power(y,2))
    theta = np.degrees(np.arctan2(y,x))
    return theta,r

class dataset(object):
    '''This class builds the data set given the pixel points'''
    def __init__(self,Threshold = 'binary',num_points = 1500):
        self.Threshold = Threshold
        self.num_points = num_points 

    def get_data_path(self,yr):
        user = getpass.getuser()
        if user == 'adityanagarajan':
            return '/Users/adityanagarajan/projects/nowcaster/data/dataset/20' + str(yr) 
        else:
            return '/mnt/deep_nowcaster/data/dataset/20' + str(yr)
    def sort_IPW_refl_files(self,yr):
        # get all the files in the dataset folder
        files = os.listdir(self.get_data_path(yr))
        
        files = filter(lambda x: 'img' not in x,files)
        
        # Filter based on IPW and reflectivity files
        IPWfiles = filter(lambda x: x[:7] == 'IPWdata' and re.findall('\d+',x)[0] == str(yr),files)
        Radarfiles = filter(lambda x: x[:9] == 'RadarRefl' and re.findall('\d+',x)[0] == str(yr),files)
        
        #Sort files DOY.hh
        IPWfiles.sort(key = lambda x: float(re.findall('\d+',x)[1]) + float(re.findall('\d+',x)[2]) * 0.01)
        Radarfiles.sort(key = lambda x: float(re.findall('\d+',x)[1]) + float(re.findall('\d+',x)[2])* 0.01)
        
        return IPWfiles,Radarfiles
        
    def sort_IPW_refl_files_imgs(self,yr):
        # get all the files in the dataset folder
        files = os.listdir(self.get_data_path(yr))
        
        # Filter the image files 
        files = filter(lambda x: 'img' in x,files)
        
        # Filter based on IPW and reflectivity files
        IPWfiles = filter(lambda x: x[:7] == 'IPWdata' and re.findall('\d+',x)[0] == str(yr),files)
        Radarfiles = filter(lambda x: x[:9] == 'RadarRefl' and re.findall('\d+',x)[0] == str(yr),files)
        
        #Sort files DOY.hh
        IPWfiles.sort(key = lambda x: float(re.findall('\d+',x)[1]) + float(re.findall('\d+',x)[2]) * 0.01)
        Radarfiles.sort(key = lambda x: float(re.findall('\d+',x)[1]) + float(re.findall('\d+',x)[2])* 0.01)
        
        return IPWfiles,Radarfiles
        

    
    def load_storm_days(self,yr,kill_dates = True):
        if yr == 14:
            storm_dates = np.load('../data/storm_dates_2014.npy').astype('int')
            # The following 3 dates are going to be removed because we do not
            # have NEXRAD files for the entire day            
            idx1 = np.where(np.all(storm_dates == [205,  14,   7,  24],axis=1))[0][0]
            storm_dates = np.delete(storm_dates,idx1,axis = 0)
            idx2 = np.where(np.all(storm_dates == [176,  14,   6,  25],axis=1))[0][0]
            storm_dates = np.delete(storm_dates,idx2,axis = 0)
            idx3 = np.where(np.all(storm_dates == [204 , 14 ,  7 , 23],axis=1))[0][0]
            storm_dates = np.delete(storm_dates,idx3,axis = 0)
        elif yr == 15:
            storm_dates = np.load('../data/storm_dates_2015.npy').astype('int')
            # Let us not use the days in July since they are all stratiform
            # cases 
            storm_dates = storm_dates[:-10,:]
        elif yr == 16:
            storm_dates = np.load('../data/storm_dates_2016.npy').astype('int')
        if kill_dates:
            if yr == 14:
                kill_these_dates = [[142,  14,   5,  22],
                                    [143,  14,   5,  23],
                                    [148,  14,   5,  28],
                                    [151,  14,   5,  31],
                                    [170,  14,   6,  19],
                                    [173,  14,   6,  22],
                                    [175,  14,   6,  24],
                                    [179,  14,   6,  28],
                                    [183,  14,   7,   2],
                                    [199,  14,   7,  18]
                                    ]
                for date in kill_these_dates:
                    idx = np.where(np.all(storm_dates == date,axis = 1))[0][0]
                    storm_dates = np.delete(storm_dates,idx,axis=0)
            elif yr == 15:
                kill_these_dates = [[132,  15,   5,  12],
                                    [138,  15,   5,  18],
                                    [140,  15,   5,  20],
                                    [142,  15,   5,  22],
                                    [147,  15,   5,  27],
                                    [170,  15,   6,  19]
                                    ]
                for date in kill_these_dates:
                    idx = np.where(np.all(storm_dates == date,axis = 1))[0][0]
                    storm_dates = np.delete(storm_dates,idx,axis = 0)
        return storm_dates
    def club_days(self,storm_dates):
        '''This function finds days in our storm dates list and clubs
        strings of days together'''
        temp_list = storm_dates[:,0].astype('S')
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
    
    def convert_IPW_img(self,arr):
        '''Converts the field array to grey scale image
        input: 100 x 100 ipw field, dtype = float
        output: 100 x 100 greyscale image, dtype = int 64'''
        map_ipw_array = np.linspace(-4,4,256,dtype='float')
        new_array_ipw = np.zeros(arr.shape,dtype='int')
        for i in range(arr.shape[0]):
            for j in range(arr.shape[0]):
                if arr[i,j] < -4.0 or arr[i,j] > 4.0:
                    print 'value out of bounds %f'%arr[i,j]
                new_array_ipw[i,j] = np.argmin(np.abs(arr[i,j] - map_ipw_array))
        return new_array_ipw
    
    def convert_reflectivity_img(self,arr):
        '''Converts the field array to grey scale image
        input: 100 x 100 reflectivity field, dtype = float
        output: 100 x 100 greyscale image, dtype = int 64'''
        map_refl_array = np.linspace(0,90,256,dtype='float')
        arr[np.isnan(arr)] = 0
        arr[arr<0] = 0
        new_array_refl = np.zeros((100,100),dtype='int')
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):        
                new_array_refl[i,j] = np.argmin(np.abs(arr[i,j] - map_refl_array))
        return new_array_refl
    
    def build_features_and_truth(self,temp_ipw_file_list,temp_radar_file_list,x_,y_):
        '''The structure of the out array is as follows:
        return tuple(ipw_array,reflectivity_array)
        ipw_array -> shape = n_examples x 6534 (6 x 1089(33x33 stretched vector) + 1)
        t,t - 30,t - 60,t - 90,t - 120,t - 150, rainfall_flag
        reflectivity_array -> shape = n_examples x (6 x 1089)'''
        Threshold = self.Threshold
        # Define IPW array (6534 for 6 time steps of IPW fields, and one for ground truth)
        out_matrixIPW = np.zeros((len(temp_ipw_file_list),1089*6 + 1),dtype='float')
        # Define reflectivity array same as IPW array        
        out_matrixRadar = np.zeros((len(temp_radar_file_list),1089*6),dtype='float')
        i_start = x_ -16
        i_end = x_ + 17
        j_start = y_ -16
        j_end = y_ + 17
        # Initialize both the array to nan, this will make it easier to drop examples 
        # which do not have all the time frames as an example will need 2 hours of prior data
        # ex. of a case with no data will be start of new UTC day from the storm dates picked
        # The data case will only contain data at 0200 UTC where we will have 4 frames
        out_matrixIPW[:] = np.nan
        out_matrixRadar[:] = np.nan
        
        matrix_ctr = 0
    
        for i_file,r_file in zip(temp_ipw_file_list,temp_radar_file_list):
            RadarMatrix = np.load(r_file)
    
            RadarMatrix[np.isnan(RadarMatrix)] = 0.0
            RadarMatrix[RadarMatrix<0.0] = 0.0
            RadarMatrixFeature = RadarMatrix.copy()
            
            if Threshold == 'binary':
                
                RadarMatrix[RadarMatrix < 24.0] = 0
                RadarMatrix[RadarMatrix >= 24.0] = 1
                
            elif Threshold == 'multiclass':
                print('Bin them reflectivity')
                RadarMatrix[RadarMatrix < 10.0] = 0
                
                RadarMatrix[np.logical_and(RadarMatrix >= 10.0,RadarMatrix < 20.0)] = 1
                
                RadarMatrix[np.logical_and(RadarMatrix >= 20.0,RadarMatrix < 30.0)] = 2
                
                RadarMatrix[np.logical_and(RadarMatrix >= 30.0,RadarMatrix < 40.0)] = 3
                
                RadarMatrix[np.logical_and(RadarMatrix >= 40.0,RadarMatrix < 50.0)] = 4
                
                RadarMatrix[RadarMatrix >= 50.0] = 5
                
            pointXY = RadarMatrix[y_,x_]
            out_matrixIPW[matrix_ctr,-1] = pointXY
            IPWMatrix = np.load(i_file)
            # Convert the IPW and reflectivity fields to images
#            IPWMatrix = self.convert_IPW_img(IPWMatrix)
#            RadarMatrixFeature = self.convert_reflectivity_img(RadarMatrixFeature)
            # Current field time step is the first 1089 feature vector (33 x 33 = 1089)
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
    
    def build_features_and_truth_imgs(self,temp_ipw_file_list,temp_radar_file_list,x_,y_):
        '''The structure of the out array is as follows:
        return tuple(ipw_array,reflectivity_array)
        ipw_array -> shape = n_examples x 6534 (6 x 1089(33x33 stretched vector) + 1)
        t,t - 30,t - 60,t - 90,t - 120,t - 150, rainfall_flag
        reflectivity_array -> shape = n_examples x (6 x 1089)'''
        Threshold = self.Threshold
        # Define IPW array (6534 for 6 time steps of IPW fields, and one for ground truth)
        out_matrixIPW = np.zeros((len(temp_ipw_file_list),1089*6 + 1),dtype='float')
        # Define reflectivity array same as IPW array        
        out_matrixRadar = np.zeros((len(temp_radar_file_list),1089*6),dtype='float')
        i_start = x_ -16
        i_end = x_ + 17
        j_start = y_ -16
        j_end = y_ + 17
        # Initialize both the array to nan, this will make it easier to drop examples 
        # which do not have all the time frames as an example will need 2 hours of prior data
        # ex. of a case with no data will be start of new UTC day from the storm dates picked
        # The data case will only contain data at 0200 UTC where we will have 4 frames
        out_matrixIPW[:] = np.nan
        out_matrixRadar[:] = np.nan
        
        matrix_ctr = 0
    
        for i_file,r_file in zip(temp_ipw_file_list,temp_radar_file_list):
            truth_file = r_file.replace('_img','')
            RadarMatrix = np.load(truth_file)
    
            RadarMatrix[np.isnan(RadarMatrix)] = 0.0
            RadarMatrix[RadarMatrix<0.0] = 0.0
            RadarMatrixFeature = np.load(r_file)
            
            if Threshold == 'binary':
                
                RadarMatrix[RadarMatrix < 24.0] = 0
                RadarMatrix[RadarMatrix >= 24.0] = 1
                
            elif Threshold == 'multiclass':
                print('Bin them reflectivity')
                RadarMatrix[RadarMatrix < 10.0] = 0
                
                RadarMatrix[np.logical_and(RadarMatrix >= 10.0,RadarMatrix < 20.0)] = 1
                
                RadarMatrix[np.logical_and(RadarMatrix >= 20.0,RadarMatrix < 30.0)] = 2
                
                RadarMatrix[np.logical_and(RadarMatrix >= 30.0,RadarMatrix < 40.0)] = 3
                
                RadarMatrix[np.logical_and(RadarMatrix >= 40.0,RadarMatrix < 50.0)] = 4
                
                RadarMatrix[RadarMatrix >= 50.0] = 5
                
            pointXY = RadarMatrix[y_,x_]
            out_matrixIPW[matrix_ctr,-1] = pointXY
            IPWMatrix = np.load(i_file)
            # Current field time step is the first 1089 feature vector (33 x 33 = 1089)
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
        
        year_day_time = np.array(map(lambda x: re.findall('\d+',x.split('/')[-1]),temp_ipw_file_list),dtype = 'int')
        
        return (year_day_time,out_matrixIPW,out_matrixRadar)

    
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
    # To build a data set we first enter this function where a set of points is given as input
    # Then we iterate through all of the storm days in the particular year and create a data set
    def make_points_frames(self,PixelPoints):
        self.days_in_sorted.sort()
        IPW_Refl_points = []
        for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
            print 'Building data set for point: (%d,%d)'%(x_,y_)
            for set_ in self.days_in_sorted:
                temp_ipw_file_list = filter(lambda x: x[9:12] in self.sorted_days[set_],self.IPWfiles_imgs)
                temp_radar_file_list = filter(lambda x: x[11:14] in self.sorted_days[set_],self.Radarfiles_imgs)
                tmp_array = self.build_features_and_truth(temp_ipw_file_list,temp_radar_file_list,x_,y_)
                IPW_Refl_points.append(tmp_array)

        return IPW_Refl_points
    
    # This function is for arranging frames with list of tuples tmp_array = [(out_matrixIPW,out_matrixRadar)]
    def arrange_frames(self,IPW_Refl_points):
        # Load IPW frames
        IPWFeatures = np.concatenate(map(lambda x: x[0],IPW_Refl_points))      
        # Drop all time stamps which do not have the last 4 frames or any row that has a nan value        
        IPWFeatures = IPWFeatures[~np.any(np.isnan(IPWFeatures),axis = 1),:]
        # Load ground truth
        Y = IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)
        # Load Refl. frames
        ReflFeatures = np.concatenate(map(lambda x: x[1],IPW_Refl_points))
        # Drop all time stamps which do not have the last 4 frames        
        ReflFeatures = ReflFeatures[~np.any(np.isnan(ReflFeatures),axis = 1),:]
        # Stack frames into volumes
        IPWFeatures = IPWFeatures[:,:-1].reshape(IPWFeatures.shape[0],6,33,33)
        ReflFeatures = ReflFeatures.reshape(ReflFeatures.shape[0],6,33,33)
        # Use frames from one hour ago, this will drip the current frame and frame at t -30
        IPWFeatures = IPWFeatures[:,2:,:,:]
        ReflFeatures = ReflFeatures[:,2:,:,:]
        # Merge IPW and reflectivity to create volume of shape number of examples x 8 x 33 x 33
        X = np.concatenate((IPWFeatures,ReflFeatures),axis=1).astype('uint8')
        # return data sets
        return X,Y
    
    # This function is to arrange frames with single days or string of days
    def arrange_frames_single(self,IPW_Refl_points):
        year_day_time = IPW_Refl_points[0]
        IPWFeatures = IPW_Refl_points[1]
        year_day_time = year_day_time[~np.any(np.isnan(IPWFeatures),axis = 1),:]
        # Load IPW frames
        # Drop all time stamps which do not have the last 4 frames or any row that has a nan value        
        IPWFeatures = IPWFeatures[~np.any(np.isnan(IPWFeatures),axis = 1),:]
        # Load ground truth
        Y = IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)
        # Load Refl. frames
        ReflFeatures = IPW_Refl_points[2]
        # Drop all time stamps which do not have the last 4 frames        
        ReflFeatures = ReflFeatures[~np.any(np.isnan(ReflFeatures),axis = 1),:]
        # Stack frames into volumes
        IPWFeatures = IPWFeatures[:,:-1].reshape(IPWFeatures.shape[0],6,33,33)
        ReflFeatures = ReflFeatures.reshape(ReflFeatures.shape[0],6,33,33)
        # Use frames from one hour ago, this will drop the current frame and frame at t -30
        IPWFeatures = IPWFeatures[:,2:,:,:]
        ReflFeatures = ReflFeatures[:,2:,:,:]
        # Merge IPW and reflectivity to create volume of shape number of examples x 8 x 33 x 33
        X = np.concatenate((IPWFeatures,ReflFeatures),axis = 1)
        # return data sets
        return (year_day_time,X.astype('uint8'),Y.astype('uint8'))
    
    def arrange_frames_CCA_experiment(self,IPW_Refl_points):
        year_day_time = IPW_Refl_points[0]
        IPWFeatures = IPW_Refl_points[1]
        # Load IPW frames
        # Load ground truth
        Y = IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)
        # Load Refl. frames
        ReflFeatures = IPW_Refl_points[2]
        # Stack frames into volumes
        IPWFeatures = IPWFeatures[:,:-1].reshape(IPWFeatures.shape[0],6,33,33)
        ReflFeatures = ReflFeatures.reshape(ReflFeatures.shape[0],6,33,33)
        # Use frames from one hour ago, this will drip the current frame and frame at t -30
        IPWFeatures = IPWFeatures[:,0,:,:]
        ReflFeatures = ReflFeatures[:,0,:,:]
        # Merge IPW and reflectivity to create volume of shape number of examples x 8 x 33 x 33
#        X = np.concatenate((IPWFeatures,ReflFeatures),axis=1)
        # return data sets
        return (year_day_time,IPWFeatures.astype('uint8'),ReflFeatures.astype('uint8'))
    
    def get_field_statistics(self,random_points):
        '''Takes as input a tuple with (time_array,ipw_features,reflectivity_features)
        Averages the fields for each time step'''
        indices_i = [(x*1089,(x+1)*1089)for x in range(4) ]
        indices_r = [(x*1089,(x+1)*1089)for x in range(4,8) ]
        IPWFeatures = random_points[1].astype('float')
        ReflFeatures = random_points[2].astype('float')
        # 1089 is the number of pixels per 33x33 image. When we flatten the entire 
        # matrix then we get a vector. Since we are using the data from an hour back
        # we use all the pixels from the slice 2178. The last element of the IPW 
        # matrix contains the ground truth so we slice up to that not including 
        # that value. 
        data = np.hstack((IPWFeatures[:,2178:-1],ReflFeatures[:,2178:],IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)))
        # For some time steps we may not have 2 hours worth of prior data which are 
        # represented by nan. We drop these examples. 
        data = data[~np.any(np.isnan(data),axis = 1),:]
        # Initialize the data array 0-4 IPW average, 4-8 IPW STD, 8-12 Refl average 
        # 12-16 Refl standard deviation, 17 output variable (binary/multiclass)
        ipw_refl_stats = np.zeros((data.shape[0],17))
        ipw_refl_stats[:,-1] = data[:,-1]
        for ix in range(len(indices_i)):
            # IPW averages and standard deviations
            ipw_refl_stats[:,ix] = np.average(data[:,indices_i[ix][0]:indices_i[ix][1]],axis = 1)
            ipw_refl_stats[:,ix + 4] = np.std(data[:,indices_i[ix][0]:indices_i[ix][1]],axis = 1)
            # Reflectivity averages and standard deviations
            ipw_refl_stats[:,ix + 8] = np.average(data[:,indices_r[ix][0]:indices_r[ix][1]],axis = 1)
            ipw_refl_stats[:,ix + 12] = np.std(data[:,indices_r[ix][0]:indices_r[ix][1]],axis = 1)
        # redurn the array of shape (num_examples,17)
        return ipw_refl_stats
    
    def get_temporal_statistics(self,ipw_refl_stats):
        temporal_stats = np.zeros((ipw_refl_stats.shape[0],10))
        # IPW temporal features
        temporal_stats[:,0] = ipw_refl_stats[:,0] - ipw_refl_stats[:,1]
        temporal_stats[:,1] = ipw_refl_stats[:,1] - ipw_refl_stats[:,2]
        temporal_stats[:,2] = ipw_refl_stats[:,2] - ipw_refl_stats[:,3]
        temporal_stats[:,3] = ipw_refl_stats[:,0] - ipw_refl_stats[:,2]
        temporal_stats[:,4] = ipw_refl_stats[:,1] - ipw_refl_stats[:,3]
        # Reflectivity temporal features
        temporal_stats[:,5] = ipw_refl_stats[:,8] - ipw_refl_stats[:,9]
        temporal_stats[:,6] = ipw_refl_stats[:,9] - ipw_refl_stats[:,10]
        temporal_stats[:,7] = ipw_refl_stats[:,10] - ipw_refl_stats[:,11]
        temporal_stats[:,8] = ipw_refl_stats[:,8] - ipw_refl_stats[:,10]
        temporal_stats[:,9] = ipw_refl_stats[:,9] - ipw_refl_stats[:,11]
        
        return np.concatenate((temporal_stats,ipw_refl_stats),axis = 1)
        
        
    def get_field_statistics_30minPrediction(self,random_points):
        '''build basic features for a 30 minute in advance prediction'''
        indices_i = [(x*1089,(x+1)*1089)for x in range(4) ]
        indices_r = [(x*1089,(x+1)*1089)for x in range(4,8) ]
        IPWFeatures = random_points[1].astype('float')
        
        ReflFeatures = random_points[2].astype('float')
        # 1089 is the number of pixels per 33x33 image. When we flatten the entire 
        # matrix then we get a vector. Since we are using the data from an hour back
        # we use all the pixels from the slice 2178. The last element of the IPW 
        # matrix contains the ground truth so we slice up to that not including 
        # that value. 
        data = np.hstack((IPWFeatures[:,1089:5445],ReflFeatures[:,1089:5445],IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)))
        # For some time steps we may not have 2 hours worth of prior data which are 
        # represented by nan. We drop these examples. 
        data = data[~np.any(np.isnan(data),axis = 1),:]
        # Initialize the data array 0-4 IPW average, 4-8 IPW STD, 8-12 Refl average 
        # 12-16 Refl standard deviation, 17 output variable (binary/multiclass)
        ipw_refl_stats = np.zeros((data.shape[0],17))
        ipw_refl_stats[:,-1] = data[:,-1]
            
        for ix in range(len(indices_i)):
            # IPW averages and standard deviations
            ipw_refl_stats[:,ix] = np.average(data[:,indices_i[ix][0]:indices_i[ix][1]],axis = 1)
            ipw_refl_stats[:,ix + 4] = np.std(data[:,indices_i[ix][0]:indices_i[ix][1]],axis = 1)
            # Reflectivity averages and standard deviations
            ipw_refl_stats[:,ix + 8] = np.average(data[:,indices_r[ix][0]:indices_r[ix][1]],axis = 1)
            ipw_refl_stats[:,ix + 12] = np.std(data[:,indices_r[ix][0]:indices_r[ix][1]],axis = 1)
        # redurn the array of shape (num_examples,17)
        return ipw_refl_stats



class reflectivity_fields(object):
    def FTPNEXRADfile(self,mon,day,yr,order_id):
        '''Downloads a reflectivity file from the NCDC database'''
        file_to_get = 'NWS_NEXRAD_NXL3_KFWS_20' +yr + mon + day + '000000_20' + yr + mon + day + '235959.tar.gz'
        ftp_NEXRAD = ftplib.FTP('ftp.ncdc.noaa.gov','anonymous','adi@gmail.com')  
        ftp_NEXRAD.cwd('pub/has' + os.sep + order_id + os.sep)
        file_list = ftp_NEXRAD.nlst()
        if file_to_get in file_list:
            print 'We Going to get that file: ' + file_to_get
            gfile = open(file_to_get,'wb')
            ftp_NEXRAD.retrbinary('RETR ' + file_to_get,gfile.write)
            gfile.close()
        else:
            print 'FATAL: File not found ' +  file_to_get
        ftp_NEXRAD.close()
        subprocess.call(['tar','-xvzf',file_to_get])
        subprocess.call(['rm',file_to_get])
    
    def keepLevel3files(self,folder_path):
        '''This function only keeps the level 3 NEXRAD files and deletes
        the rest given a particular folder.'''
        file_list = os.listdir(folder_path)
        delete_files = filter(lambda x: x[:18] != 'KFWD_SDUS54_N0RFWS',file_list)
        for df in delete_files:
            os.remove(folder_path + os.sep + df)
    
    def ConvertToNETCDF(self,folder_path):
        '''Use the java toolsUI-4.6.jar to convert to .nc files. Takes a folder
        path of raw files and dumps into same folder but removes original file'''
        java_script = 'toolsUI-4.6.jar'
        ucar = 'ucar.nc2.FileWriter'
        keep_files = os.listdir(folder_path)
        for raw_file in keep_files:
            temp_in = folder_path + raw_file
            temp_out = folder_path + raw_file + '.nc'
            subprocess.call(['java','-classpath',java_script,ucar,'-in',temp_in,'-out',temp_out,])
            # remove the raw file we only need .nc
            os.remove(folder_path + raw_file)

    
    def reflectivity_polar_to_cartesian(self,rad):
        '''Given a nexrad dataset object return the cartesian plot 
        of the reflectivity field'''
        m = 100
        # Initialize an empty array to hold the reflectivity values in the cartesian coordinates
        gridZ = np.empty((m,m))
        gridZ.fill(np.nan)
        # Make the 150x150 km2 grid
        gridX = np.arange(-150.0,151.0,300.0/(m-1))
        gridY = np.arange(-150.0,151.0,300.0/(m-1))

        xMesh,yMesh = np.meshgrid(gridX,gridY)
        xMesh,yMesh = np.meshgrid(gridX,gridY)
    
        gridA,gridR = cart2pol(xMesh,yMesh)
        # Get the vector of azimuth angles
        azimuthVector = rad.variables['azimuth'][:]

        # Get the range gates
        rangeVector = rad.variables['gate'][:]

        startRange = rangeVector[0]

        gateWidth = np.median(np.diff(rangeVector))

        startRange = startRange /1000.0
        gateWidth = gateWidth / 1000.0

        # Get the level 3 products
        Z = rad.variables['BaseReflectivity'][:]
    
        for a in range(azimuthVector.size):
        
            I = np.less(np.abs(gridA - azimuthVector[a]),1.0)
    
            J = np.floor(((gridR[np.abs(gridA - azimuthVector[a]) < 1.0] - startRange)/gateWidth ))
    
            gridZ[I] = Z[a,tuple(J)]
    
        return gridZ.T

# Test case

def test_ipw_converter():
    data_builder = dataset()
    ipw_array = np.linspace(-5.,5.,10000).reshape(100,100)
    refl_array = np.linspace(0.,90.,10000).reshape(100,100)
    ipw_img = data_builder.convert_IPW_img(ipw_array)
    for a in range(ipw_img.shape[0]):
        print ipw_img[a,:]
        print ipw_array[a,:]

def test_refl_converter():
    data_builder = dataset()
    refl_array = np.linspace(0.,90.,10000).reshape(100,100)
    refl_img = data_builder.convert_reflectivity_img(refl_array)
    for a in range(refl_img.shape[0]):
        print refl_img[a,:]
        print refl_array[a,:]

def test_slicing_fields(x_,y_,ipw_array,refl_array):
    gridX = np.arange(-150.0,151.0,300.0/(100-1))
    gridY = np.arange(-150.0,151.0,300.0/(100-1))
    i_start = x_ - 16
    i_end = x_ + 17
    j_start = y_ -16
    j_end = y_ + 17
    gridIPW = np.zeros((33,33))
    gridZ = np.zeros((33,33))
    gridIPW[:] = ipw_array[j_start:j_end,i_start:i_end]
    gridZ[:] = refl_array[j_start:j_end,i_start:i_end]
    
    x_range_start = gridX[x_] - 16.0*(300.0/99.0)
    y_range_start = gridY[y_] - 16.0*(300.0/99.0)

    x_range_end = gridX[x_] + 17.0*(300.0/99.0)
    y_range_end = gridY[y_] + 17.0*(300.0/99.0)

    gridX_ = np.arange(x_range_start,x_range_end,300./99.)
    gridY_ = np.arange(y_range_start,y_range_end,300./99.)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.pcolor(gridX_,gridY_,gridIPW,cmap='jet', vmin=-3.0, vmax=3.0)
    plt.plot(gridX[x_],gridY[y_],'r*')
    plt.grid()
    
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))

    plt.subplot(2,2,2)

    plt.pcolor(gridX,gridY,ipw_array,cmap='jet', vmin=-3.0, vmax=3.0)
    plt.plot(gridX[x_],gridY[y_],'r*')
    plt.grid()
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))

    plt.subplot(2,2,3)
    plt.pcolor(gridX_,gridY_,gridZ,cmap='jet', vmin=10.0, vmax=60.0)
    plt.plot(gridX[x_],gridY[y_],'r*')
    plt.grid()
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))


    plt.subplot(2,2,4)
    plt.pcolor(gridX,gridY,refl_array,cmap='jet', vmin=10.0, vmax=60.0)
    plt.plot(gridX[x_],gridY[y_],'r*')
    plt.grid()
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))
    
    
def convert_fields_to_images():
    '''Convert the ipw and reflectivity field arrays to gray scale image 
    arrays'''
    data_builder = dataset()
    storm_dates_all = {}
    for yr in [14,15,16]:
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


def main():
#    test_ipw_refl_converter()
#    test_refl_converter()
    convert_fields_to_images()
#    x = dataset()
#    PixelPoints = x.sample_random_pixels()
#    ipw,refl = x.sort_IPW_refl_files(14)
#    base_path = '../data/dataset/2014/'
#    file_idx = 345
#    ipw_array = np.load(base_path + ipw[file_idx])
#    refl_array = np.load(base_path + refl[file_idx])
#    pt_idx = 65
#    test_slicing_fields(PixelPoints[pt_idx][0],PixelPoints[pt_idx][1],ipw_array ,refl_array)
    
    
    
    

if __name__ == '__main__':
    main()