# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:49:17 2016

@author: adityanagarajan
"""

import numpy as np
import os
from random import shuffle,seed
import cPickle as pkl
import re
import BuildDataSet

from sklearn.cross_decomposition import CCA



seed(1234)
def make_dataset_CCA(data_builder):
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
                
                ipw_refl_tensors = data_builder.arrange_frames_CCA_experiment(temp_array)
                
                points_array.append((ipw_refl_tensors))
                
                
        # Save each string of days to a pkl file
        save_path = '../data/TrainTest/CCA_points/'
        print save_path + 'IPW_refl_frames{0}_{1}.pkl'.format(x_,y_)
        temp_file = file(save_path + 'IPW_refl_frames{0}_{1}.pkl'.format(x_,y_),'wb')
        pkl.dump(points_array,temp_file,protocol = pkl.HIGHEST_PROTOCOL)
        temp_file.close()

def load_array():
    base_path = '../data/TrainTest/CCA_points/'
    file_list = os.listdir(base_path)
    ipw = []
    refl = []
    for fi in file_list[:2]:
        arr = np.load(base_path + fi)
        print len(arr)
        for set_ in range(len(arr)):
            ipw.append(arr[set_][1])
            refl.append(arr[set_][2])
    return np.concatenate(ipw).astype('float'),np.concatenate(refl).astype('float')

def fuse_fields(ipw,refl):
    cca = CCA(n_components = 10,max_iter = 1000)
    cca.fit(ipw,refl)
    return cca
    
def main(build_data=False):
    data_builder = BuildDataSet.dataset(num_points = 100)
    if build_data:
        make_dataset_CCA(data_builder)
    ipw,refl = load_array()
    
    print ipw.nbytes,refl.nbytes
    
    cca = fuse_fields(ipw.reshape(ipw.shape[0],-1),refl.reshape(refl.shape[0],-1))
    f1 = file('cca_pkl_file.pkl','wb')
    pkl.dump(cca,f1)
    f1.close()
    
    
    

if __name__ == '__main__':
    main()