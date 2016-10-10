# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:30:29 2016

@author: adityanagarajan
"""


# Load custom modules
import DFWnet
import BuildDataSet
import ModelMetrics
# Load modules required
import numpy as np
import re
import os
import cPickle as pkl
import sys

# Load modules from sklearn package
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_decomposition import CCA

def get_val_doys(storm_dates):
    '''May: 121-151, June: 152 - 181 July: 182 - 212 August: 213 - 243'''
    yr_mon = {14: [5,6,7,8],
              15: [5,6],
              16: [5,6,7]}
    blocks = [str(yr) + '_' + str(mon) for yr in [14,15,16] for mon in yr_mon[yr]]
    val_blocks = []
    train_blocks = []
    for bl in blocks:
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
    for yr in [14,15,16]:
        storm_dates_all[yr] = data_builder.load_storm_days(yr)    
    storm_dates_all = np.concatenate((storm_dates_all[14],storm_dates_all[15],storm_dates_all[16]))
    train,val = get_val_doys(storm_dates_all)
    return train,val

def fit_CCA(tr_block,data_builder):
    '''We fit a CCA to some 100 odd points???
    '''
    # train on number of points
    num_points = 100
    PixelPoints = data_builder.sample_random_pixels()
    points_array_ipw = []
    points_array_refl = []
    for yr in [14,15]:
        doy_strings = data_builder.club_days(tr_block[tr_block[0][:,1] == yr])
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
            for x_,y_ in zip(PixelPoints[:num_points,0],PixelPoints[:num_points,1]):
                temp_array = data_builder.build_features_and_truth_imgs(temp_ipw_files,temp_refl_files,x_,y_)
                points_array_ipw.append(temp_array[1])
                points_array_refl.append(temp_array[2])
    X_ = np.vstack(points_array_ipw)
    Y_ = np.vstack(points_array_refl)
    mdl = CCA(n_components = 10)
    print 'Fitting a CCA...'
    mdl.fit(X_[:,:1089],Y_[:,:1089])
    ipw_frames = X_[:,2178:-1]
    refl_frames = Y_[:,2178:]
    del X_
    del Y_
    ipw_frames = ipw_frames[~np.any(np.isnan(ipw_frames),axis = 1),:]
    refl_frames = refl_frames[~np.any(np.isnan(refl_frames),axis = 1),:]
    
#    indices = [(x*1089,(x+1)*1089)for x in range(4) ]
#    # the number of components times 4
#    ipw_refl_fusion = np.zeros((ipw_frames.shape[0],80))
    print 'Building the feature fusion..'
    
    return mdl

def transform_CCA(cca_mdl,points_array_ipw,points_array_refl):
    X_ = np.vstack(points_array_ipw)
    Y_ = np.vstack(points_array_refl)
    ipw_frames = X_[:,2178:]
    refl_frames = Y_[:,2178:]
    ipw_frames = ipw_frames[~np.any(np.isnan(ipw_frames),axis = 1),:]
    refl_frames = refl_frames[~np.any(np.isnan(refl_frames),axis = 1),:]
    ipw_refl_fusion = np.zeros((ipw_frames.shape[0],81))
    ipw_refl_fusion[:,-1] = ipw_frames[:,-1]
    del X_
    del Y_
    indices = [(x*1089,(x+1)*1089)for x in range(4) ]
    for ix in range(4):
        temp_1 = cca_mdl.transform(ipw_frames[:,indices[ix][0]:indices[ix][1]],refl_frames[:,indices[ix][0]:indices[ix][1]])
        ipw_refl_fusion[:,ix*10:(ix + 1)*10] = temp_1[0]
        ipw_refl_fusion[:,(ix + 4)*10:(ix + 5)*10] = temp_1[1]
    
    return ipw_refl_fusion
    
    
def transform_predict_RF(cca_mdl,tr_block,val_block,data_builder):
    PixelPoints = data_builder.sample_random_pixels()
    train_fused_features = []
    for yr in [14,15]:
        doy_strings = data_builder.club_days(tr_block[tr_block[:,1] == yr])
        days_in_sorted = doy_strings.keys()
        days_in_sorted.sort()
        ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(yr)
        for set_ in days_in_sorted:
            points_array_ipw = []
            points_array_refl = []
            print 'Building data set for year: %d and string of days %s'%(yr,set_)
            # Get the required files only
            temp_ipw_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],ipw_files)
            temp_refl_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],refl_files)
            temp_ipw_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_ipw_files)
            temp_refl_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_refl_files)
            for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
                temp_array = data_builder.build_features_and_truth_imgs(temp_ipw_files,temp_refl_files,x_,y_)
                points_array_ipw.append(temp_array[1])
                points_array_refl.append(temp_array[2])
            ipw_refl_fusion = transform_CCA(cca_mdl,points_array_ipw,points_array_refl)
            
            train_fused_features.append(ipw_refl_fusion)
    
    train_fused_features = np.vstack(train_fused_features)
    
    print 'Fitting RF classifier'
    RF_mdl = RandomForestClassifier(n_estimators = 400,n_jobs = -1,max_features = 'auto')
    RF_mdl.fit(train_fused_features[:,:-1],train_fused_features[:,-1])
    
    doy_strings = data_builder.club_days(val_block)
    days_in_sorted = doy_strings.keys()
    days_in_sorted.sort()
    ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(val_block[0,1])
    val_fused_features = []
    for set_ in days_in_sorted:
        points_array_ipw = []
        points_array_refl = []
        print 'Building data set for year: %d and string of days %s'%(val_block[0,1],set_)
        # Get the required files only
        temp_ipw_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],ipw_files)
        temp_refl_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],refl_files)
        temp_ipw_files = map(lambda x: '../data/dataset/20' + str(val_block[0,1]) + os.sep + x,temp_ipw_files)
        temp_refl_files = map(lambda x: '../data/dataset/20' + str(val_block[0,1]) + os.sep + x,temp_refl_files)
        for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
            temp_array = data_builder.build_features_and_truth_imgs(temp_ipw_files,temp_refl_files,x_,y_)
            points_array_ipw.append(temp_array[1])
            points_array_refl.append(temp_array[2])
        ipw_refl_fusion = transform_CCA(cca_mdl,points_array_ipw,points_array_refl)
        val_fused_features.append(ipw_refl_fusion)
    
    val_fused_features = np.vstack(val_fused_features)
    Y_hat = RF_mdl.predict(val_fused_features[:,:-1])
    Y_score = RF_mdl.predict_proba(val_fused_features[:,:-1])
    
    performance_validation = ModelMetrics.NOWCAST_performance((Y_hat,val_fused_features[:,-1],Y_score[:,1]))
    print 'Precision = %.2f'%performance_validation.p_score
    print 'Recall = %.2f'%performance_validation.r_score
    print 'F1 score = %.2f'%performance_validation.f1
    print 'Area under the curve = %.2f'%performance_validation.average_precision
        
    print 'POD = %.2f'%performance_validation.POD
    print 'FAR = %.2f'%performance_validation.FAR
    print 'CSI = %.2f'%performance_validation.CSI            
    
def arrange_training_validation(tr,val):
    '''return a liat of tuples containing 7 elements where each tuple represents
    the files needed to be loaded for the training and validation sets'''
    file_list = []
    for yr in [14,15,16]:
        file_list.extend(os.listdir('../data/TrainTest/20' + str(yr) + os.sep))
    
    file_list = filter(lambda x: x[-4:] == '.pkl',file_list)
    train_files = []
    val_files= []
    for t,v in zip(tr,val):
        train_list = map(lambda x: (x[0],x[1]),t)
        val_list = map(lambda x: (x[0],x[1]),v)
        train_files.append(filter(lambda x: (int(re.findall('\d+',x)[1]),int(re.findall('\d+',x)[0])) in train_list,file_list))
        val_files.append(filter(lambda x: (int(re.findall('\d+',x)[1]),int(re.findall('\d+',x)[0])) in val_list,file_list))
    return train_files,val_files
        
def main(make_data_set = False, prediction_lead = '60',feature_set = 'ipw_refl',n_estimators = 300,file_name = 'RF_experiment_1.pkl'): 
    print 'Saving file to ' + '../output/RF_experiments/' + str(n_estimators) + file_name
    if make_data_set == 'False':
        make_data_set = False
    else:
        make_data_set = True
    data_builder = BuildDataSet.dataset(num_points = 500)
    training_blocks,validation_blocks = build_training_validation_sets(data_builder)
    
    cca_mdl = fit_CCA(training_blocks[0],data_builder)
    transform_predict_RF(cca_mdl,training_blocks[0],validation_blocks[0],data_builder)
    
#    train,validation = arrange_training_validation(training_blocks,validation_blocks)
#    performance = RF_classifier(train,validation,n_estimators,feature_set)
#    # calculate the averages here before you pkl
#    f1 = file('../output/RF_experiments/' + str(n_estimators) + file_name,'wb')
#    pkl.dump(performance,f1,protocol = pkl.HIGHEST_PROTOCOL)
#    f1.close()
    
if __name__ == '__main__':
    kwargs = {}
    main()
#    if len(sys.argv) < 2:
#        print 'Usage python RF_prediction_experiments.py [make_data_set] [prediction_lead] [feature_set] [file_name]'
#    else:
#        kwargs['make_data_set'] = sys.argv[1]
#        kwargs['prediction_lead'] = sys.argv[2]
#        kwargs['feature_set'] = sys.argv[3]
#        kwargs['n_estimators'] = int(sys.argv[4])
#        kwargs['file_name'] = sys.argv[5]
#        main(**kwargs)
    
