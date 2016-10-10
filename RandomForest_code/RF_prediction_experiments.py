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

def make_dataset_RF(data_builder):
    '''This function will make the data set for RF experiments by taking 
    the first order and second order statistics of the points around the 
    33 x 33 window of the predicted points. The data set will be saved to 
    .pkl files where each .pkl file will contain a months worth of data.
    This will make it easier for us to load the data in terms of the 
    training and validation set blocks.
    '''
    PixelPoints = data_builder.sample_random_pixels()
    print PixelPoints.shape
    storm_dates_all = {}
    for yr in [14,15,16]:
        storm_dates_all[yr] = data_builder.load_storm_days(yr)
        # load the dictionary which gives us the days which are 
        # clubbed together. 
        doy_strings = data_builder.club_days(storm_dates_all[yr])
        days_in_sorted = doy_strings.keys()
        days_in_sorted.sort()
        ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(yr)
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
                field_stats = data_builder.get_field_statistics(temp_array)
                field_stats = data_builder.get_temporal_statistics(field_stats)
                points_array.append(field_stats)
            print np.vstack(points_array).shape
            # Save each string of days to a pkl file
            temp_file = file('../data/TrainTest/20' + str(yr) + os.sep + 'IPW_refl_features{0}_{1}.pkl'.format(yr,set_),'wb')
            pkl.dump(np.vstack(points_array),temp_file,protocol = pkl.HIGHEST_PROTOCOL)
            temp_file.close()

def make_dataset_RF_30minPrediction(data_builder):
    '''This function will make the data set for RF experiments by taking 
    the first order and second order statistics of the points around the 
    33 x 33 window of the predicted points. The data set will be saved to 
    .pkl files where each .pkl file will contain a months worth of data.
    This will make it easier for us to load the data in terms of the 
    training and validation set blocks. 
    '''
    PixelPoints = data_builder.sample_random_pixels()
    print PixelPoints.shape
    storm_dates_all = {}
    for yr in [14,15,16]:
        storm_dates_all[yr] = data_builder.load_storm_days(yr)
        # load the dictionary which gives us the days which are 
        # clubbed together. 
        doy_strings = data_builder.club_days(storm_dates_all[yr])
        days_in_sorted = doy_strings.keys()
        days_in_sorted.sort()
        ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(yr)
        temp_data_set = {}
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
                field_stats = data_builder.get_field_statistics_30minPrediction(temp_array)
                field_stats = data_builder.get_temporal_statistics(field_stats)
                points_array.append(field_stats)
            temp_data_set[str(yr) + '_' + set_] = np.vstack(points_array)
            print np.vstack(points_array).shape
            # Save each string of days to a pkl file
            temp_file = file('../data/TrainTest/20' + str(yr) + os.sep + 'IPW_refl_features{0}_{1}.pkl'.format(yr,set_),'wb')
            pkl.dump(np.vstack(points_array),temp_file,protocol = pkl.HIGHEST_PROTOCOL)
            temp_file.close()

#def check_for_TrainTest():
    
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

def RF_classifier(train_files,validation_files,n_estimators = 100,max_depth = 6,feature_set = 'ipw_refl',max_features = None):
    '''Apply the Random forest classifier to each of the 7 cross validation blocks
    and test with the corresponding month as the validation set'''
    print 'Running Experiment with the following parameters'
    print '-'*50
    print 'Features to use: ' + feature_set
    print 'Number of estimators %d'%n_estimators
    yr_mon = {14: [5,6,7,8],
              15: [5,6],
              16: [5,6,7]}
    blocks = [str(yr) + '_' + str(mon) for yr in [14,15,16] for mon in yr_mon[yr]]
    ctr=0
    # loop thru each training and validation block
    
    models = []
    for tr,val in zip(train_files[:4],validation_files[:4]):
        print '-'*50
        print 'Validation year = %s Month = %s'%tuple(blocks[ctr].split('_'))
        ctr+=1
        # loop thru each file in the training set
        train = []
        for t in tr:
            file_year = re.findall('\d+',t)[0]
            base_path = '../data/TrainTest/20' + file_year + os.sep 
            train.append(np.load(base_path + t))
        train = np.vstack(train)
        validation = []
        for v in val:
            file_year = re.findall('\d+',v)[0]
            base_path = '../data/TrainTest/20' + file_year + os.sep 
            validation.append(np.load(base_path + v))
        validation = np.vstack(validation)
        print 'Size of training set is: '
        if feature_set == 'ipw_refl':
            print 'Processing IPW + reflectivity features'
            feature_slice = slice(-1)
            X_train = train[:,feature_slice]
            y_train = train[:,-1]
            X_val = validation[:,feature_slice]
            y_val = validation[:,-1]
            
        else:
            print 'Processing reflectivity features only'
            feature_slice = slice(5,10)
            X_train = np.concatenate((train[:,feature_slice],train[:,18:26]),axis = 1)
            y_train = train[:,-1]
            X_val = np.concatenate((validation[:,feature_slice],validation[:,18:26]),axis=1)
            y_val = validation[:,-1]
        print X_train.shape,y_train.shape
        print 'Fit RF for max_depth %d '%max_depth
        print 'Max number of features: ' + str(max_features)
        mdl = RandomForestClassifier(n_estimators = n_estimators,n_jobs = -1,max_features = max_features,max_depth = int(max_depth),verbose = 1)
        mdl.fit(X_train,y_train)
        Y_train_score = mdl.predict_proba(X_train)
        Y_train_hat = mdl.predict(X_train)
        # loop thru each file in the validation set
        Y_hat = mdl.predict(X_val)
        Y_score = mdl.predict_proba(X_val)
        print Y_train_hat,y_train,Y_train_score[:,1]
        performance_train = ModelMetrics.NOWCAST_performance((Y_train_hat,y_train,Y_train_score[:,1]))
        performance_validation = ModelMetrics.NOWCAST_performance((Y_hat,y_val,Y_score[:,1]))
        
        print 'Precision = %.2f'%performance_validation.p_score
        print 'Recall = %.2f'%performance_validation.r_score
        print 'F1 score = %.2f'%performance_validation.f1
        print 'Area under the curve = %.2f'%performance_validation.average_precision
        
        print 'POD = %.2f'%performance_validation.POD
        print 'FAR = %.2f'%performance_validation.FAR
        print 'CSI = %.2f'%performance_validation.CSI
        performance_validation.mdl = mdl
        models.append((performance_validation,performance_train))
        
    return models
        
def main(make_data_set = False, prediction_lead = '30',feature_set = 'ipw_refl',n_estimators = 300,file_name = 'RF_experiment_1.pkl',max_depth = 6,max_features = None): 
    print 'Saving file to ' + '../output/RF_experiments/' + str(n_estimators) + file_name + str(max_features)  + '_max_depth' + str(max_depth) + '.pkl'
    if make_data_set == 'False':
        make_data_set = False
    else:
        make_data_set = True
    
    if max_features == 'None':
        print 'Setting max_features to None!'
        max_features = None
    data_builder = BuildDataSet.dataset(num_points = 500)
    training_blocks,validation_blocks = build_training_validation_sets(data_builder)
    if make_data_set:
        if prediction_lead == '60':
            print 'Building data set for 60 minute prediction'
            make_dataset_RF(data_builder)
        else:
            print 'Building data set for 30 minute prediction'
            make_dataset_RF_30minPrediction(data_builder)
    train,validation = arrange_training_validation(training_blocks,validation_blocks)
    performance = RF_classifier(train,validation,n_estimators,max_depth,feature_set,max_features)
    # calculate the averages here before you pkl
    f1 = file('../output/RF_experiments/' + str(n_estimators) + file_name + str(max_features) + '_max_depth' + str(max_depth) + '.pkl','wb')
    pkl.dump(performance,f1,protocol = pkl.HIGHEST_PROTOCOL)
    f1.close()

if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) < 2:
        print 'Usage python RF_prediction_experiments.py [make_data_set] [prediction_lead] [feature_set] [file_name]'
    else:
        kwargs['make_data_set'] = sys.argv[1]
        kwargs['prediction_lead'] = sys.argv[2]
        kwargs['feature_set'] = sys.argv[3]
        kwargs['n_estimators'] = int(sys.argv[4])
        kwargs['file_name'] = sys.argv[5]
        kwargs['max_depth'] = int(sys.argv[6])
        kwargs['max_features'] = sys.argv[7]
        main(**kwargs)
    
