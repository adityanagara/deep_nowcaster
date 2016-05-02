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

# Load modules from sklearn package
from sklearn.ensemble import RandomForestClassifier


def get_val_doys(storm_dates):
    '''May: 121-151, June: 152 - 181 July: 182 - 212 August: 213 - 243'''
    yr_mon = {14: [5,6,7,8],
              15: [5,6,7,8]}
    blocks = [str(yr) + '_' + str(mon) for yr in yr_mon.keys() for mon in yr_mon[yr]]
    val_blocks = []
    train_blocks = []
    for bl in blocks[:-1]:
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
    for yr in [14,15]:
        storm_dates_all[yr] = data_builder.load_storm_days(yr)    
    storm_dates_all = np.concatenate((storm_dates_all[14],storm_dates_all[15]))
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
    storm_dates_all = {}
    for yr in [14,15]:
        storm_dates_all[yr] = data_builder.load_storm_days(yr)
        # load the dictionary which gives us the days which are 
        # clubbed together. 
        doy_strings = data_builder.club_days(storm_dates_all[yr])
        days_in_sorted = doy_strings.keys()
        days_in_sorted.sort()
        
        ipw_files,refl_files = data_builder.sort_IPW_refl_files(yr)
        PixelPoints = data_builder.sample_random_pixels()
        print PixelPoints.shape
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
                temp_array = data_builder.build_features_and_truth(temp_ipw_files,temp_refl_files,x_,y_)
                field_stats = data_builder.get_field_statistics(temp_array)
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
    for yr in [14,15]:
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

def RF_classifier(train_files,validation_files):
    '''Apply the Random forest classifier to each of the 7 cross validation blocks
    and test with the corresponding month as the validation set'''
    yr_mon = {14: [5,6,7,8],
              15: [5,6,7,8]}
    blocks = [str(yr) + '_' + str(mon) for yr in yr_mon.keys() for mon in yr_mon[yr]]
    ctr=0
    # loop thru each training and validation block
    mdl = RandomForestClassifier(n_estimators = 500,n_jobs = -1,class_weight = 'auto',max_features = 'auto')
    for tr,val in zip(train_files,validation_files):
        print '-'*50
        print 'Validation year = %s Month = %s'%tuple(blocks[ctr].split('_'))
        ctr+=1
        # loop thru each file in the training set
        train = []
        for t in tr:
            if re.findall('\d+',t)[0] == '14':
                base_path = '../data/TrainTest/2014/'
            else:
                base_path = '../data/TrainTest/2015/'
            train.append(np.load(base_path + t))
        train = np.vstack(train)
        mdl.fit(train[:,:16],train[:,-1])
        # loop thru each file in the validation set
        validation = []
        for v in val:
            if re.findall('\d+',v)[0] == '14':
                base_path = '../data/TrainTest/2014/'
            else:
                base_path = '../data/TrainTest/2015/'
            validation.append(np.load(base_path + v))
        validation = np.vstack(validation)
        Y_hat = mdl.predict(validation[:,:16])
        Y_score = mdl.predict_proba(validation[:,:16]) 
        
        performance = ModelMetrics.NOWCAST_performance((Y_hat,validation[:,-1],Y_score[:,1]))
        
        print 'Precision = %.2f'%performance.p_score
        print 'Recall = %.2f'%performance.r_score
        print 'F1 score = %.2f'%performance.f1
        print 'Area under the curve = %.2f'%performance.average_precision
        
        print 'POD = %.2f'%performance.POD
        print 'FAR = %.2f'%performance.FAR
        print 'CSI = %.2f'%performance.CSI
        
    return performance
        
def main(make_data_set = False): 
    data_builder = BuildDataSet.dataset(num_points = 1000)
    training_blocks,validation_blocks = build_training_validation_sets(data_builder)
    if make_data_set:    
        make_dataset_RF(data_builder)
    train,validation = arrange_training_validation(training_blocks,validation_blocks)
    performance = RF_classifier(train,validation)
    f1 = file('../output/RF_experiment2.pkl','wb')
    pkl.dump(performance,f1,protocol = pkl.HIGHEST_PROTOCOL)
    f1.close()
    
if __name__ == '__main__':
    main()
