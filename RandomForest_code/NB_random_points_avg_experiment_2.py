# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 14:24:39 2015

@author: adityanagarajan
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
import cPickle

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import nowcast

random_points_file = file('data/TrainTest/RandomPoints/ipw_refl_avg_field_random_points.pkl','rb')
data_set = cPickle.load(random_points_file)
random_points_file.close()

data_set = np.concatenate((data_set))

print data_set.shape

np.random.seed(12345)
number_of_rain = data_set[data_set[:,-1] == 1.0].shape[0]

number_of_no_rain = data_set[data_set[:,-1] == 0.0].shape[0]

print number_of_rain, number_of_no_rain

kf = cross_validation.KFold(data_set.shape[0],n_folds = 7, shuffle = True,random_state = 12345)


kf_dict = {}

kf_num = 0

num_trees = [4,6,7,8,9,10]

for train_index,test_index in kf:
    X_train, X_test = data_set[train_index,:8],data_set[test_index,:8]
    Y_train, Y_test = data_set[train_index,-1],data_set[test_index,-1]

#    mdl = RandomForestClassifier(n_estimators = 200,n_jobs = -1,class_weight = 'auto')
    mdl = GaussianNB()
        
    mdl.fit(X_train,Y_train)
    Y_hat = mdl.predict(X_test)
    Y_hat_prob = mdl.predict_proba(X_test)
    p_score = precision_score(Y_test,Y_hat)
    r_score = recall_score(Y_test,Y_hat)
    f1 = f1_score(Y_test,Y_hat)
    print f1    
    kf_dict[kf_num] = [(Y_train[Y_train == 1.0].shape[0],Y_test[Y_test == 1.0].shape[0]),(Y_hat,Y_hat_prob,Y_test),round(p_score,2),round(r_score,2),round(f1,2),mdl]
    kf_num+=1



model_metrics_file = file('model_metrics/NB_ipw_refl_random_points_avg.pkl','wb')

cPickle.dump(kf_dict,model_metrics_file,protocol=cPickle.HIGHEST_PROTOCOL)

model_metrics_file.close()

print 'Done!'











