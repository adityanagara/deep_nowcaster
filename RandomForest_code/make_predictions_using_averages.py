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

f = file('ipw_refl_avg_field_averages_.pkl','rb')
data_set = cPickle.load(f)
f.close()

print len(data_set)

data_set = np.concatenate((data_set))

print data_set.shape

np.random.seed(12345)
number_of_rain = data_set[data_set[:,-1] == 1.0].shape[0]

number_of_no_rain = data_set[data_set[:,-1] == 0.0].shape[0]

print number_of_rain,number_of_no_rain


kf = cross_validation.KFold(data_set.shape[0],n_folds = 7, shuffle = True,random_state = 12345)

ipw_refl = 8

kf_dict = {}
kf_num = 0

for train_index,test_index in kf:
    X_train, X_test = data_set[train_index,:8],data_set[test_index,:8]
    Y_train, Y_test = data_set[train_index,-1],data_set[test_index,-1]
#    mdl = GaussianNB()

    mdl = RandomForestClassifier(n_estimators = 500,n_jobs = -1,max_features = 2,class_weight = 'auto')
#    mdl = SVC(C = 10.0,class_weight = 'auto')
    mdl.fit(X_train,Y_train)
    Y_hat = mdl.predict(X_test)
    Y_hat_prob = mdl.predict_proba(X_test)
    p_score = precision_score(Y_test,Y_hat)
    r_score = recall_score(Y_test,Y_hat)
    f1 = f1_score(Y_test,Y_hat)
    print f1    
    kf_dict[kf_num] = [(Y_train[Y_train == 1.0].shape[0],Y_test[Y_test == 1.0].shape[0]),(Y_hat,Y_hat_prob,Y_test),round(p_score,2),round(r_score,2),round(f1,2),mdl]
    kf_num+=1
    break

#f = file('model_metrics/GaussianNB_model_refl.pkl','wb')
#
#cPickle.dump(kf_dict,f,protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()
#
#print 'Done!'











