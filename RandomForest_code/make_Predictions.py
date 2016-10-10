# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:16:04 2015

@author: adityanagarajan
"""

import numpy as np
from matplotlib import pyplot as plt
import DFWnet
import os


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import KFold

PixelX = range(20,50)
PixelY = range(50,80)

PixelPoints = [(x,y) for x in PixelX for y in PixelY]


PixelPoints = np.array(PixelPoints)



FeatureMatrix = np.load('data/TrainTest/FeatureMatrix.npy')

OutputMatrix = np.load('data/TrainTest/OutputMatrix.npy')

kf = KFold(1104, n_folds=6)

DFW = DFWnet.CommonData()

DFW.doytodate(14,230)

print DFW.mon + '/' + DFW.day

x_train = FeatureMatrix[:960,:]
x_test = FeatureMatrix[960:,:]

y_train = OutputMatrix[:,:960]
y_test = OutputMatrix[:,960:]

print x_train.shape,x_test.shape
print y_train.shape,y_test.shape

mdl = LogisticRegression(penalty = 'l2',C = 100.0)

ctr = 0
y_hat = np.zeros((900,144,2))
for point in range(900):
    print ctr
    print 'Starting model for pixel point: (%d, %d)'%(PixelPoints[point,0],PixelPoints[point,1])
    
    mdl.fit(x_train,y_train[point,:])
    
    y_hat[point,:,:] = mdl.predict_proba(x_test)
    
    ctr+=1

print 'Done!'

np.save('y_hat.npy',y_hat)


#%%

temp_yhats = np.load('data/TrainTest/temp_y_hats.npy')

doy_list = ['128', '129', '132', '133', '143', '144', '145', '146', '151', '196', '197', '198', '199', '209', '210', '211', '212', '223', '228', '229', '230', '231', '241']



indicies = [x*48 for x in range(1,4)]

new_temp_yhats = temp_yhats[:101,:,:]

new_temp_yhats = new_temp_yhats[1,:96,1]

ctr_1 = 0

new_temp_yhats = np.around(new_temp_yhats, decimals=4)


temporal_list = []

for idx in range(new_temp_yhats.size):
    
    if idx+4 < 95:
        print new_temp_yhats[idx:idx+4]
        temporal_list.append(new_temp_yhats[idx:idx+4])








#%%


#%%


X = np.array(temporal_list)

y_test_temp = y_test[1,5:96]

model = RandomForestClassifier(n_jobs=-1)

#model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)

model.fit(X,y_test_temp)

rain_states = model.predict(X)

print rain_states

print y_test[1,:96]



#%%  