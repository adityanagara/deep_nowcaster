# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:54:03 2015

@author: adityanagarajan
"""

import numpy as np
from matplotlib import pyplot as plt
import DFWnet
import os
import sys


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier


'''
This model is trained on rest of the days and tested on May8, May9, May12
'''

PixelX = range(20,50)
PixelY = range(50,80)

PixelPoints = [(x,y) for x in PixelX for y in PixelY]


PixelPoints = np.array(PixelPoints)

FeatureMatrix = np.load('data/TrainTest/FeatureMatrix.npy')

OutputMatrix = np.load('data/TrainTest/OutputMatrix.npy')



DFW = DFWnet.CommonData()

DFW.doytodate(14,230)

print DFW.mon + '/' + DFW.day

x_train = FeatureMatrix[144:,:]
x_test = FeatureMatrix[:144,:]

y_train = OutputMatrix[:,144:]
y_test = OutputMatrix[:,:144]

print x_train.shape,x_test.shape
print y_train.shape,y_test.shape

#mdl = LogisticRegression(penalty = 'l2',C = 100.0)
mdl = RandomForestClassifier(n_estimators = 100,max_features = 5000,n_jobs=-1)


ctr = 0

start_range = sys.argv[1]
end_range = sys.argv[2]

print start_range,end_range

y_hat = np.zeros((900,144,2))
for point in range(int(start_range),int(end_range)):
    print ctr
    print 'Starting model for pixel point: (%d, %d)'%(PixelPoints[point,0],PixelPoints[point,1])
    
    mdl.fit(x_train,y_train[point,:])
    
    y_hat[point,:,:] = mdl.predict_proba(x_test)
    
    ctr+=1

print 'Done!'

np.save('y_hat_Random_Forest_' + str(end_range) + '_.npy',y_hat)


##%%
#
## Merge the individual files
#y_hats_1 = np.load('y_hat_2_225_.npy')
#y_hats_2 = np.load('y_hat_2_450_.npy')
#y_hats_3 = np.load('y_hat_2_625_.npy')
#y_hats_4 = np.load('y_hat_2_900_.npy')
#y_hats = np.concatenate((y_hats_1[:225,:,:],y_hats_2[225:450,:,:],y_hats_3[450:625,:,:],y_hats_4[625:,:,:]))
#
#
##%%
## Evaluate spatial model with ROC


