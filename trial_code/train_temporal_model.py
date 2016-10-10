# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:11:04 2015

@author: adityanagarajan
"""

import numpy as np
from matplotlib import pyplot as plt
#import DFWnet
#import os


from sklearn.linear_model import LogisticRegression



from sklearn.ensemble import RandomForestClassifier

#from sklearn.cross_validation import KFold




#%%
PixelX = range(20,50)
PixelY = range(50,80)

PixelPoints = [(x,y) for x in PixelX for y in PixelY]


PixelPoints = np.array(PixelPoints)

temp_yhats = np.load('y_hat.npy')

doy_list = ['128', '129', '132', '133', '143', '144', '145', '146', '151', '196', '197', '198', '199', '209', '210', '211', '212', '223', '228', '229', '230', '231', '241']

OutputMatrix = np.load('data/TrainTest/OutputMatrix.npy')

y_test = OutputMatrix[:,960:]

#y_test = y_test[:,:]
indicies = [x*48 for x in range(1,4)]

new_temp_yhats = temp_yhats[:,:,:]

new_temp_yhats = np.around(new_temp_yhats,3)

doy_230_232_train_x = np.zeros((900,92,4))

doy_241_test_x = np.zeros((900,44,4))

for point in range(new_temp_yhats.shape[0]):
    
    temporal_list = []
    
    temp_vals = new_temp_yhats[point,:96,1]
    temp_vals = np.around(temp_vals,3)
    temp_out = y_test[point,:96]
    for idx in range(temp_vals.size):
        if idx+3 < 95:
            print temp_vals[idx:idx+3]
            print temp_vals[idx+4]
            doy_230_232_train_x[point,idx,:3] = temp_vals[idx:idx+3]
            doy_230_232_train_x[point,idx,3] = temp_out[idx+4]
    
    test_temp_vals = new_temp_yhats[point,96:,1]
    
    test_temp_out = y_test[point,96:]
    
    for idx2 in range(test_temp_vals.size):
        if idx2 + 3 < 47:
            print test_temp_vals[idx2:idx2+3]
            print test_temp_out[idx2+4]
            doy_241_test_x[point,idx2,:3] = test_temp_vals[idx2:idx2+3]
            doy_241_test_x[point,idx2,3] = test_temp_out[idx2+4]
            


for x,y in zip(np.where(doy_230_232_train_x[:,:,-1] == 1)[0],np.where(doy_230_232_train_x[:,:,-1] == 1)[1]):
    print x,y
    
    print doy_230_232_train_x[x,y,:]
    

    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Find the grid points which have atleast 1 sample of rainfall
y_test = OutputMatrix[:,960:]
y_test = y_test[:,:96]
ctr_pts = 0
points_list = []
for te in range(y_test.shape[0]):
    if np.where(y_test[te,:] == 1.0)[0].shape[0] > 2:
        print 'Use data point %d '%te
        points_list.append(te)
        ctr_pts+=1
print ctr_pts



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = RandomForestClassifier(n_jobs=-1)

temporal_y_hat = np.zeros((900,44,2))

for point in points_list:
    print 'Temporal model for pixel point (%d, %d)'%(PixelPoints[point,0],PixelPoints[point,1])
    model.fit(doy_230_232_train_x[point,:,:3],doy_230_232_train_x[point,:,-1])
    temporal_y_hat[point,:,:] = model.predict_proba(doy_241_test_x[point,:,:3])
    print point

print 'Done'   

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_test = OutputMatrix[:,960:]

final_y_hat = np.array(map(lambda x: temporal_y_hat[x,:,:],points_list))
final_y_test = np.array(map(lambda x: doy_241_test_x[x,:,-1],points_list))

from sklearn.metrics import roc_curve, auc





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%
model = RandomForestClassifier(n_jobs=-1)



#new_temp_yhats.shape[0]

for point in range(1,2):
    
    print 'Temporal model for pixel point (%d, %d)'%(PixelPoints[point,0],PixelPoints[point,1])
    
    model.fit(doy_230_232_train_x[point,:,:3],doy_230_232_train_x[point,:,-1])
    
    temporal_y_hat[point,:,:] = model.predict_proba(doy_241_test_x[point,:,:3])
    
    
    print point

#%%






#%%


from sklearn.tree import DecisionTreeClassifier

temporal_y_hat_2 = np.zeros((101,44,2))

#new_temp_yhats.shape[0]
model_2 = LogisticRegression()
for point in range(3,4):
    print 'Temporal model for pixel point (%d, %d)'%(PixelPoints[point,0],PixelPoints[point,1])
    
    model_2.fit(doy_230_232_train_x[point,:,:3],doy_230_232_train_x[point,:,-1])
    
    temporal_y_hat_2[point,:,:] = model_2.predict_proba(doy_241_test_x[point,:,:3])
    print temporal_y_hat_2[point,:,:]

#%%














