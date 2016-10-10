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

# Merge the individual files
y_hats_1 = np.load('y_hat_2_225_.npy')
y_hats_2 = np.load('y_hat_2_450_.npy')
y_hats_3 = np.load('y_hat_2_625_.npy')
y_hats_4 = np.load('y_hat_2_900_.npy')
y_hats = np.concatenate((y_hats_1[:225,:,:],y_hats_2[225:450,:,:],y_hats_3[450:625,:,:],y_hats_4[625:,:,:]))

y_test = OutputMatrix[:,960:]

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




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from sklearn.metrics import roc_curve, auc

roc_auc_list = []
for point in range(final_y_hat.shape[0]):
    fpr,tpr,thrsh = roc_curve(final_y_test[point,:],final_y_hat[point,:,-1])
    if np.where(np.isnan(fpr))[0].size > 0 or np.where(np.isnan(tpr))[0].size > 0:
        print 'No rain at all in this pixel (%d) '%point
        roc_auc = np.nan
        
    else:
        roc_auc = auc(fpr,tpr)
    
    roc_auc_list.append(roc_auc)
    
    print roc_auc

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

for train_index, test_index in kf:
    ctr = 0
#    print("TRAIN:", train_index.shape, "TEST:", test_index.shape)

    for px in range(OutputMatrix.shape[0]):
        x = FeatureMatrix[:,:]
    
        y = OutputMatrix[3,:]
        
        x_train,x_test = x[train_index],x[test_index]
        y_train,y_test = y[train_index],y[test_index]
        
        print x_train.shape,x_test.shape
        print y_train.shape,y_test.shape
        
        mdl = RandomForestClassifier(n_estimators = 100,max_features = 5000,n_jobs=-1)
        
        mdl.fit(x_train,y_train)
        
        y_cap = mdl.predict_proba(x_test)
        
        FeatureImportance.append(mdl.feature_importances_)
        
        fpr,tpr,thrsh = roc_curve(y_test,y_cap[:,1])
        if np.where(np.isnan(fpr))[0].size > 0 or np.where(np.isnan(tpr))[0].size > 0:
            print 'No rain at all in this pixel (%d,%d) '%(PixelPoints[px,0],PixelPoints[px,1])
            
            roc_auc = np.nan
        else:
            roc_auc = auc(fpr,tpr)
            print 'Train Test Split Number: %d %d) Area Under the curve for pixel point %d,%d = %f'%(ctr1,ctr,PixelPoints[px,0],PixelPoints[px,1],roc_auc)
        rocScores[ctr1,ctr] = roc_auc
        ctr+=1
        
    ctr1+=1

'''













