# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:20:07 2015

@author: adityanagarajan
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import re


base_dir = 'data/TrainTest/Features/'
file_list = os.listdir(base_dir)


file_list = filter(lambda x: x[-4:] == '.npy',file_list)

print len(file_list)

PixelX = range(33,50)
PixelY = range(50,66)

PixelPoints = [(x,y) for x in PixelX for y in PixelY]

PixelPoints = np.array(PixelPoints)

gridX = np.arange(-150.0,151.0,300.0/(100-1))
gridY = np.arange(-150.0,151.0,300.0/(100-1))

# Filter top left of the domain
domain_list = filter(lambda x: int(re.findall('\d+',x)[0]) in np.unique(PixelPoints[:,0]) and int(re.findall('\d+',x)[1]) in np.unique(PixelPoints[:,1]),file_list)

print len(domain_list)

domain_list.sort(key = lambda x: int(x[-7:-4]))

ipw_files = filter(lambda x: x[:3] == 'IPW',domain_list)

radar_files = filter(lambda x: x[:5] == 'Radar',domain_list)

print len(ipw_files) + len(radar_files)

temp_file_list_ipw = filter(lambda x: x[-7:-4] == '129',ipw_files)

temp_file_list_refl = filter(lambda x: x[-7:-4] == '129',radar_files)



#Load IPW array for the domain
IPWFeatures = np.concatenate(map(lambda x: np.load(base_dir + x),temp_file_list_ipw))

print 'IPW done!'
#Load refl values for the domain
ReflFeatures = np.concatenate(map(lambda x: np.load(base_dir + x),temp_file_list_refl))

print 'Refl done!'

data = np.hstack((IPWFeatures[:,:-1],ReflFeatures,IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)))


print 'Done!'

print data.shape



x_train = data[~np.any(np.isnan(data),axis = 1),:]


from sklearn.ensemble import RandomForestClassifier

mdl = RandomForestClassifier(n_estimators = 100,n_jobs = -1,verbose=1)

print 'Training Model!'
clf = mdl.fit(x_train[:,:-1],x_train[:,-1])

#%%
#Generate test points
PixelX = range(50,66)
PixelY = range(50,66)

PixelPoints = [(x,y) for x in PixelX for y in PixelY]

PixelPoints = np.array(PixelPoints)

domain_list = filter(lambda x: int(re.findall('\d+',x)[0]) in np.unique(PixelPoints[:,0]) and int(re.findall('\d+',x)[1]) in np.unique(PixelPoints[:,1]),file_list)

domain_list.sort(key = lambda x: int(x[-7:-4]))

ipw_files_test = filter(lambda x: x[:3] == 'IPW',domain_list)

radar_files_test = filter(lambda x: x[:5] == 'Radar',domain_list)

temp_file_list_ipw_test = filter(lambda x: x[-7:-4] == '129',ipw_files_test)

temp_file_list_refl_test = filter(lambda x: x[-7:-4] == '129',radar_files_test)

#Load IPW array for the domain
IPWFeatures_test = np.concatenate(map(lambda x: np.load(base_dir + x),temp_file_list_ipw_test))

print 'IPW done!'
#Load refl values for the domain
ReflFeatures_test = np.concatenate(map(lambda x: np.load(base_dir + x),temp_file_list_refl_test))

data_test = np.hstack((IPWFeatures_test[:,:-1],ReflFeatures_test,IPWFeatures_test[:,-1].reshape(IPWFeatures_test.shape[0],1)))

x_test = data_test[~np.any(np.isnan(data_test),axis = 1),:]
print x_test.shape
print 'Done!'

# Uncomment to plot the domain
#%%
y_hat = clf.predict(x_test[:,:-1])

y_hat_prob = clf.predict_proba(x_test[:,:-1])


#%%
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

#precision, recall, thresholds = precision_recall_curve(
#...     y_true, y_scores)
precision, recall, thresholds = precision_recall_curve(x_test[:,-1],y_hat_prob[:,-1])

print precision, recall, thresholds

print f1_score(x_test[:,-1],y_hat)


#%%
PixelX = range(50,66)
PixelY = range(50,66)

PixelPoints = [(x,y) for x in PixelX for y in PixelY]

PixelPoints = np.array(PixelPoints)

gridX = np.arange(-150.0,151.0,300.0/(100-1))
gridY = np.arange(-150.0,151.0,300.0/(100-1))

#Plot domain
for p in PixelPoints:
    plt.plot(gridX[p[0]],gridY[p[1]],'r*')

plt.xlabel('Easting')
plt.ylabel('Northing')
plt.title('Experimental subdomain consisting of 33x33 grid')

plt.grid()
plt.xlim((-150.0,150.0))
plt.ylim((-150.0,150.0))
#%%


    



