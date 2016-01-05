# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:20:52 2015

@author: adityanagarajan
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

import cPickle
import BuildDataSet

import nowcast

average_fields = []
indices_i = [(x*1089,(x+1)*1089)for x in range(4) ]
indices_r = [(x*1089,(x+1)*1089)for x in range(4,8) ]

data_builder = BuildDataSet.dataset()

'''
>>> f = file('obj.save', 'rb')
>>> loaded_obj = cPickle.load(f)
>>> f.close()
'''
base_file = 'data/TrainTest/RandomPoints/'

file_list = os.listdir(base_file)
file_list = filter(lambda x: x[-4:] == '.pkl' and x[:3] == 'IPW',file_list)

print file_list

for pkl_file in file_list:
    print pkl_file
    
    f1 = file(base_file + pkl_file)
    random_points = cPickle.load(f1)
    f1.close()

    IPWFeatures = np.concatenate(map(lambda x: x[0].astype('float32'),random_points))

    print 'IPW done!'
        
    ReflFeatures = np.concatenate(map(lambda x: x[1].astype('float32'),random_points))

    print 'Refl done!'
    # 2178 -> use the fields from a whole time step back
    data = np.hstack((IPWFeatures[:,2178:-1],ReflFeatures[:,2178:],IPWFeatures[:,-1].reshape(IPWFeatures.shape[0],1)))

    data = data[~np.any(np.isnan(data),axis = 1),:]

    ipw_avg_fields = np.zeros((data.shape[0],9))


    ipw_avg_fields[:,-1] = data[:,-1]

    for ix in range(len(indices_i)):
        ipw_avg_fields[:,ix] = np.average(data[:,indices_i[ix][0]:indices_i[ix][1]],axis = 1)
    
        ipw_avg_fields[:,ix + 4] = np.average(data[:,indices_r[ix][0]:indices_r[ix][1]],axis = 1)

    average_fields.append(ipw_avg_fields)


out_file = file(base_file + 'ipw_refl_avg_field_random_points.pkl','wb')

cPickle.dump(average_fields,out_file,protocol=cPickle.HIGHEST_PROTOCOL)

out_file.close()

    
    

'''
        for ix in range(len(indices)):
            ipw_avg_fields[:,ix] = np.average(data[:,indices[ix][0]:indices[ix][1]],axis = 1)
            ipw_avg_fields[:,ix + 4] = np.average(data[:,indices_r[ix][0]:indices_r[ix][1]],axis = 1)
        
        average_fields.append(ipw_avg_fields)
        
        ctr+=1
        
'''




