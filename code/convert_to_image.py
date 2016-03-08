# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 17:00:20 2016

@author: adityanagarajan
This script converts the ipw and reflectivit fields from floating point to 
8 bit int 
"""
import numpy as np
import BuildDataSet
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import DFWnet

file_path = 'data/TrainTest/'

data_builder = BuildDataSet.dataset(yr = '2014')
network = DFWnet.CommonData()

map_ipw_array = np.linspace(-5,5,256)
map_refl_array = np.linspace(0,90,256)


#Trust me and uncomment this code for converting IPW to image
#for f in data_builder.IPWfiles:
#    arr = np.load(file_path + f)
#    new_array = np.zeros((100,100),dtype='uint8')
#    for i in range(arr.shape[0]):
#        for j in range(arr.shape[1]):
#            new_array[i,j] = np.argmin(np.abs(arr[i,j] - map_ipw_array))
#    np.save(file_path + f.split('.')[0] + 'img.npy',new_array)
#
#
#for r in data_builder.Radarfiles:
#    arr = np.load(file_path + r)
#    arr[np.isnan(arr)] = 0.0
#    arr[arr<0.0] = 0.0
#    new_array_refl = np.zeros((100,100),dtype='uint8')
#    for i in range(arr.shape[0]):
#        for j in range(arr.shape[1]):
#            new_array_refl[i,j] = np.argmin(np.abs(arr[i,j] - map_refl_array))
#    np.save(file_path + r.split('.')[0] + 'img.npy',new_array_refl)
#
#print 'Done!'

file_name = 'RadarRefl14128_0img.npy'
arr = np.load(file_path + file_name)
plt.figure()
plt.imshow(arr, cmap = cm.Greys_r)
gridX = np.arange(-150.0,151.0,300.0/(100-1))
gridY = np.arange(-150.0,151.0,300.0/(100-1))
plt.title(file_name)

xMesh,yMesh = np.meshgrid(gridX,gridY)

arr2 = np.load(file_path + 'RadarRefl14128_0.npy')
arr2[arr2 < 10.0] = np.nan
arr2 = np.ma.array(arr2,mask=np.isnan(arr2))
plt.figure()
plt.pcolor(gridX,gridY,arr2,cmap='jet', vmin=0, vmax=60)
plt.colorbar()
plt.grid()
plt.xlim((-150.0,150.0))
plt.ylim((-150.0,150.0))
plt.title(file_name)

'''
   gridZ = gridZ.T
    np.save('data/TrainTest/RadarRefl' + str(yr) + str(doy) + '_' + str(t) +  '.npy',gridZ)
    
    #Plot values greater than 30 dbZ
    gridZ[gridZ < 30.0] = np.nan
    
    
    
    
    gridIPW = gridIPWfields(t,IPWvals,W,doy)
    
    # Mask all values with nan
    gridZ = np.ma.array(gridZ, mask=np.isnan(gridZ))
    plt.figure()
    
    plt.pcolor(gridX,gridY,gridIPW,cmap='gist_ncar', vmin=-3.0, vmax=3.0)

    np.save('data/TrainTest/IPWdata' +str(yr) + str(doy) + '_' + str(t) + '.npy',gridIPW)
#    cbar = plt.colorbar()
#    cbar.set_label('Normalized IPW vals')
    
    plt.pcolor(gridX,gridY,gridZ,cmap='jet', vmin=10, vmax=60)
'''






    





