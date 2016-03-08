# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:23:41 2015

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

domain_list = filter(lambda x: int(re.findall('\d+',x)[0]) in np.unique(PixelPoints[:,0]) and int(re.findall('\d+',x)[1]) in np.unique(PixelPoints[:,1]),file_list)

print len(domain_list)

domain_list.sort(key = lambda x: int(x[-7:-4]))

ipw_files = filter(lambda x: x[:3] == 'IPW',domain_list)

radar_files = filter(lambda x: x[:5] == 'Radar',domain_list)

print len(ipw_files) + len(radar_files)

file_number = 210
print ipw_files[file_number]
temp_ipw_file = ipw_files[file_number]

temp_ipw_array = np.load(base_dir + temp_ipw_file)

temp_radar_array = np.load(base_dir + radar_files[file_number])

verify_ipw = np.load('data/TrainTest/IPWdata128_32.npy')
verify_radar = np.load('data/TrainTest/RadarRefl128_32.npy')
verify_output = verify_radar.copy()

x_ = int(re.findall('\d+',temp_ipw_file)[0])
y_ = int(re.findall('\d+',temp_ipw_file)[1])

gridIPW = np.zeros((33,33))
gridZ = np.zeros((33,33))
out_point = np.zeros((100,100))

#gridIPW = verify_ipw.copy()
#[0, 1089, 2178, 3267, 4356, 5445]

gridIPW[:] = np.nan
gridZ[:] = np.nan

Thrashold = 24.0

verify_output[np.isnan(verify_output)] = 0.0


verify_output[verify_output < Thrashold] = 0.0
verify_output[verify_output >= Thrashold] = 1.0


verify_radar[np.isnan(verify_radar)] = 0.0
#verify_radar = np.ma.array(verify_radar, mask=np.isnan(verify_radar))

i_start = x_ -16
i_end = x_ + 17
j_start = y_ -16
j_end = y_ + 17


gridIPW[:] = verify_ipw[j_start:j_end,i_start:i_end]
gridZ[:] = verify_radar[j_start:j_end,i_start:i_end]
out_point[y_,x_] = verify_output[y_,x_]

print verify_output[y_,x_]
#print out_point[y_,x_]
x_range_start = gridX[x_] - 16.0*(300.0/99.0)
y_range_start = gridY[y_] - 16.0*(300.0/99.0)

x_range_end = gridX[x_] + 17.0*(300.0/99.0)
y_range_end = gridY[y_] + 17.0*(300.0/99.0)

gridX_ = np.arange(x_range_start,x_range_end,300./99.)
gridY_ = np.arange(y_range_start,y_range_end,300./99.)

#pointXY = RadarMatrix[x_,y_]
#gridIPW = np.ma.array(gridIPW, mask=np.isnan(gridIPW))


plt.figure()
plt.subplot(2,2,1)
plt.pcolor(gridX_,gridY_,gridIPW,cmap='jet', vmin=-3.0, vmax=3.0)
plt.plot(gridX[x_],gridY[y_],'r*')
plt.grid()
#plt.xlim((x_range_start,x_range_end))
#plt.ylim((y_range_start,y_range_end))

plt.xlim((-150.0,150.0))
plt.ylim((-150.0,150.0))

plt.subplot(2,2,2)

plt.pcolor(gridX,gridY,verify_ipw,cmap='jet', vmin=-3.0, vmax=3.0)
plt.plot(gridX[x_],gridY[y_],'r*')
plt.grid()
plt.xlim((-150.0,150.0))
plt.ylim((-150.0,150.0))

plt.subplot(2,2,3)
plt.pcolor(gridX_,gridY_,gridZ,cmap='jet', vmin=10.0, vmax=60.0)
plt.plot(gridX[x_],gridY[y_],'r*')
plt.grid()
plt.xlim((-150.0,150.0))
plt.ylim((-150.0,150.0))


plt.subplot(2,2,4)
plt.pcolor(gridX,gridY,verify_radar,cmap='jet', vmin=10.0, vmax=60.0)
plt.plot(gridX[x_],gridY[y_],'r*')
plt.grid()
plt.xlim((-150.0,150.0))
plt.ylim((-150.0,150.0))

plt.figure()
plt.subplot(1,2,1)
plt.pcolor(gridX,gridY,out_point,cmap='jet', vmin=0.0, vmax=1.0)
#plt.plot(gridX[x_],gridY[y_],'g*')
plt.grid()
plt.xlim((-150.0,150.0))
plt.ylim((-150.0,150.0))


plt.subplot(1,2,2)
plt.pcolor(gridX,gridY,verify_output,cmap='jet', vmin=0.0, vmax=1.0)
plt.plot(gridX[x_],gridY[y_],'g*')
plt.grid()

plt.xlim((-150.0,150.0))
plt.ylim((-150.0,150.0))


    










