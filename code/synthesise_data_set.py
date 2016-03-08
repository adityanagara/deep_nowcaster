# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:23:41 2015

@author: adityanagarajan
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import re


PixelX = range(33,66)
PixelY = range(33,66)

gridX = np.arange(-150.0,151.0,300.0/(100-1))
gridY = np.arange(-150.0,151.0,300.0/(100-1))

# Find the grid points of interest 33x33 = 1089
PixelPoints = [(x,y) for x in PixelX for y in PixelY]


PixelPoints = np.array(PixelPoints)

# These points have been verified


verify_ipw = np.load('data/TrainTest/IPWdata128_45.npy')
verify_radar = np.load('data/TrainTest/RadarRefl128_45.npy')
verify_output = verify_radar.copy()

Thrashold = 24.0

verify_output[np.isnan(verify_output)] = 0.0


verify_output[verify_output < Thrashold] = 0.0
verify_output[verify_output >= Thrashold] = 1.0


verify_radar[np.isnan(verify_radar)] = 0.0

gridIPW = np.zeros((33,33))
gridZ = np.zeros((33,33))
out_point = np.zeros((100,100))

gridIPW[:] = np.nan
gridZ[:] = np.nan

#inputFieldRangeX = range(x0 - 16, x0 + 17)
#
#inputFieldRangeY = range(y0 + 17,y0 - 16,-1)


#i_start = 33
#i_end = 66
#j_start = 33
#j_end = 66
point_num = 859
x_ = PixelPoints[point_num][0]
y_ = PixelPoints[point_num][1]

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




















