# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:58:54 2015

@author: adityanagarajan
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import time
import re

TrainTestdir = 'data/TrainTest'

files = os.listdir(TrainTestdir)


IPWfiles = filter(lambda x: x[:7] == 'IPWdata' ,files)

Radarfiles = filter(lambda x: x[:9] == 'RadarRefl',files)

#Sort files DOY.hh
IPWfiles.sort(key = lambda x: float(x[7:10]) + float(x[x.index('_') + 1: x.index('.')]) * 0.01)

Radarfiles.sort(key = lambda x: float(x[9:12]) + float(x[x.index('_') + 1: x.index('.')])* 0.01)



# Pull out june data for now
IPWfiles = filter(lambda x: int(x[7:10]) < 152 or int(x[7:10]) > 181,IPWfiles)

Radarfiles = filter(lambda x: int(x[9:12]) < 152 or int(x[9:12]) > 181,Radarfiles)

# Pull out 205 >> 07/24
IPWfiles = filter(lambda x: int(x[7:10]) != 205, IPWfiles)

Radarfiles = filter(lambda x: int(x[9:12]) != 205,Radarfiles)

gridX = np.arange(-150.0,151.0,300.0/(100-1))
gridY = np.arange(-150.0,151.0,300.0/(100-1))

for gps,nexrad in zip(IPWfiles,Radarfiles):
    gridIPW = np.load('data/TrainTest/' + gps)
    gridZ = np.load('data/TrainTest/' + nexrad)
    gridZ[gridZ < 30.0] = np.nan
    gridZ = np.ma.array(gridZ, mask=np.isnan(gridZ))
    
    plt.figure()
    plt.pcolor(gridX,gridY,gridIPW,cmap='jet', vmin=-3.0, vmax=3.0)
    
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))
    
    plt.pcolor(gridX,gridY,gridZ,cmap='jet', vmin=0.0, vmax=60.0)
    
    plt.title(gps)
    print gps
    print nexrad
    
    plt.savefig('output/FileVerification/'  + 'file' + re.findall('\d+',gps)[0] + '_' + re.findall('\d+',gps)[1] + '.png' )
    

    
    

#gridIPW = np.load('data/TrainTest/IPWdata129_0.npy')
#

#
#plt.grid()
#plt.xlim((-150.0,150.0))
#plt.ylim((-150.0,150.0))
#
#
#plt.pcolor(gridX,gridY,gridIPW,cmap='jet', vmin=-3.0, vmax=3.0)


