# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:26:07 2016

@author: adityanagarajan
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fft

file_path = '../data/RadarData/Decimated/14128refl_decimated.npy'

arr = np.load(file_path)

m = 100

gridX = np.arange(-150.0,151.0,300.0/(m-1))
gridY = np.arange(-150.0,151.0,300.0/(m-1))

print arr.shape

for i in range(arr.shape[0]):
    gridZ = arr[i,...]
    gridZ[gridZ < 30.0] = np.nan
    # Mask values with nan
    gridZ = np.ma.array(gridZ, mask=np.isnan(gridZ))

    plt.figure()
    plt.pcolor(gridX,gridY,gridZ,cmap='jet', vmin=10, vmax=60)
    plt.grid()
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))



