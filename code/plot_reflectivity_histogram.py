# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:45:40 2016

@author: adityanagarajan
purpose: This code generates the histogram of reflectivity for the storm cases 
in 2014
"""

import BuildDataSet
import numpy as np
from matplotlib import pyplot as plt


data_builder = BuildDataSet.dataset()

file_list =  data_builder.Radarfiles



bins = [-30.,-20.,-10.,0.,10.,20.,30.,40.,50.,60.]

refl_hist = []

for f1 in file_list:
    RadarMatrix = np.load(data_builder.TrainTestdir + f1)
    refl_histogram = np.histogram(RadarMatrix,bins)
    refl_hist.append(refl_histogram[0])
    print refl_histogram

refl_hist_array = np.array(refl_hist)

width = 0.5
x_axis = np.arange(9)
plt.figure()
refl_sum = np.sum(refl_hist_array, axis=0)
hist,edge_slice = (refl_sum,np.array(bins))
plt.bar(x_axis,hist,width,align='center')
x_ticks = [str(bins[x]) +' : ' + str(bins[x + 1]) for x in range(len(bins)) if x < len(bins) -1]
plt.xticks(range(9),x_ticks)
plt.title('Histogram of reflectivity')
plt.grid()
plt.show()

plt.figure()
refl_sum = np.sum(refl_hist_array, axis=0)
refl_sum_log = np.log(refl_sum)
hist,edge_slice = (refl_sum_log,np.array(bins))
plt.bar(x_axis,hist,width,align='center')
x_ticks = [str(bins[x]) +' : ' + str(bins[x + 1]) for x in range(len(bins)) if x < len(bins) -1]
plt.xticks(range(9),x_ticks)
plt.title('Log of the reflectivity')
plt.grid()
plt.show()




