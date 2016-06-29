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
import os
import BuildDataSet

plt.ion()

def main():
    data_builder = BuildDataSet.dataset(num_points = 500)
    pixels = data_builder.sample_random_pixels()
    base_path = '../data/RadarData/Decimated/'
    file_list = os.listdir(base_path)
    file_list = filter(lambda x: x[-4:] == '.npy',file_list)
    full_data = np.zeros((500,len(file_list),48))
    pt_ctr = 0
    for x_,y_ in pixels:
        print x_,y_
        for f_ctr,f in enumerate(file_list):
            temp_array = np.load(base_path + f)            
            full_data[pt_ctr,f_ctr,:] = temp_array[:,y_,x_]
        pt_ctr+=1
    np.save(base_path + 'full_data_decimated.npy',full_data)

def histogram_averages():
    base_path = '../data/RadarData/Averages/'
    arr = np.load(base_path + 'full_data_averages.npy')
    counts,bins = np.histogram(arr,bins = [0.,10.,20.,30.,40.,50.,60.])
    binsc = bins[:-1] + np.diff(bins)/2.
    plt.figure()
    plt.bar(binsc[1:],counts[1:]/(counts.sum()*1.0), width = np.diff(bins)[0])
    plt.title('Histogram for 30 minute averages of reflectivity')
    plt.grid()
#    plt.bar(binsc1, counts1/(counts1.sum()*1.0), width=np.diff(bins1)[0])
#    plt.grid(True)
    print counts,bins

def histogram_decimated():
    base_path = '../data/RadarData/Decimated/'
    arr = np.load(base_path + 'full_data_decimated.npy')
    counts,bins = np.histogram(arr,bins = [0.,10.,20.,30.,40.,50.,60.])
    binsc = bins[:-1] + np.diff(bins)/2.
    plt.figure()
    plt.bar(binsc[1:],counts[1:]/(counts.sum()*1.0), width = np.diff(bins)[0])
    plt.title('Histogram for 30 minute decimated files of reflectivity')
    plt.grid()
#    plt.bar(binsc1, counts1/(counts1.sum()*1.0), width=np.diff(bins1)[0])
#    plt.grid(True)
    print counts,bins
    

if __name__ == '__main__':
#    main()
    histogram_averages()
    histogram_decimated()

'''
counts1, bins1 = np.histogram(df_train["accuracy"], bins=50)
binsc1 = bins1[:-1] + np.diff(bins1)/2.

counts2, bins2 = np.histogram(df_test["accuracy"], bins=50)
binsc2 = bins2[:-1] + np.diff(bins2)/2.

plt.figure(0, figsize=(14,4))

plt.subplot(121)
plt.bar(binsc1, counts1/(counts1.sum()*1.0), width=np.diff(bins1)[0])
plt.grid(True)
plt.xlabel("Accuracy")
plt.ylabel("Fraction")
plt.title("Train")
'''

#data_builder = BuildDataSet.dataset()
#
#file_list =  data_builder.Radarfiles
#
#
#
#bins = [-30.,-20.,-10.,0.,10.,20.,30.,40.,50.,60.]
#
#refl_hist = []
#
#for f1 in file_list:
#    RadarMatrix = np.load(data_builder.TrainTestdir + f1)
#    refl_histogram = np.histogram(RadarMatrix,bins)
#    refl_hist.append(refl_histogram[0])
#    print refl_histogram
#
#refl_hist_array = np.array(refl_hist)
#
#width = 0.5
#x_axis = np.arange(9)
#plt.figure()
#refl_sum = np.sum(refl_hist_array, axis=0)
#hist,edge_slice = (refl_sum,np.array(bins))
#plt.bar(x_axis,hist,width,align='center')
#x_ticks = [str(bins[x]) +' : ' + str(bins[x + 1]) for x in range(len(bins)) if x < len(bins) -1]
#plt.xticks(range(9),x_ticks)
#plt.title('Histogram of reflectivity')
#plt.grid()
#plt.show()
#
#plt.figure()
#refl_sum = np.sum(refl_hist_array, axis=0)
#refl_sum_log = np.log(refl_sum)
#hist,edge_slice = (refl_sum_log,np.array(bins))
#plt.bar(x_axis,hist,width,align='center')
#x_ticks = [str(bins[x]) +' : ' + str(bins[x + 1]) for x in range(len(bins)) if x < len(bins) -1]
#plt.xticks(range(9),x_ticks)
#plt.title('Log of the reflectivity')
#plt.grid()
#plt.show()




