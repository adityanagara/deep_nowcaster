# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:48:20 2016

@author: adityanagarajan
This script plots a histogram of the difference in ipw values firstly. 
One of the things we need to do is filter out stations with a very large difference 
in the ipw values and then take the monthly averages. 
"""

import numpy as np
from matplotlib import pyplot as plt
import DFWnet
import BuildDataSet
import ftplib
import os
from netCDF4 import Dataset
import shutil

Months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

def ipw_diff_distribution():
    x = DFWnet.CommonData()
    
    ipw_2014 = x.IPWvals_2014
    
    print ipw_2014.shape
    
    plt.figure()
    
    print ipw_2014[0,35,0]
    
    plt.plot(ipw_2014[:,35,1:].reshape(365*48))
    
    for i in range(44):
        stn_diff = np.diff(ipw_2014[:,i,1:].reshape(365*48).astype('float'))
    
        stn_diff[np.isnan(stn_diff)] = 0.
    
        stn_diff = [abs(num) for num in stn_diff]
        
        counts,bins = np.histogram(stn_diff)
        
        bins1 = bins[:-1] + np.diff(bins) / 2.
        
        print bins1,counts/(counts.sum()*1.)
        plt.figure()
        plt.bar(bins1,counts/(counts.sum()*1.),width = np.diff(bins1)[0])
        
        plt.title('IPW difference distribution for station: ' + ipw_2014[0,i,0])
        plt.grid()
        
        plt.savefig('../output/ipw_difference_distribution/Plot_' + str(i) + '.png')

    plt.show()

def ipw_diff_distribution_monthly():
    
    x = DFWnet.CommonData()
    
    # test may diff
    x.date2doy(14,5,1)
    start_day = x.doy
    x.date2doy(14,6,1)
    end_day = x.doy
    
    
    ipw_2014 = x.IPWvals_2014
    
    for i in range(44):
        
        stn_diff = np.diff(ipw_2014[start_day:end_day,i,1:].reshape(-1,).astype('float'))
        
        stn_diff[np.isnan(stn_diff)] = 0.
        
        stn_diff = np.array([abs(num) for num in stn_diff])
        bins_ = [10*b for b in range(5)] # ,bins = bins_
        counts,bins = np.histogram(stn_diff)
        print ipw_2014[0,i,0],counts,bins
        stn_diff = stn_diff[stn_diff > 20.]
        print stn_diff
        
        
        
    
def reflectivity_monthly_distribution():
    '''Plots the distribution for each month in the data set. Given an 
    archive of NEXRAD files we are going to seek to find the days where
    rainfall was present to pupulate out data set. Try and compute the 
    monthly distribution of rainfall the prediction domain.'''
    DFW = DFWnet.CommonData()
    reflectivity = BuildDataSet.reflectivity_fields()
    # Loop thru all the days in the data set 121 - 144
    # Download each file keep the level 3 data throw the rest
    # then check for each day whether there was a storm or not
    # if there was a storm keep that folder
    # else delete that folder and its contents
    order_dict = {}
    order_dict = {14: 'HAS010777764', 15: 'HAS010777767'}
    initial = os.getcwd()
    for yr in [14]:
        for d in range(159,244):
            DFW.doytodate(int(yr),d)
#            file_to_get = 'NWS_NEXRAD_NXL3_KFWS_20' +DFW.yr + DFW.mon + DFW.day + '000000_20' + DFW.yr + DFW.mon + DFW.day + '235959.tar.gz'
            new_dir = '../data/RadarData/NEXRAD/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + DFW.day + os.sep
            
            if not os.path.exists(new_dir):                
                os.mkdir(new_dir)
            os.chdir(new_dir)
            reflectivity.FTPNEXRADfile(DFW.mon,DFW.day,DFW.yr,order_dict[yr])
            os.chdir(initial)
            reflectivity.keepLevel3files(new_dir)
            reflectivity.ConvertToNETCDF(new_dir)
            # Develop a logic here which takes files that have an average greater than  
            # 20 dBZ for its rainy days
            file_list = os.listdir(new_dir)
            # define an array the with size of the number of files in that day
            # and a 100x100 grid to hold each time step worth of data
            out_array = np.zeros((len(file_list),100,100))
            time_array = []
            for i,fl in enumerate(file_list):
                rad = Dataset(new_dir + fl)
                out_array[i,...] = reflectivity.reflectivity_polar_to_cartesian(rad)
                time_array.append(rad.time_coverage_end.split('T')[1])
                os.remove(new_dir + fl)
            
            np.save(new_dir + 'reflectivity_array_' + str(yr) + '_' + str(d) + '.npy',out_array)
            np.save(new_dir + 'time_array_' + str(yr) + '_' + str(d) + '.npy',time_array)
            

def main():
    ipw_diff_distribution_monthly()
#    reflectivity_monthly_distribution()
    
if __name__ == '__main__':
    main()
    