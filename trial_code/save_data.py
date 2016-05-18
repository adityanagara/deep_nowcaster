# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:25:19 2015

@author: adityanagarajan
This script takes all the IPW values for the entire year from each day csv file
and dumps it into a single numpy array. 
"""

import  numpy as np
import DFWnet
import os

DFW = DFWnet.CommonData()
def save_to_numpy(yr):
    mon_day_dict = {}
    # /var/www/html/gpsmet/DFWnet/2015/IPW/
    file_path = '/var/www/html/gpsmet/DFWnet/20' + str(yr) + '/IPW/'

    for x in range(1,366):
        DFW.doytodate(15,x)
        if DFW.mon not in mon_day_dict.keys():
            mon_day_dict[DFW.mon] = []
        mon_day_dict[DFW.mon].append(DFW.day)
    
    get_files =['IPW20' + str(yr) + '{0}{1}.csv'.format(x,y) for x in sorted(mon_day_dict) for y in mon_day_dict[x]]
    sites = DFW.sites[:,0]
    n = len(get_files)
    IPWval = np.zeros((n,len(sites),49)).astype('S')
    for g in range(len(get_files)):
        if os.path.exists(file_path + get_files[g]):
            print get_files[g]
            for s in range(len(sites)):
                temp = file_path + get_files[g]
                data = np.loadtxt(temp,delimiter=',',dtype='S',skiprows=1)
                data[data =='-999.99'] = np.nan
                IPWval[g,s,:] = data[data[:,0] == sites[s]]
    np.save('2015IPW_data.npy',IPWval)
    
if __name__ == '__main__':
    yr = 15
    save_to_numpy(yr)

#
#for s in range(len(sites)):
#    itr = 0
#    for f in get_files:
#        temp = file_path + f
#    
#        if os.path.exists(temp):
#        
#            data = np.loadtxt(temp,delimiter=',',dtype='S',skiprows=1)
#            data[data =='-999.99'] = np.nan
#
#            IPWval[itr,s,:] = data[data[:,0] == sites[s],1:].astype('float')
#            
#            itr+=1  
#    plt.figure(figsize=(12, 8))
#    plt.xticks(np.arange(min(np.hstack((t[:,0,:]))), max(np.hstack((t[:,0,:])))+1, 720),[str(x).zfill(2) for x in range(1,13)])
#    plt.ylim((0,80))
#    plt.plot(np.hstack((t[:,0,:])),np.hstack((IPWval[:,s,:])),label = sites[s])
#    plt.ylabel('Integrater Precipitable Water (mm)')
#    plt.xlabel('Time')
#    plt.title('1 year IPW Plots for station: ' + sites[s])
#    plt.grid()
#    if not os.path.exists(file_path + 'Plots/' + sites[s]):
#        os.mkdir(file_path + 'Plots/' + sites[s])
#    plt.savefig(file_path +  'Plots/' + sites[s] + '/2014_Year_Plot.png')
#    plt.close()

# Plot 12 graphes under each site for the month






