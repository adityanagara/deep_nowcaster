# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:12:51 2015

@author: adityanagarajan
"""

import numpy as np
import DFWnet
import os

import pandas as pd

def get_IPW_vals(doy):
    
    IPWvals = np.load('data/2014Data_update1.npy')
    return IPWvals[doy -1,:,:]


# Function which returns a matrix of normalized values (Normalization done wrt to month)
def NormalizeIPW_Normal(DAT,doy):
    DFW = DFWnet.CommonData()
    
    DFW.doytodate(14,doy)
    
    SummaryStats = pd.read_csv('output/SitesSummary.csv',index_col=0)
    
    IPWdataN = np.array(map(lambda x: (x[1:].astype('float') - SummaryStats[SummaryStats.site.values == x[0]][DFW.mon + '_Avg'].values)/SummaryStats[SummaryStats.site.values == x[0]][DFW.mon + '_Std'].values,DAT))
    
    return IPWdataN.astype('float')


DFW = DFWnet.CommonData()
Anomalies = []
for doy in range(1,366):
    
    IPWvals = get_IPW_vals(doy)
    
    IPWvals = NormalizeIPW_Normal(IPWvals,doy)
    
    tempArr = np.where(IPWvals > 2.0)
    
    if tempArr[0].size > 30:
        
        DFW.doytodate(14,doy)
        
        print 'Weather anomaly on ' + DFW.mon + '/' + DFW.day + '/' + '2014'
        
        Anomalies.append([DFW.mon,DFW.day])


        



#time_index = ['{0}{1}'.format(str(x).zfill(2),str(y).zfill(2)) for x in range(24) for y in [0,30]]
#
#
## Function that finds radar data closest to the given GPS met data
#def find_closest_radar_data(t):
#    # List the files for the day in question
#    files = os.listdir('data/RadarData/MAY09')
#    
#    hr = t[:2]
#    mn = t[-2:]
#    # Get files only with .nc extension
#    files = filter(lambda x: x[-3:] == '.nc',files)
#    
#    # Filter all files for the hour in question
#    temp_file = filter(lambda x: x[-7:-3][:2] == hr,files)
#    
#    
#    # Find the closest file to the given time stame (up to 5 minutes ahead is fine)
#    if mn == '30':
#        the_file = filter(lambda x: int(x[-7:-3][-2:])  >= int(mn) -6  and int(x[-7:-3][-2:]) <= int(mn) + 6,temp_file)
#        diff = map(lambda x: abs(int(x[-7:-3][-2:]) - int(mn)),the_file)
#        the_file.sort(key = lambda x: abs(int(x[-7:-3][-2:]) - int(mn)))
#        
#    else:
#        temp_file = filter(lambda x: x[-7:-3][:2] == hr or x[-7:-3][:2] == str(int(hr) -1).zfill(2),files)       
#        the_file = filter(lambda x: 
#            int(x[-7:-3][:2] == str(int(hr) -1).zfill(2) and int(x[-7:-3][-2:]) >= 55  or (x[-7:-3][:2] == hr and int(x[-7:-3][-2:])  >= int(mn)  and int(x[-7:-3][-2:]) <= int(mn) + 6)),
#                temp_file)
#        the_file.sort(key = lambda x: abs((float(x[-7:-3][:2]) + float(x[-7:-3][-2:])/60.0) - int(hr) ))
#        
#        diff = map(lambda x: abs((float(x[-7:-3][:2]) + float(x[-7:-3][-2:])/60.0) - int(hr) ),the_file)
#     
#    print hr
#    print diff
#    print the_file
#    
#    return the_file[0]
#    
#
#
#for t in range(48):
#    file_name = find_closest_radar_data(time_index[t])
    
    


