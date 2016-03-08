# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:49:18 2015

@author: adityanagarajan
"""

import numpy as np
import os
import sys
import csv
import datetime
import pandas as pd



def date2doy(yr,mn,day):
    return datetime.datetime(yr, mn, day).timetuple().tm_yday

def doy2date(yr,doy):
    return datetime.date.fromordinal(datetime.date(yr, 1, 1).toordinal() + doy - 1).timetuple() 


#sites = np.loadtxt('/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/data/KFWS_GPS_ASOS_locations.csv',dtype='S',delimiter = ',',skiprows = 1)

sites = np.loadtxt('/home/aditya/UMASS/RINEXmetgen/SupportFiles/KFWS_GPS_ASOS_locations.csv',dtype='S',delimiter = ',',skiprows = 1)

yr = sys.argv[1]
doy = sys.argv[2]
#year = '14'
#doy = '128'

columns = ['site']
columns.extend(['{0}:{1}'.format(x,y) for x in range(24) for y in [15,45]])
writeData = pd.DataFrame(columns=columns)
writeDataPres = pd.DataFrame(columns = columns)
writeDataTemp = pd.DataFrame(columns=columns)

i=0
for s in sites:
    
    base_path = '/home/aditya/UMASS/DFWnet' + os.sep + s[-1] + '/2014/IPW' +os.sep + 'met_'+ s[0] + '.' + yr + doy
    #base_path = '/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/GAMIT/TxDOT/net1/met_okar.14128'
    print base_path
    if os.path.exists(base_path):
        metData = np.loadtxt(base_path,dtype=np.float,skiprows=4)
        metData = metData[metData[:,1] == float(doy),:]
        '''
        write the name of the file
        IPWyyyymmdd.csv
        Header:
        site 00:15,00:45,01:15,01:45....
        '''
        print metData[:,8].shape
        temp_list = [s[0]]
        temp_list.extend(map(lambda x: '%.2f'%x, metData[:,8]))
        writeData.loc[i] = temp_list
        temp_list = [s[0]]
        temp_list.extend(map(lambda x: '%.2f'%x, metData[:,10]))
        writeDataPres.loc[i] = temp_list
        temp_list = [s[0]]
        temp_list.extend(map(lambda x: '%.2f'%x, metData[:,11]))
        writeDataTemp.loc[i] = temp_list
        
        i+=1
    else:
        temp_list = [s[0]]
        temp_list.extend([-9.9]*48)
        writeData.loc[i] = temp_list
        i+=1

d = doy2date(2014,int(doy))
writeData.to_csv('IPW20' + yr + str(d.tm_mon).zfill(2) + str(d.tm_mday).zfill(2) + '.csv',index=False)
writeDataPres.to_csv('PR20' + yr + str(d.tm_mon).zfill(2) + str(d.tm_mday).zfill(2) + '.csv',index=False)
writeDataTemp.to_csv('Temp20' + yr + str(d.tm_mon).zfill(2) + str(d.tm_mday).zfill(2) + '.csv',index=False)

        
        
    
        
        
    
    
    
    