# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:49:18 2015

@author: adityanagarajan
This script reads GAMIT met files which contain ipw,temp,pressure and puts them into csv 
files, one file for a day. This script only works for 2015 data. 
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

#yr = sys.argv[1]
#doy = sys.argv[2]
#year = '14'
#doy = '128'

def build_met_array(yr,doy):
    columns = ['site']
    columns.extend(['{0}:{1}'.format(x,y) for x in range(24) for y in [15,45]])
    writeData = pd.DataFrame(columns=columns)
    writeDataPres = pd.DataFrame(columns = columns)
    writeDataTemp = pd.DataFrame(columns=columns)

    i=0
    print '-'*30
    for s in sites:
    
        base_path = '/home/aditya/UMASS/DFWnetDB' + os.sep + s[-1] + '/20' + str(yr) + os.sep +doy  + os.sep + 'met_'+ s[0] + '.' + str(yr)[-2:] + doy
        #base_path = '/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/GAMIT/TxDOT/net1/met_okar.14128'
#        print base_path
        if os.path.exists(base_path):
            metData = np.loadtxt(base_path,dtype=np.float,skiprows=4)
            metData = metData[metData[:,1] == float(doy),:]
            if metData[:,8].shape[0] > 47:
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
                print 'Full data NA: ' + base_path
                temp_list = [s[0]]
                temp_list.extend([-9.9]*48)
                writeData.loc[i] = temp_list
                print 'error missing data for ' + s[0]
                i+=1
        else:
            print 'File not found ' + base_path
            temp_list = [s[0]]
            temp_list.extend([-9.9]*48)
            writeData.loc[i] = temp_list
            i+=1

    d = doy2date(int(yr),int(doy))
    out_path = '/var/www/html/gpsmet/DFWnet/20' + str(yr) + os.sep 
    writeData.to_csv(out_path + 'IPW' + os.sep + 'IPW' + '20'+ str(yr) + str(d.tm_mon).zfill(2) + str(d.tm_mday).zfill(2) + '.csv',index=False)
    writeDataPres.to_csv(out_path + 'Pressure' + os.sep + 'PR' + '20' + str(yr) + str(d.tm_mon).zfill(2) + str(d.tm_mday).zfill(2) + '.csv',index=False)
    writeDataTemp.to_csv(out_path + 'Temperature' + os.sep +  'Temp' +'20' + str(yr) + str(d.tm_mon).zfill(2) + str(d.tm_mday).zfill(2) + '.csv',index=False)


if __name__ ==  '__main__':
    yr = 15
    doy_list = [str(x) for x in range(121,243)]
    for doy in doy_list:
        build_met_array(yr,doy)    
    
        
        
    
    
    
    
