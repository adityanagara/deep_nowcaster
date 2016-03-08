# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:53:17 2015

@author: adityanagarajan
"""

import numpy as np
import datetime
import math
import time

class CommonData:
    def __init__(self):
        #KFWS Dallas
        self.KFWSlat = 32.57278
        self.KFWSlong = -97.30278
        self.sites = np.loadtxt('/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/data/KFWS_GPS_ASOS_locations.csv',dtype='S',delimiter = ',',skiprows = 1)
        self.base_path = '/home/aditya/UMASS/DFWnet/'
        self.IPWvals = np.load('/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/data/2014Data_update1.npy')
        self.IPWvals_2015 = np.load('/Users/adityanagarajan/projects/nowcaster/data/2015IPW_data.npy')
        self.Prvals = np.load('/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/data/2014PressureData_update1.npy')
        self.cnvl_ip = '129.107.93.30'
    def doytodate(self,yr,doy):
        self.yr = str(datetime.date.fromordinal(datetime.date(yr, 1, 1).toordinal() + doy - 1).timetuple().tm_year)
        self.mon = str(datetime.date.fromordinal(datetime.date(yr, 1, 1).toordinal() + doy - 1).timetuple().tm_mon).zfill(2)
        self.day =  str(datetime.date.fromordinal(datetime.date(yr, 1, 1).toordinal() + doy - 1).timetuple().tm_mday).zfill(2)
        
    def date2doy(self,yr,mn,day):
        self.doy =  datetime.datetime(yr, mn, day).timetuple().tm_yday
    
    def rt_compute_gps_week(self):
        secsInWeek = 604800 
        secsInDay = 86400 
        gpsEpoch = (1980, 1, 6, 0, 0, 0)  # (year, month, day, hh, mm, ss)
        epochTuple= gpsEpoch + (-1,-1,0) 
        t0=time.mktime(epochTuple)
        time_tuple= (time.gmtime().tm_year, time.gmtime().tm_mon, time.gmtime().tm_mday, time.gmtime().tm_hour, time.gmtime().tm_min,time.gmtime().tm_sec)
        time_tuple = time_tuple + (-1,-1,0)
        secFract = time.gmtime().tm_sec % 1
        t=time.mktime(time_tuple)
        t = t+14
        tdiff = t-t0
        gpsWeek = int(math.floor(tdiff/secsInWeek))
        gpsSOW = (tdiff % secsInWeek)  + secFract
        gpsDay = int(math.floor(gpsSOW/secsInDay))
        return (gpsWeek,gpsDay)
    
    def make_alpha_dict(self):
        alpha_dict = {}
        numa_dict = {}
        alpha_list = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        for i in range(0,26):
            alpha_dict[i] = alpha_list[i]
            numa_dict[i] = str(i).zfill(2)
        return alpha_dict, numa_dict
    


'''
def date2doy(yr,mn,day):
    return datetime.datetime(yr, mn, day).timetuple().tm_yday

def doy2date(yr,doy):
    return datetime.date.fromordinal(datetime.date(yr, 1, 1).toordinal() + doy - 1).timetuple() 


'''

        
    