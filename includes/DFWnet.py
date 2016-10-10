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
        self.sites = np.loadtxt('/Users/adityanagarajan/projects/nowcaster/data/KFWS_GPS_ASOS_locations_new.csv',dtype='S',delimiter = ',',skiprows = 1)
        self.base_path = '/home/aditya/UMASS/DFWnet/'
        self.IPWvals_2014 = np.load('../data/2014IPW_data.npy')
        self.IPWvals_2015 = np.load('../data/2015IPW_data.npy')
        self.IPWvals_2016 = np.load('../data/2016IPW_data.npy')
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
    
    def distance_on_unit_sphere(self,lat1, long1, lat2, long2):
        '''Returns distance between two points in km.
        inpot lat1, long1, lat2, long2 in decimal degrees'''
        # Convert latitude and longitude to 
        # spherical coordinates in radians.
        degrees_to_radians = math.pi/180.0

        # phi = 90 - latitude
        phi1 = (90.0 - lat1)*degrees_to_radians
        phi2 = (90.0 - lat2)*degrees_to_radians

        # theta = longitude
        theta1 = long1*degrees_to_radians
        theta2 = long2*degrees_to_radians

        # Compute spherical distance from spherical coordinates.

        # For two locations in spherical coordinates 
        # (1, theta, phi) and (1, theta, phi)
        # cosine( arc length ) = 
        #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
        # distance = rho * arc length

        cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
           math.cos(phi1)*math.cos(phi2))
        arc = math.acos( cos )

        # Remember to multiply arc by the radius of the earth 
        # in your favorite set of units to get length.
        return arc * 6373.0
    


'''
def date2doy(yr,mn,day):
    return datetime.datetime(yr, mn, day).timetuple().tm_yday

def doy2date(yr,doy):
    return datetime.date.fromordinal(datetime.date(yr, 1, 1).toordinal() + doy - 1).timetuple() 


'''

        
    