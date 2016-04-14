# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:50:28 2015

@author: adityanagarajan
"""

'''
This code check the processing for the entire year by parsing GAMIT.fatal file in the doy directory

'''

import os
#import numpy as np
import datetime

def date2doy(yr,mn,day):
    return datetime.datetime(yr, mn, day).timetuple().tm_yday

def doy2date(yr,doy):
    return datetime.date.fromordinal(datetime.date(yr, 1, 1).toordinal() + doy - 1).timetuple() 


file_path = '/home/aditya/UMASS/DFWnetDB/' #+ 'net1/2014'

doy_list = [str(x).zfill(3) for x in range(121,243)]

nets = ['net1','net2','net3','net4']

yr = '2015'

for n in nets:
    for d in doy_list:
        if os.path.exists(file_path + n+ os.sep + yr + os.sep + d + os.sep):
            if os.path.exists(file_path + n+ os.sep + yr + os.sep + d + os.sep + 'GAMIT.fatal'):
                print 'Data not available: ' + n + ' ' + d
            else:
                continue
        else:
            print 'Day folder not available: ' +n + ' '+ d
            
        

