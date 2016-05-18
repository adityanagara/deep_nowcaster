# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:06:31 2015

@author: aditya

Note: Script to find all weather stations encompessing the KFWS radar and 
    map the weather station closest to the GPS site

    
"""
import os
import numpy as np
import math
import csv



def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

 
def distance_on_unit_sphere(lat1, long1, lat2, long2):
 
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
    return arc
    

#KFWS Dallas
lat1 = 32.57278
long1 = -97.30278

file_path = 'data/AllStations.txt'
gps_file_path = 'data/KFWS_Final_sites_with_data.csv'
sites_arr = np.zeros((584,4),dtype='|S6')
gps_sites = np.loadtxt(gps_file_path,delimiter=',',dtype='S')

ctr = 0
with open(file_path,'r') as f:
    for line in f:
        sites_arr[ctr,0] = line[20:24]
        sites_arr[ctr,1] = str(float(line[39:44].split()[0])  + float(line[39:44].split()[1])/60.0)
        sites_arr[ctr,2] = '-' + str(float(line[47:53].split()[0]) + float(line[47:53].split()[1])/60.0)
        sites_arr[ctr,3] = line[62:63]        
        ctr +=1
        
num = 0
wx_sites = []
for i in range(sites_arr.shape[0]):
    if distance_on_unit_sphere(lat1,long1,float(sites_arr[i,1]),float(sites_arr[i,2]))* 6373.0 <= 230.0 and sites_arr[i,3] == 'X':
        wx_sites.append(sites_arr[i,:])
        num = num + 1
        
wx_sites = np.array(wx_sites)
gps_wx = {}
for s in gps_sites:
    #print s
    temp_dict = {}
    for wx in wx_sites:
        temp_dict[wx[0]] = distance_on_unit_sphere(float(s[1]),float(s[2]),float(wx[1]),float(wx[2])) * 6373.0
    gps_wx[s[0]]=  temp_dict.keys()[temp_dict.values().index(min(temp_dict.values()))]
    

writer = csv.writer(open('data/gps_wxstation.csv', 'wb+'))
for key, value in gps_wx.items():
    print key,value
    writer.writerow([key, value])
writer.close()
      





