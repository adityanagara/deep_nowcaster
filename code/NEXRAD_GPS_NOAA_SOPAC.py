# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 18:18:28 2015

@author: Aditya
"""

'''
    This script check for 
    KFWS NEXRAD : 55.572, -97.302
'''


import math
import ftplib
from StringIO import StringIO
import re

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

def convert_gpsmet_coord(latitude,longitude):
    if latitude[0] == '-':
        la_Degrees = latitude[:3]
        la_minutes = latitude[3:5]
        la_seconds = latitude[5:]
    else:
        la_Degrees = latitude[:2]
        la_minutes = latitude[2:4]
        la_seconds = latitude[4:]
    if isfloat(la_Degrees) and isfloat(la_minutes) and isfloat(la_seconds):
        la_Degrees = float(la_Degrees)
        la_minutes = float(la_minutes)/60.0
        la_seconds = float(la_seconds)/3600.0
    if longitude[0] == '-':
        lo_Degrees = longitude[:3]
        lo_minutes = longitude[3:5]
        lo_seconds = longitude[5:]
    else:
        lo_Degrees = longitude[:2]
        lo_minutes = longitude[2:4]
        lo_seconds = longitude[4:]
    if isfloat(lo_Degrees) and isfloat(lo_minutes) and isfloat(lo_seconds):
        lo_Degrees = float(lo_Degrees)
        lo_minutes = float(lo_minutes)/60.0
        lo_seconds = float(lo_seconds)/3600.0
    if la_Degrees > 0.0:
        out_lat = la_Degrees + la_minutes + la_seconds
    else:
        out_lat = la_Degrees - la_minutes - la_seconds
    if lo_Degrees > 0.0:
        out_long = lo_Degrees + lo_minutes + lo_seconds
    else:
        out_long = lo_Degrees - lo_minutes - lo_seconds
    return float(out_lat),float(out_long)


'''
       Latitude (N is +)      : +331424.64
       Longitude (E is +)     : -1164017.08
       
'''

#ftp_sopac.close()
#KFWS Dallas
lat1 = 32.57278
long1 = -97.30278

#KNKX
#lat1 = 32.91889
#long1 = -117.04194


ftp_noaa = ftplib.FTP('www.ngs.noaa.gov','anonymous','adi@gmail.com')

file_list_noaa = ftp_noaa.nlst('/cors/station_log/')

out_site_file = 'KFWS_230km_sites.csv'
#print file_list_noaa
temp_file = open(out_site_file,'w+')
temp_file.write('GPS site id' + ',' + 'latitude' + ',' + 'longitude' + '\n')
temp_file.close()

for log_file in file_list_noaa:
    #print log_file[-8:]
    if log_file[-8:] == '.log.txt':
        #print log_file[18:22]
        temp_log = StringIO()
        ftp_noaa.retrbinary('RETR ' + log_file, temp_log.write)
        site_log_file = temp_log.getvalue()
        ind_lat = site_log_file.find('Latitude (N is +)      :')
        ind_long = site_log_file.find('Longitude (E is +)     :')
        ind_height = site_log_file.find('Elevation (m,ellips.)  :')
        print 'height'
        print site_log_file[ind_height + 25:ind_height + 30]
        if isfloat(site_log_file[ind_lat + 25 : ind_lat + 35]) and isfloat(site_log_file[ind_long + 25 : ind_long + 36]):
            lat2 = float(site_log_file[ind_lat + 25 : ind_lat + 35])
            long2 = float(site_log_file[ind_long + 25 : ind_long + 36])
            #print lat2,long2
            lat2,long2 = convert_gpsmet_coord(str(lat2),str(long2))
            print lat2,long2
            if distance_on_unit_sphere(lat1,long1,lat2 ,long2) * 6373.0 <= 230.0:
                print 'Found site in radar range: ' + log_file[18:22]
                
                with open(out_site_file,'a') as f:
                    f.write(str(log_file[18:22]) + ',' + str(lat2) + ',' + str(long2) + ',' + site_log_file[ind_height + 25:ind_height + 30] +'\n')
#            NEXRAD_GPS_list.append(log[20:24])
#            site_coordinates_lat.append(lat2)
#            site_coordinates_long.append(long2)
        
        
        
        
#/cors/station_log/

ftp_noaa.close()

print '----------------------------------------------------'

#print NEXRAD_GPS_list
#print site_coordinates_lat
#print site_coordinates_long



