# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 18:18:28 2015

@author: Aditya
"""

'''
    KNKX NEXRAD : 32.91889, -117.04194 (from DDMMSS)
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

ftp_sopac =  ftplib.FTP('garner.ucsd.edu','anonymous','adi@123.com')
#ftp_noaa = ftplib.FTP('www.ngs.noaa.gov','anonymous','adi@gmail.com')


#KNKX
#lat1 = 32.91889
#long1 = -117.04194

#KSOX
#lat1 = 33.81778
#long1 = -117.635

#KEYX
lat1 = 35.09778
long1 = -117.56

#lat2 = +33.142464
#long2 = -116.401708
files_list=ftp_sopac.nlst('/pub/docs/site_logs/')
print files_list 
log_file_list=[]
for logfile in files_list:
    if logfile[-4:] == '.log':
        log_file_list.append(logfile)

print log_file_list
site_Coordinates= []
NEXRAD_GPS_list = []
site_coordinates_lat = []
site_coordinates_long = []
g = open('KEYX_GPS_SOPAC_sites.csv','w+')
g.write('GPS site id' + ',' + 'latitude' + ',' + 'longitude' + '\n')
g.close()
#ftp_sopac.retrbinary('RETR /pub/docs/site_logs/p482.log', logs.write)

for log in log_file_list:
    logs = StringIO()
    ftp_sopac.retrbinary('RETR ' + log, logs.write)
    site_log_file = logs.getvalue()
    ind_lat = site_log_file.find('Latitude (N is +)      :')
    ind_long = site_log_file.find('Longitude (E is +)     :')
    if isfloat(site_log_file[ind_lat + 25 : ind_lat + 35]) and isfloat(site_log_file[ind_long + 25 : ind_long + 36]):
        lat2 = float(site_log_file[ind_lat + 25 : ind_lat + 35])
        long2 = float(site_log_file[ind_long + 25 : ind_long + 36])
        if distance_on_unit_sphere(lat1,long1,lat2 * 1/10000.0,long2* 1/10000.0) * 6373.0 <= 150.0:
            print 'Found site in Radar rande: ' + log[20:24]
            with open('KEYX_GPS_SOPAC_sites.csv','a') as f:
                f.write(str(log[20:24]) + ',' + str(lat2) + ',' + str(long2) + '\n')
            NEXRAD_GPS_list.append(log[20:24])
            site_coordinates_lat.append(lat2)
            site_coordinates_long.append(long2)
    print log[20:24]
    print site_log_file[ind_lat + 25 : ind_lat + 35],site_log_file[ind_long + 25 : ind_long + 36]

    
print len(log_file_list)

#[\n\r].*Object Name:\s*([^\n\r]*)        
# /pub/docs/site_logs/p482.log
'''
       Latitude (N is +)      : +331424.64
       Longitude (E is +)     : -1164017.08
       
'''

ftp_sopac.close()

#dist = distance_on_unit_sphere(lat1,long1,lat2,long2)

print '----------------------------------------------------'

print NEXRAD_GPS_list
print site_coordinates_lat
print site_coordinates_long



