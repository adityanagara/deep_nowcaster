# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:08:36 2016

@author: adityanagarajan
"""

import numpy as np
import pandas as pd
import DFWnet;reload(DFWnet)
import os
import urllib2


def parse_ASOS_metaData():
    file_path = '../data/asos-stations.txt'
    with open(file_path,'rb') as f1:
        f_lines = f1.read()
        split_lines = f_lines.split('\n')
        delimiter_key = split_lines[3].strip().split(' ')
        delimiter_key = [de.count('-') for de in delimiter_key]
        delimiter_key.insert(0,0)
        header = split_lines[2].split()
        ASOSframe = pd.DataFrame(columns = header)
        start_index = np.cumsum((delimiter_key)) + np.arange(15)
        delimiter_key = delimiter_key[1:]
        for l in range(4,len(split_lines) - 1):
            temp = split_lines[l]
            for h,head in enumerate(header):
                s = start_index[h]
                e = s + delimiter_key[h]
                field = temp[s:e].strip()
                if field:
                    ASOSframe.loc[l - 4,head] = field
    ASOSframe.to_csv('asos-stations.csv')
    return ASOSframe
    
#KFWS Dallas
lat1 = 32.57278
long1 = -97.30278

ASOS_stations = parse_ASOS_metaData()
'''
for s in gps_sites:
    #print s
    temp_dict = {}
    for wx in wx_sites:
        temp_dict[wx[0]] = distance_on_unit_sphere(float(s[1]),float(s[2]),float(wx[1]),float(wx[2])) * 6373.0
    gps_wx[s[0]]=  temp_dict.keys()[temp_dict.values().index(min(temp_dict.values()))]
'''

x = DFWnet.CommonData()
asos_station_list = []
for s in range(ASOS_stations.shape[0]):
    if x.distance_on_unit_sphere(lat1,long1,float(ASOS_stations.iloc[s].LAT),float(ASOS_stations.iloc[s].LON)) <= 230.0:
        asos_station_list.append((ASOS_stations.iloc[s].CALL,float(ASOS_stations.iloc[s].LAT),float(ASOS_stations.iloc[s].LON),ASOS_stations.iloc[s].ELEV))
        print ASOS_stations.iloc[s].CALL,ASOS_stations.iloc[s].LAT,ASOS_stations.iloc[s].LON,ASOS_stations.iloc[s].ELEV

print asos_station_list

sensor_locations = pd.read_csv('../data/KFWS_GPS_ASOS_locations.csv')

GPS_ASOS = {}
for s in range(sensor_locations.shape[0]):
    GPS_ASOS[sensor_locations.ix[s,'GPSid']] = []
    for wx in asos_station_list:
        GPS_ASOS[sensor_locations.ix[s,'GPSid']].append((wx[0],x.distance_on_unit_sphere(sensor_locations.ix[s,'GPSlat'],sensor_locations.ix[s,'GPSlong'],wx[1],wx[2])))
        
for siteid in GPS_ASOS.keys():
    GPS_ASOS[siteid].sort(key = lambda x: x[1])
    GPS_ASOS[siteid] = GPS_ASOS[siteid][0]
    

asos_station_list = pd.DataFrame(asos_station_list,columns = ['id','lat','long','elev'])

#asos_station_list.ix['elev'] = asos_station_list.elev.apply(lambda x: float(x) / 3.2808)

print GPS_ASOS

'''
http://weather.noaa.gov/data/nsd_cccc.txt
for f in files_list:
    response = urllib2.urlopen(baseURL + f)
    html = response.read().split('\n')
    html = map(lambda x: x.split(),html)
'''

for site in GPS_ASOS.keys():
    print GPS_ASOS[site][0]
    sensor_locations.ix[sensor_locations.GPSid == site,'newASOSid'] = 'K' + GPS_ASOS[site][0]
    sensor_locations.ix[sensor_locations.GPSid == site,'newASOSlat'] = asos_station_list.ix[asos_station_list.id == GPS_ASOS[site][0],'lat'].values[0]
    sensor_locations.ix[sensor_locations.GPSid == site,'newASOSlong'] = asos_station_list.ix[asos_station_list.id == GPS_ASOS[site][0],'long'].values[0]
    sensor_locations.ix[sensor_locations.GPSid == site,'newASOSheight'] = asos_station_list.ix[asos_station_list.id == GPS_ASOS[site][0],'elev'].values[0]
    sensor_locations.ix[sensor_locations.GPSid == site,'newDistance'] = GPS_ASOS[site][1]

# Once the file dumps we need to manually fill in unavailable data with
# values

sensor_locations['newASOSheight'] = sensor_locations.newASOSheight.apply(lambda x: float(x) / 3.2808)
#sensor_locations.to_csv('../data/KFWS_GPS_ASOS_locations_new.csv',index = False)


    
    
    
    

    

        
        
        





        




    
    
    








        
        
        
    




    

    
    


    