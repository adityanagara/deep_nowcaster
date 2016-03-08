# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:48:47 2015

@author: adityanagarajan
"""

import os
import numpy as np
import re
import csv
import time
import pandas as pd
import datetime
import math
import sys


def date2doy(yr,mn,day):
    return datetime.datetime(yr, mn, day).timetuple().tm_yday

def doy2date(yr,doy):
    return datetime.date.fromordinal(datetime.date(yr, 1, 1).toordinal() + doy - 1).timetuple() 

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def parse_met_var(site,gps_asos_info,wmo_file_name,line):
    presG = ''
    tempd = ''
    tempw = ''
    altm = 0
    altQ = 0
    tempdry = -99.0
    tempwet = -99.0
    pressWT = 0
    pressSV = 0
    for l in line:
        Mstr  = re.search('^[0-9][0-9]/[0-9][0-9]|^[0-9][0-9]/M[0-9][0-9]|^M[0-9][0-9]/M[0-9][0-9]',l)
        if Mstr:
            Mstr = re.split(r'/|\\',Mstr.group(0))
            tempd,tempw = Mstr
            if 'M' in tempd:
                #print tempd
                tempdry = float(tempd.strip('M')) * -1.0
            else:
                tempdry = float(tempd) + 0.0
            if 'M' in tempw:
                #print tempw
                tempwet = float(tempw.strip('M')) * -1
            else:
                #print tempw
                tempwet = float(tempw) + 0.0
            if tempdry > 60.0:
                tempdry = 15.0
                tempwet = 10.2
        Astr = re.search('^A\d\d\d\d',l)
        if Astr:
            #print 'We only come here'
            #print Astr.group(0)
            altm = float(Astr.group(0).strip('A'))/100.0
            if altm < 26:
                altm = 0
        Astr = re.search('^Q\d\d\d\d',l)
        if Astr:
            #print 'We never come here'
            #print Astr.group(0)
            altQ = float(Astr.group(0).strip('Q'))
            hCH = float(gps_asos_info.ASOSheight[gps_asos_info.GPSid.values == site].values)
        if tempdry >= -99.0 and tempwet >= -99.0:
            tempK = tempdry + 273.15
            pressWT = 6.112 * (10 ** ((7.5*tempwet)/(237.7+tempwet)))
            pressSV = 6.112 * (10 ** ((7.5*tempdry)/(237.7+tempdry)))
        else:
            tempK = 288.15
        if altm:
            ps1 = altm**0.1903
            ps2 = ps1 - 4.3077E-5*float(gps_asos_info.ASOSheight[gps_asos_info.GPSid.values == site].values)
            press = ps2**5.255
            press *= 33.864
            presG = press * (10.0 ** (0.0148*float(gps_asos_info.ASOS_GPS[gps_asos_info.GPSid == site].values)/tempK))
        elif altQ:
            altFrac = math.exp((9.80665*hCH)/(287.04*tempK))
            ps2 = altQ/altFrac
            presG = ps2 * (10.0**(0.0148*float(gps_asos_info.ASOS_GPS[gps_asos_info.GPSid == site].values)/tempK))
                
        if tempdry >= -99.0 and tempwet >= -99.0:
            relHum = (pressWT/pressSV) * 100.0
        else:
            relHum = -9.9
    return presG,tempdry,relHum

def create_new_met_file(out_file_path,site):
    template_file = '/home/aditya/UMASS/RINEXmetgen/SupportFiles/ssssDDD0.YYm'
    with open(template_file,'r') as f:
        template_data = f.read()
        template_data = template_data.replace('SSSS',site.upper())
        template_data = template_data.replace('YYYY',str(time.gmtime().tm_year))
        template_data = template_data.replace('$$',str(time.gmtime().tm_mon).zfill(2))
        template_data = template_data.replace('%%',str(time.gmtime().tm_mday).zfill(2))
        template_data = template_data.replace('hh',str(time.gmtime().tm_hour).zfill(2))
        template_data = template_data.replace('mm',str(time.gmtime().tm_min).zfill(2))
        with open(out_file_path,'w+') as w: w.write(template_data)

def makeMetRINEX_file(site,wmo_file_name,data,year):   
    
    gps_asos_info = pd.read_csv('/home/aditya/UMASS/RINEXmetgen/SupportFiles/KFWS_GPS_ASOS_locations.csv')    
    out_base = '/home/aditya/UMASS' + os.sep + 'DFWnetDB' + os.sep + gps_asos_info.Network[gps_asos_info.GPSid.values == site].values[0] + os.sep + str(year) + os.sep + 'met'    
    for line in data:

        presG,tempdry,relHum = parse_met_var(site,gps_asos_info,wmo_file_name,line)
        # Check if the file name matches with the file content time stamps
        if wmo_file_name[6:8] == line[1][:2]:
            doy = str(date2doy(int(wmo_file_name[:4]),int(wmo_file_name[4:6]),int(wmo_file_name[6:8]))).zfill(3)
            out_file_path = out_base + os.sep + site  + doy + '0' +'.' + wmo_file_name[2:4] + 'm'
        # If not lets do something
        # What to do????
        else:
            if wmo_file_name[4:6] == '01':
                # Write in previous year last days file
                doy = str(date2doy(int(wmo_file_name[:4]) - 1,12,31)).zfill(3)
                out_file_path = out_base + os.sep + site  + doy + '0' +'.' + str(int(wmo_file_name[2:4]) -1) + 'm'
            #elif wmo_file_name[6:8] == '01':
            else:
                # Write in previous days file
                doy = date2doy(int(wmo_file_name[:4]),int(wmo_file_name[4:6]),int(wmo_file_name[6:8])) -1
                doy = str(doy).zfill(3)
                out_file_path = out_base + os.sep + site  + doy + '0' +'.' + wmo_file_name[2:4] + 'm'
        if not os.path.exists(out_file_path):
            create_new_met_file(out_file_path,site)             
        with open(out_file_path,'a') as w:
            time_stamp = '\n ' + wmo_file_name[2:4].rjust(2) + wmo_file_name[4:6].rjust(3) + line[1][:2].rjust(4)[-3:] + line[1][2:4].rjust(5)[-3:] + line[1][4:6].rjust(6)[-3:] + '00'.rjust(7)[-3:] 
            if isfloat(presG) and isfloat(tempdry) and isfloat(relHum):
                data_to_write = time_stamp + ''.rjust(1) + '%.1f'%presG +  ''.rjust(3) + '%.1f'%tempdry + ''.rjust(3) + '%.1f'%relHum                
                w.write(data_to_write)
            else:
                print '<<Something went wrong>>'
                print 'Site: ' + site + '; ' + 'File name: ' + wmo_file_name
                print presG
                print tempdry
                print relHum
'''
time_stamp = '\n ' + str(int(str(time.gmtime().tm_year)[-2:])).rjust(2)  + 
str((time.gmtime().tm_mon)).rjust(3) + 
str(time.gmtime().tm_mday).rjust(4)[-3:] + 
str(time.gmtime().tm_hour).rjust(5)[-3:] + str(time.gmtime().tm_min).rjust(6)[-3:] 
+ str(time.gmtime().tm_sec).rjust(7)[-3:]
f.write(time_stamp + ''.rjust(1) + pr + ''.rjust(3) + str(ta) + ''.rjust(3)  + str(rh))

'''            
    

#makeMetRINEX_file('cnvl','045','14')
        
      
        
# Parse file

def parse_file_and_dump(fname,year):
    file_name = fname#'2014010100_sao.wmo'
    file_path = '/home/aditya/UMASS/' + str(year) + '_met' + os.sep + file_name
    gps_asos_info = pd.read_csv('/home/aditya/UMASS/RINEXmetgen/SupportFiles/KFWS_GPS_ASOS_locations.csv')    
    mydict = dict(zip(list(gps_asos_info.GPSid.values),list(gps_asos_info.ASOSid.values)))
    all_sites = []
    mydict['txth'] = 'KBKD'
    mydict['txsy'] = 'KCWC'
    with open(file_path,'r') as f:
        for line in f:
            if not line.split():
                continue 
            else:
                all_sites.append(line.split())
    for i in mydict.iterkeys():
        wmo_data = filter(lambda x: x[0] == mydict[i],all_sites)        
        makeMetRINEX_file(i,file_name,wmo_data,year)

year = int(sys.argv[1])
#year = 2015
# This function generates the file names for an entire year
def gen_wmo_file_list(year):
    days_yr = [str(x).zfill(3) for x in range(121,123)]
    FileNameList = [str(doy2date(year,int(x)).tm_year) + str(doy2date(year,int(x)).tm_mon).zfill(2) + str(doy2date(year,int(x)).tm_mday).zfill(2) for x in days_yr]
    FileNameList = [x + '{}'.format(str(y).zfill(2)) + '_sao.wmo' for x in FileNameList for y in range(24)]
    return FileNameList
    

FileNameList = gen_wmo_file_list(year)

print FileNameList

#file_names = ['20140507{0}_sao.wmo'.format(str(x).zfill(2)) for x in range(24)]
#
#file_names.append('2014010508_sao.wmo')

for f in FileNameList:
    parse_file_and_dump(f,year)
