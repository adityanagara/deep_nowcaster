# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:31:11 2016

@author: adityanagarajan
"""

import os
import numpy as np
import pandas as pd
import ftplib
from StringIO import StringIO
import math
import re
import DFWnet
import time

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


sensors = pd.read_csv('/home/aditya/UMASS/RINEXmetgen/SupportFiles/KFWS_GPS_ASOS_locations_new.csv')

DFW = DFWnet.CommonData()

def parse_met_var(site,GPS_height,ASOS_height,ASOS_GPS,line):
    presG = ''
    tempd = ''
    tempw = ''
    altm = 0
    altQ = 0
    tempdry = -99.9
    tempwet = -99.9
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
                tempdry = float(tempd)
            if 'M' in tempw:
                #print tempw
                tempwet = float(tempw.strip('M')) * -1.0
            else:
                #print tempw
                tempwet = float(tempw) 
            if tempdry > 60.0:
                tempdry = 15.0
                tempwet = 10.2
            # convert temperature to MSL
            tempdry = tempdry + 273.16 + (0.0065 * ASOS_height * 0.001)
            tempdry = tempdry - 273.16 - (0.0065 * GPS_height * 0.001)
            # convert MSL temperature to GPS station temp
        Astr = re.search('^A\d\d\d\d',l)
        if Astr:
            #print 'We only come here'
            #print Astr.group(0)
            altm = float(Astr.group(0).strip('A'))/100.0
            if altm < 26.0:
                altm = 0.0
        Astr = re.search('^Q\d\d\d\d',l)
#        ASOS_height = float(gps_asos_info.newASOSheight[gps_asos_info.GPSid.values == site].values)
#        ASOS_GPS = float(gps_asos_info.newASOS_GPS[gps_asos_info.GPSid == site].values)
        if Astr:
            #print 'We never come here'
            #print Astr.group(0)
            altQ = float(Astr.group(0).strip('Q'))
            hCH = ASOS_height
        if tempdry >= -99.0 and tempwet >= -99.0:
            tempK = tempdry + 273.15
            pressWT = 6.112 * (10 ** ((7.5*tempwet)/(237.7+tempwet)))
            pressSV = 6.112 * (10 ** ((7.5*tempdry)/(237.7+tempdry)))
        else:
            tempK = 288.15
        if altm:
            ps1 = altm**0.1903
            ps2 = ps1 - 4.3077E-5*ASOS_height
            press = ps2**5.255
            press *= 33.864
            presG = press * (10.0 ** (0.0148*ASOS_GPS/tempK))
        elif altQ:
            altFrac = math.exp((9.80665*hCH)/(287.04*tempK))
            ps2 = altQ/altFrac
            presG = ps2 * (10.0**(0.0148*ASOS_GPS/tempK))
                
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


def makeMetRINEX_file(site,time,presG,tempdry,relHum,year,doy):   
    out_base = '/home/aditya/UMASS' + os.sep + 'DFWnetwork' + os.sep + sensors.Network[sensors.GPSid.values == site].values[0] + os.sep + '20' + str(year) + os.sep + 'met'    
    out_file_path = out_base + os.sep + site  + str(doy) + '0' +'.' + str(year) + 'm'
    if not os.path.exists(out_file_path):
        create_new_met_file(out_file_path,site)             
    with open(out_file_path,'a') as w:
        time_stamp = '\n ' + time[0].rjust(2) + time[1].rjust(3) + time[2].rjust(4)[-3:] + time[3].rjust(5)[-3:] + time[4].rjust(6)[-3:] + '00'.rjust(7)[-3:] 
        if isfloat(presG) and isfloat(tempdry) and isfloat(relHum) and tempdry >= -90.0:
            data_to_write = time_stamp + ''.rjust(1) + '%.1f'%presG +  ''.rjust(3) + '%.1f'%tempdry + ''.rjust(3) + '%.1f'%relHum                
            w.write(data_to_write)
        else:
            print '<<Something went wrong>>'
            print 'Site: ' + site + '; ' + 'File name: '
            print presG
            print tempdry
            print relHum

def download_ASOS_files(site,yr = '16'):
    ftp_NCDC = ftplib.FTP('ftp.ncdc.noaa.gov','anonymous','adi@gmail.com')
    ftp_NCDC.cwd('/pub/data/asos-fivemin/6401-20' + yr)
    file_list = ftp_NCDC.nlst()
    #file_list = filter(lambda x: x[5:9] == 'KDAL',file_list)
    file_list = ['64010{0}20{1}{2}.dat'.format(site.newASOSid,yr,str(x).zfill(2)) for x in range(4,8)]
    print file_list
    # Loop through all times in the data set May 01 00:00 - August 31 23:55 UTC
    # From May 1st to September 1st
    DFW.date2doy(int(yr),5,1)
    start_day = DFW.doy
    DFW.date2doy(int(yr),9,1)
    end_day = DFW.doy
    # Make a list of calendar days
    calendar_days = []
    for doy in range(start_day,end_day):
        DFW.doytodate(int(yr),doy)
        calendar_days.append((DFW.yr,DFW.mon,DFW.day,doy))
    print 'Processing site: '
    print site
    for day in calendar_days:
        print day
        # loop thru time but keep in mind to grab from local time UTC -6
        if day[2] == '01':
            # need two files in this case
            file_list = ['64010{0}20{1}{2}.dat'.format(site.newASOSid,yr,str(int(day[1]) -1).zfill(2)),'64010{0}20{1}{2}.dat'.format(site.newASOSid,yr,day[1])]
            assert len(file_list) == 2,'Some file is missing..'
            temp_file = StringIO()
            ftp_NCDC.retrbinary('RETR '+ file_list[0],temp_file.write)
            temp_file = temp_file.getvalue()
            temp_file = temp_file.split('\n')
            DFW.doytodate(int(yr),day[3] -1)
            previous_day = DFW.mon + os.sep + DFW.day + os.sep + DFW.yr
            temp_file = filter(lambda x: x.split()[1][-8:] == previous_day,temp_file[:-1])
            # filter out UTC next day stuff
            temp_file = filter(lambda x: x.split()[5][:2] == day[2],temp_file)
            for line in temp_file:
                line = line.split()
                time = line[5]
                time_step = [day[0],day[1],day[2],time[2:4],time[4:6]]
                P,T,RH = parse_met_var(site.GPSid,site.GPSheight,site.newASOSheight,site.newASOS_GPS,line)
                makeMetRINEX_file(site.GPSid,time_step,P,T,RH,yr,day[3])
            temp_file = StringIO()
            ftp_NCDC.retrbinary('RETR '+ file_list[1],temp_file.write)
            temp_file = temp_file.getvalue()
            temp_file = temp_file.split('\n')
            DFW.doytodate(int(yr),day[3])
            current_day = DFW.mon + os.sep + DFW.day + os.sep + DFW.yr
            temp_file = filter(lambda x: x.split()[1][-8:] == current_day,temp_file[:-1])
            temp_file = filter(lambda x: x.split()[5][:2] == day[2],temp_file)
            for line in temp_file:
                line = line.split()
                time = line[5]
                time_step = [day[0],day[1],day[2],time[2:4],time[4:6]]
                P,T,RH = parse_met_var(site.GPSid,site.GPSheight,site.newASOSheight,site.newASOS_GPS,line)
                makeMetRINEX_file(site.GPSid,time_step,P,T,RH,yr,day[3])
        else:
            # only one file required
            file_name = '64010{0}20{1}{2}.dat'.format(site.newASOSid,yr,day[1])
            temp_file = StringIO()
            ftp_NCDC.retrbinary('RETR '+ file_name,temp_file.write)
            temp_file = temp_file.getvalue()
            temp_file = temp_file.split('\n')
            DFW.doytodate(int(yr),day[3] -1)
            previous_day = DFW.mon + os.sep + DFW.day + os.sep + DFW.yr
            temp_file = filter(lambda x: x.split()[1][-8:] == previous_day,temp_file[:-1])
            # filter out UTC next day stuff
            temp_file = filter(lambda x: x.split()[5][:2] == day[2],temp_file)
            for line in temp_file:
                line = line.split()
                time = line[5]
                time_step = [day[0],day[1],day[2],time[2:4],time[4:6]]
                P,T,RH = parse_met_var(site.GPSid,site.GPSheight,site.newASOSheight,site.newASOS_GPS,line)
                makeMetRINEX_file(site.GPSid,time_step,P,T,RH,yr,day[3])
            temp_file = StringIO()
            ftp_NCDC.retrbinary('RETR '+ file_name,temp_file.write)
            temp_file = temp_file.getvalue()
            temp_file = temp_file.split('\n')
            DFW.doytodate(int(yr),day[3])
            current_day = DFW.mon + os.sep + DFW.day + os.sep + DFW.yr
            temp_file = filter(lambda x: x.split()[1][-8:] == current_day,temp_file[:-1])
            temp_file = filter(lambda x: x.split()[5][:2] == day[2],temp_file)
            for line in temp_file:
                line = line.split()
                time = line[5]
                time_step = [day[0],day[1],day[2],time[2:4],time[4:6]]
                P,T,RH = parse_met_var(site.GPSid,site.GPSheight,site.newASOSheight,site.newASOS_GPS,line)
                makeMetRINEX_file(site.GPSid,time_step,P,T,RH,yr,day[3])
    ftp_NCDC.close()

'''
64010K12N201602.dat
'''
#download_ASOS_files(sensors.ix[9,:])

yr = 14
for site in range(sensors.shape[0]):
    download_ASOS_files(sensors.ix[site,:],str(yr))


