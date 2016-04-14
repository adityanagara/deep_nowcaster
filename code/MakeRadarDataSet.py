# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:43:19 2015

@author: adityanagarajan
"""

import os
import DFWnet
import ftplib
import numpy as np
import pandas as pd
import subprocess


def get_IPW_vals(doy):
    
    IPWvals = np.load('data/2014Data_update1.npy')
    return IPWvals[doy -1,:,:]


# Function which returns a matrix of normalized values (Normalization done wrt to month)
def NormalizeIPW_Normal(DAT,doy):
    DFW = DFWnet.CommonData()
    
    DFW.doytodate(14,doy)
    
    SummaryStats = pd.read_csv('output/SitesSummary.csv',index_col=0)
    
    IPWdataN = np.array(map(lambda x: (x[1:].astype('float') - SummaryStats[SummaryStats.site.values == x[0]][DFW.mon + '_Avg'].values)/SummaryStats[SummaryStats.site.values == x[0]][DFW.mon + '_Std'].values,DAT))
    
    return IPWdataN.astype('float')


# Function that finds radar data closest to the given GPS met data
def find_closest_radar_data(t,files):
    
    hr = t[:2]
    mn = t[-2:]

    temp_file = filter(lambda x: x[-4:][:2] == hr,files)
    
    
    # Find the closest file to the given time stame (up to 5 minutes ahead is fine)
    if mn == '30':
        the_file = filter(lambda x: int(x[-4:][-2:])  >= int(mn) -13  and int(x[-4:][-2:]) <= int(mn) + 13,temp_file)
#        diff = map(lambda x: abs(int(x[-4:][-2:]) - int(mn)),the_file)
        the_file.sort(key = lambda x: abs(int(x[-4:][-2:]) - int(mn)))
        
    else:
        temp_file = filter(lambda x: x[-4:][:2] == hr or x[-4:][:2] == str(int(hr) -1).zfill(2),files)       
        the_file = filter(lambda x: 
            int(x[-4:][:2] == str(int(hr) -1).zfill(2) and int(x[-4:][-2:]) >= 50  or (x[-4:][:2] == hr and int(x[-4:][-2:])  >= int(mn)  and int(x[-4:][-2:]) <= int(mn) + 10)),
                temp_file)
        the_file.sort(key = lambda x: abs((float(x[-4:][:2]) + float(x[-4:][-2:])/60.0) - int(hr) ))
        
#        diff = map(lambda x: abs((float(x[-4:][:2]) + float(x[-4:][-2:])/60.0) - int(hr) ),the_file)
    
    file_to_return = []
    if len(the_file) > 0:
        file_to_return = the_file[0]
    else:
        print 'WARNING: Missing File for time --> '  + t
    return file_to_return

def GetNEXRADfile(mon,day):
    file_to_get = 'NWS_NEXRAD_NXL3_KFWS_2014' + mon + day + '000000_2014' + mon + day + '235959.tar.gz'
    
    ftp_NEXRAD = ftplib.FTP('ftp.ncdc.noaa.gov','anonymous','adi@gmail.com')  
    ftp_NEXRAD.cwd('pub/has/HAS010640668/')
    file_list = ftp_NEXRAD.nlst()
    if file_to_get in file_list:
        print 'We Going to get that file: ' + file_to_get
        gfile = open(file_to_get,'wb')
        ftp_NEXRAD.retrbinary('RETR ' + file_to_get,gfile.write)
        gfile.close()
    else:
        print 'FATA: File not found ' +  file_to_get
    
    ftp_NEXRAD.close()
    subprocess.call(['tar','-xvzf',file_to_get])
    subprocess.call(['rm',file_to_get])


def KeepRequiredFiles(file_path):
    file_list = os.listdir(file_path)
    files_to_keep = []
    time_index = ['{0}{1}'.format(str(x).zfill(2),str(y).zfill(2)) for x in range(24) for y in [0,30]]
    file_list = filter(lambda x: x[:18] == 'KFWD_SDUS54_N0RFWS',file_list)
    
    for t in range(48):
        files_to_keep.append(find_closest_radar_data(time_index[t],file_list))
    
    return files_to_keep

def Deletefiles(file_path,keep_files):
    file_list = os.listdir(file_path)
    for f in file_list:
        if f not in keep_files:
            os.remove(file_path + f)

def ConvertToNETCDF(file_path,keep_files):
    java_script = '/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/code/toolsUI-4.6.jar'
    ucar = 'ucar.nc2.FileWriter'
    files = keep_files
    for raw_file in files:
        temp_in = file_path + raw_file
        temp_out = file_path + raw_file + '.nc'
        subprocess.call(['java','-classpath',java_script,ucar,'-in',temp_in,'-out',temp_out,])




'''
NWS_NEXRAD_NXL3_KFWS_20140706000000_20140706235959.tar.gz
'''

DFW = DFWnet.CommonData()
Anomalies = []
Months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
#for doy in range(1,366):
#    
#    IPWvals = get_IPW_vals(doy)
#    
#    IPWvals = NormalizeIPW_Normal(IPWvals,doy)
#    
#    tempArr = np.where(IPWvals > 2.0)
#    
#    if tempArr[0].size > 30:
#        
#        DFW.doytodate(14,doy)
#        dirToMake = Months[int(DFW.mon) - 1] + DFW.day
#        print 'Weather anomaly on ' + DFW.mon + '/' + DFW.day + '/' + '2014'
#        print dirToMake
#        
##        if not os.path.exists('data/RadarData/NEXRAD/' + dirToMake):
##            os.mkdir('data/RadarData/NEXRAD/' + dirToMake)
##        initial = os.getcwd()
##        
##        os.chdir('data/RadarData/NEXRAD/' + dirToMake)
##        
##        GetNEXRADfile(DFW.mon,DFW.day)
##        f1 = KeepRequiredFiles(Months[int(DFW.mon) - 1],'11')
##        Deletefiles(f1)
##        ConvertToNETCDF(f1)
##        
##        os.chdir(initial)
#        Anomalies.append([DFW.mon,DFW.day])

#AnomaliesExtra = np.array((['05','09'],['05','13'],['05','24'],['05','26']))
#
#Anomalies = AnomaliesExtra

#Anomalies = [['05','09'],['05','13'],['05','24'],['05','26']]
#
#print Anomalies
#
#for a in Anomalies:
#    dirToMake = Months[int(a[0]) -1 ] + a[1]
#    print 'Making directory: ' + dirToMake
#    if not os.path.exists('data/RadarData/NEXRAD/' + dirToMake):
#        os.mkdir('data/RadarData/NEXRAD/' + dirToMake)
#    
#    initial = os.getcwd()
#        
#    os.chdir('data/RadarData/NEXRAD/' + dirToMake)
#        
#    GetNEXRADfile(a[0],a[1])
#    f1 = KeepRequiredFiles()
#    Deletefiles(f1)
#    ConvertToNETCDF(f1)
#        
#    os.chdir(initial)
    
file_path = '/Users/adityanagarajan/projects/nowcaster/data/RadarData/NEXRAD/2015/MAY08/'
f1 = KeepRequiredFiles(file_path)
print f1
Deletefiles(file_path,f1)
ConvertToNETCDF(file_path,f1)









