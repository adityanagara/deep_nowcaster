# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:19:09 2016

@author: adityanagarajan
This script makes the radar data set for the nowcasting experiments. 
After downloading the files thru bulk order from the NCDC (http://www.ncdc.noaa.gov/nexradinv/)
archive onto a folder, this file will take all the relevent files and DELETE the 
rest. 
"""

import numpy as np
import os
import DFWnet
import subprocess
import ftplib

DFW = DFWnet.CommonData()
Months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

def GetNEXRADfile(mon,day,yr,order_id = 'HAS010640668'):
    file_to_get = 'NWS_NEXRAD_NXL3_KFWS_20' +yr + mon + day + '000000_20' + yr + mon + day + '235959.tar.gz'
    ftp_NEXRAD = ftplib.FTP('ftp.ncdc.noaa.gov','anonymous','adi@gmail.com')  
    ftp_NEXRAD.cwd('pub/has' + os.sep + order_id + os.sep)
    file_list = ftp_NEXRAD.nlst()
    if file_to_get in file_list:
        print 'We Going to get that file: ' + file_to_get
        gfile = open(file_to_get,'wb')
        ftp_NEXRAD.retrbinary('RETR ' + file_to_get,gfile.write)
        gfile.close()
    else:
        print 'FATAL: File not found ' +  file_to_get
    ftp_NEXRAD.close()
    subprocess.call(['tar','-xvzf',file_to_get])
    subprocess.call(['rm',file_to_get])
    
def extract_reflectivity_files(doy,yr):
    ''' This function moves the relevant files and untars them to the 
    specific directiry YYYY/MONDD'''
    # common name of Level 3 reflectivity products at 0.5 VCP
    # KFWD_SDUS54_N0RFWS_201405230003.nc --> may 23rd 2014
    # NWS_NEXRAD_NXL3_KFWS_20140501000000_20140501235959.tar.gz
    # /Users/adityanagarajan/projects/nowcaster/data/RadarData/NEXRAD/2014/raw_files/
    base_path = '../data/RadarData/NEXRAD/20' + str(yr) + '/raw_files/'
    DFW.doytodate(int(yr),doy)
#    print '../data/RadarData/NEXRAD/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + DFW.day
    new_dir = '../data/RadarData/NEXRAD/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + DFW.day
    if not os.path.exists(new_dir):
        file_token = '20' + str(yr) + DFW.mon + DFW.day
        file_list = os.listdir(base_path)
        file_list = filter(lambda x: x[-7:] == '.tar.gz',file_list)
        print file_token
        temp_file = filter(lambda x: file_token in x,file_list)
        print temp_file
        os.mkdir('../data/RadarData/NEXRAD/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + DFW.day)
        subprocess.call(['cp',base_path + temp_file[0],new_dir])
        # Get the initial directory that we were in
        initial = os.getcwd()
        # Move into the directory of the day folder to extract the level3 products
        os.chdir('../data/RadarData/NEXRAD/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + DFW.day + os.sep)
        subprocess.call(['tar','-xzvf',temp_file[0]])
        print os.getcwd()        
        print temp_file[0]
        os.chdir(initial)

def find_closest_radar_data(t,files):
    '''This function finds the closest file to that particular hour'''
    hr = t[:2]
    mn = t[-2:]
    temp_file = filter(lambda x: x[27:29] == hr,files)
    # Find the closest file to the given time stame (up to 5 minutes ahead is fine)
    if mn == '30':
        the_file = filter(lambda x: int(x[-4:][-2:])  >= int(mn) -13  and int(x[-4:][-2:]) <= int(mn) + 13,temp_file)
        the_file.sort(key = lambda x: abs(int(x[-4:][-2:]) - int(mn)))
        
    else:
        temp_file = filter(lambda x: x[-4:][:2] == hr or x[-4:][:2] == str(int(hr) -1).zfill(2),files)       
        the_file = filter(lambda x: 
            int(x[-4:][:2] == str(int(hr) -1).zfill(2) and int(x[-4:][-2:]) >= 50  or (x[-4:][:2] == hr and int(x[-4:][-2:])  >= int(mn)  and int(x[-4:][-2:]) <= int(mn) + 20)),
                temp_file)
        the_file.sort(key = lambda x: abs((float(x[-4:][:2]) + float(x[-4:][-2:])/60.0) - int(hr) ))
    file_to_return = []
    if len(the_file) > 0:
        file_to_return = the_file[0]
    else:
        print 'WARNING: Missing File for time --> '  + t
    return file_to_return

def KeepRequiredFiles(doy,yr):
    DFW.doytodate(int(yr),doy)
    file_path = '../data/RadarData/NEXRAD/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + DFW.day
    file_list = os.listdir(file_path)
    files_to_keep = []
    time_index = ['{0}{1}'.format(str(x).zfill(2),str(y).zfill(2)) for x in range(24) for y in [0,30]]
    delete_files = filter(lambda x: x[:18] != 'KFWD_SDUS54_N0RFWS',file_list)
    # We only want the scan at the 0.5deg VCP, thus we are goint to delete the rest
    for df in delete_files:
        os.remove(file_path + os.sep + df)
    file_list = filter(lambda x: x[:18] == 'KFWD_SDUS54_N0RFWS',file_list)
    # We are noe going to find the closest file to 00 and 30 and obtain 48 such 
    # nexrad files
    
#    for t in range(48):
#        files_to_keep.append(find_closest_radar_data(time_index[t],file_list))
    return file_list

def Deletefiles(doy,yr,keep_files):
    '''Delete all the files which are not required'''
    DFW.doytodate(int(yr),doy)
    file_path = '../data/RadarData/NEXRAD/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + DFW.day + os.sep
    file_list = os.listdir(file_path)
    for f in file_list:
        if f not in keep_files:
            print file_path + f
            os.remove(file_path + f)

def ConvertToNETCDF(doy,yr,keep_files):
    '''Use the java toolsUI-4.6.jar to convert to .nc files'''
    DFW.doytodate(int(yr),doy)
    java_script = 'toolsUI-4.6.jar'
    ucar = 'ucar.nc2.FileWriter'
    file_path = '../data/RadarData/NEXRAD/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + DFW.day + os.sep
    for raw_file in keep_files:
        temp_in = file_path + raw_file
        temp_out = file_path + raw_file + '.nc'
        subprocess.call(['java','-classpath',java_script,ucar,'-in',temp_in,'-out',temp_out,])
        # remove the raw file we only need .nc
        os.remove(file_path + raw_file)

def main(yr):
    initial = os.getcwd()
    order_dict = {}
    order_dict[14] = 'HAS010756704'
    order_dict[15] = 'HAS010756706'
    if yr == 14:
        storm_dates = np.load('../data/storm_dates_2014.npy').astype('int')
        # The following 3 dates are going to be removed because we do not
        # have NEXRAD files for the entire day            
        idx1 = np.where(np.all(storm_dates == [205,  14,   7,  24],axis=1))[0][0]
        storm_dates = np.delete(storm_dates,idx1,axis = 0)
        idx2 = np.where(np.all(storm_dates == [176,  14,   6,  25],axis=1))[0][0]
        storm_dates = np.delete(storm_dates,idx2,axis = 0)
        idx3 = np.where(np.all(storm_dates == [204 , 14 ,  7 , 23],axis=1))[0][0]
        storm_dates = np.delete(storm_dates,idx3,axis = 0)
    elif yr == 15:
        storm_dates = np.load('../data/storm_dates_2015.npy').astype('int')
    for d in storm_dates:
        DFW.doytodate(int(yr),d[0])
        new_dir = '../data/RadarData/NEXRAD/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + DFW.day
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        os.chdir(new_dir)
        GetNEXRADfile(DFW.mon,DFW.day,DFW.yr,order_id = order_dict[yr])
        os.chdir(initial)
        nexrad_files = KeepRequiredFiles(d[0],yr)
#        Deletefiles(d[0],yr,nexrad_files)
#        DFW.doytodate(int(yr),d[0])
#        file_path = '../data/RadarData/NEXRAD/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + DFW.day + os.sep
#        nexrad_files = os.listdir(file_path)
        ConvertToNETCDF(d[0],yr,nexrad_files)

'''
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

'''

if __name__ == '__main__':
    yr = [14,15]
    for y in yr:
        main(y)