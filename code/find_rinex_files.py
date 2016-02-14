# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:52:21 2015

@author: aditya
"""

import numpy as np
import ftplib
import os
import sys
import subprocess

import DFWnet

import shutil


def download_files(doy,net):
    
    print doy
    DFW = DFWnet.CommonData()
#    data = np.loadtxt('/home/aditya/UMASS/RINEXmetgen/SupportFiles/KFWS_GPS_ASOS_locations.csv',dtype='S',delimiter = ',',skiprows = 1)
    data = DFW.sites
    
#    doy = sys.argv[1]

    ftp_noaa = ftplib.FTP('www.ngs.noaa.gov','anonymous','adi@gmail.com')

    ftp_noaa.cwd('/cors/rinex/2015' + os.sep + doy )

    sites = ftp_noaa.nlst()

    initial = os.getcwd()
    
#    os.chdir('/home/aditya/UMASS/DFWnet/net4/2015/rinex')
    
    data = data[data[:,-1] == net,:]
    
    

    for tx_site in data[:,0]:
        if tx_site in sites:
#            print data[data[:,0] == tx_site,-1][0]
#            print os.getcwd()
            os.chdir('/home/aditya/UMASS/DFWnet' + os.sep + data[data[:,0] == tx_site,-1][0] + '/2015/rinex')
        
            ftp_noaa.cwd('/cors/rinex/2015' + os.sep + doy + os.sep + tx_site)
            
            dyna_file_name =   tx_site + doy + '0' + '.' + '15o.gz'
            print dyna_file_name
            
            siteList = ftp_noaa.nlst()
            print siteList
            print dyna_file_name in siteList
            if dyna_file_name in siteList:
                
                gfile = open(dyna_file_name,'wb')
                ftp_noaa.retrbinary('RETR ' + dyna_file_name,gfile.write)
                gfile.close()
                print 'RINEX obs found site: ' + tx_site

    os.chdir(initial)
    ftp_noaa.close()

#def download_met_files(doy):
#    
#
#    
#    sites = ['ac20','conz','p019','unbj']
#    
#    ftp_sopac = ftplib.FTP('garner.ucsd.edu','anonymous','adi@gmail.com')
#    
#    ftp_sopac.cwd('/archive/garner/met/2014' + os.sep + doy)
#    
#    initial = os.getcwd()
#    
#    os.chdir('/home/aditya/UMASS/DFWnet' + os.sep + 'net1' + '/2014/met')
#    check_Sites = ftp_sopac.nlst()
#    check_Sites = map(lambda x: x[:4],check_Sites)
#    for s in sites:
#        dyna_file_name = s + doy + '0' + '.14m.Z'
#        if s in check_Sites:
#            print 'Met file found Found ' + s
#        
#        
#            gfile = open(dyna_file_name,'wb')
#            ftp_sopac.retrbinary('RETR ' + dyna_file_name,gfile.write)
#            gfile.close()
#            print 'found file: ' + s
#            shutil.copy2('/home/aditya/UMASS/DFWnet' + os.sep + 'net1' + '/2014/met/' + dyna_file_name,'/home/aditya/UMASS/DFWnet' + os.sep + 'net2' + '/2014/met/')
#        
#            shutil.copy2('/home/aditya/UMASS/DFWnet' + os.sep + 'net1' + '/2014/met/' + dyna_file_name,'/home/aditya/UMASS/DFWnet' + os.sep + 'net3' + '/2014/met/')
#        
#            shutil.copy2('/home/aditya/UMASS/DFWnet' + os.sep + 'net1' + '/2014/met/' + dyna_file_name,'/home/aditya/UMASS/DFWnet' + os.sep + 'net4' + '/2014/met/')
#            
#            print 'Uncompressing files'
#                
#            subprocess.call(['uncompress','-f','/home/aditya/UMASS/DFWnet' + os.sep + 'net1' + '/2014/met/' + dyna_file_name])
#            subprocess.call(['uncompress','-f','/home/aditya/UMASS/DFWnet' + os.sep + 'net2' + '/2014/met/' + dyna_file_name])
#            subprocess.call(['uncompress','-f','/home/aditya/UMASS/DFWnet' + os.sep + 'net3' + '/2014/met/' + dyna_file_name])
#            subprocess.call(['uncompress','-f','/home/aditya/UMASS/DFWnet' + os.sep + 'net4' + '/2014/met/' + dyna_file_name])
#        else:
#            print 'Met file not found ' + s
#    os.chdir(initial)
    
        
#doy = sys.argv[1]

#def run_gamit(doy):
#    
#    os.chdir('/home/aditya/UMASS/DFWnet' + os.sep + 'net1' + '/2014')
#    
#    subprocess.call(['sh_gamit','-expt','net1','-d','2014',doy,'-orbit','IGSF','-met'])
#    print '%%%% Prodessing for  ' + doy + '  is complete %%%%'
    
#doy = '003'
    
def split_files():
    with open('MissingData2.txt','rb') as f:
        getFiles = f.read()
    getFiles = getFiles.split('\n')
    temp = [item.split(':') for item in getFiles[:4]]
    temp_dict = dict((k.strip(),v.strip()) for k,v in temp)
    for n in temp_dict.keys():
        
        temp_dict[n] = [k.strip() for k in temp_dict[n].strip('[]').replace("'"," ").strip().split(',')]
        
    return temp_dict

def run_gamit(doy,net):
    
    os.chdir('/home/aditya/UMASS/DFWnet' + os.sep + net + '/2014')
    
    subprocess.call(['sh_gamit','-expt',net,'-d','2014',doy,'-orbit','IGSF','-met'])
    print '%%%% Processing for  ' + doy + '  is complete %%%%'


    
#file_dict = split_files()

net = sys.argv[1]

#doy_list = [str(x).zfill(3) for x in range(365,366)]
#doy_list = file_dict[net]

doy = sys.argv[2]

download_files(doy,net)
#for doy in doy_list:
#     download_files(doy,net)
#    download_met_files(doy)
#    run_gamit(doy)
        
        
        
        
    
    
    
    
    





