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
    data = DFW.sites
    ftp_noaa = ftplib.FTP('www.ngs.noaa.gov','anonymous','adi@gmail.com')
    ftp_noaa.cwd('/cors/rinex/2015' + os.sep + doy )
    sites = ftp_noaa.nlst()
    initial = os.getcwd()
    data = data[data[:,-1] == net,:]
    for tx_site in data[:,0]:
        if tx_site in sites:
            os.chdir('/home/aditya/UMASS/DFWnetDB' + os.sep + data[data[:,0] == tx_site,-1][0] + '/2015/rinex')
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
    os.chdir('/home/aditya/UMASS/DFWnetDB' + os.sep + net + '/2015')
    subprocess.call(['sh_gamit','-expt',net,'-d','2015',doy,'-orbit','IGSF','-met'])
    print '%%%% Processing for  ' + doy + '  is complete %%%%'

net = sys.argv[1]
#doy_list = [str(x).zfill(3) for x in range(188,212)]
doy_list = ['180','181']
for doy in doy_list:
    download_files(doy,net)
    run_gamit(doy,net)
        
        

    
    
    
    





