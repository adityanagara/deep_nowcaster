#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      GPSMET1
#
# Created:     04/11/2014
# Copyright:   (c) GPSMET1 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import subprocess
import os
import time
import shutil

def alpha_dict():
    alpha = {}
    alpha_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    for i in range(0,26):
        alpha[i] = alpha_list[i]
    return alpha

def get_file_no():
    dir = 'C:\Daily_GPS_Data' + os.sep + '20140804'
    list = os.listdir(dir)
    num_list = []
    for file in list:
        index = file.index('.')
        num_list.append(file[4:index])
    num_list = [int(i) for i in num_list]
    this_file = max(num_list) - 1
    return num_list,this_file

def call_rinex_coms(output_file_o):
    initial = os.getcwd()
    os.chdir('C:\Users\GPSMET1\Desktop\RINEX\RinexSLX_v2.8.3')
    all_files,current_file =get_file_no()
    stn='cnvl'
    input_file = 'C:\Daily_GPS_Data' + os.sep + '20140804' + os.sep + stn + str(current_file) + '.bin'
    #output_file_o='C:\Daily_GPS_Data' + os.sep + '20140804' + os.sep + stn + str(current_file) + '.14o'
    output_fine_n='C:\Daily_GPS_Data' + os.sep + '20140804' + os.sep + stn + str(current_file) + '.' + str(time.gmtime().tm_year)[-2:] + 'n'
    proc = subprocess.Popen(['RinexSLX', '-R', '-P1', input_file,output_file_o,output_fine_n ],stdout = subprocess.PIPE,stderr = subprocess.PIPE)
    time.sleep(10)
    out,error = proc.communicate()
    proc.terminate()
    print output_file_o
    print output_fine_n
    print 'processes terminated'

def check_RINEX():
    file_alpha=alpha_dict()
    all_files,current_file =get_file_no()
    input_file= 'C:\Daily_GPS_Data' + os.sep + '20140804' + os.sep + 'cnvl' + str(current_file) + '.' + str(time.gmtime().tm_year)[-2:] +  'o'
    if os.path.exists(input_file):
        print 'File Exists'
        #print input_file
    else:
        #print input_file
        call_rinex_coms(input_file)
        print 'force create the file'


check_RINEX()