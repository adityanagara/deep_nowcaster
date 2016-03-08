# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:11:23 2016

@author: adityanagarajan
Script to convert from the old 2014 data format for .npy files 
to the new format which includes the years. 
"""
import os
import numpy as np
import shutil 
file_path = 'data/dataset/'
file_path_out = 'data/TrainTest/'
npy_files = os.listdir(file_path)
print npy_files
npy_files = filter(lambda x: x[-4:] == '.npy',npy_files)
print npy_files
for npy_file in npy_files:
    if npy_file[:9] == 'RadarRefl':
        new_file = npy_file[:9] + '14' + npy_file[9:]
        print new_file
    elif npy_file[:7] == 'IPWdata':
        new_file = npy_file[:7] + '14' + npy_file[7:]
        print new_file
    shutil.move(file_path + npy_file,file_path_out + new_file)




