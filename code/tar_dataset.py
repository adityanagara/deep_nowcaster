# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:25:24 2016

@author: adityanagarajan
"""

import os
import tarfile





def main(yr):
    base_path = '../data/dataset/20' + str(yr) + os.sep
    file_list = os.listdir(base_path)
    file_list = filter(lambda x: 'img' in x,file_list)
    print file_list
    
    
    

if __name__ == '__main__':
    yrs = [14,15]
    for y in yrs:
        main(y)