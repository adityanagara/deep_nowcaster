# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 18:42:21 2016

@author: adityanagarajan
"""

from matplotlib import pyplot as plt
import numpy as np




def main():
    base_path = '../data/dataset/2015/'
    
    ipw_img = 'IPWdata15_168_25_img.npy'
    
    refl_img = 'RadarRefl15_168_25_img.npy'
    
    ipw_arr = np.load(base_path + ipw_img)
    
    refl_arr = np.load(base_path + refl_img)
    
    
    plt.figure()
    plt.imshow(ipw_arr,origin = 'lower',cmap = 'gray')
    
    plt.figure()
    plt.imshow(refl_arr,origin = 'lower',cmap = 'gray')
    
    plt.show()

if __name__ == '__main__':
    main()