# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:49:37 2015

@author: adityanagarajan
"""



import os
import numpy as np
from matplotlib import pyplot as plt
import re


class BuildNowcaster():
    def __init__(self):
        self.base_dir = 'data/dataset/'
        self.domain_points = (range(17,83),range(17,83))
        
    def sort_filter_files(self,PixelX,PixelY,doy_set):
        file_list = os.listdir(self.base_dir)
        file_list = filter(lambda x: x[-4:] == '.npy',file_list)


        PixelPoints = [(x,y) for x in PixelX for y in PixelY]

        PixelPoints = np.array(PixelPoints)

        # Filter top left of the domain
        domain_list = filter(lambda x: int(re.findall('\d+',x)[0]) in np.unique(PixelPoints[:,0]) and int(re.findall('\d+',x)[1]) in np.unique(PixelPoints[:,1]),file_list)
        

        domain_list.sort(key = lambda x: int(x[-7:-4]))

        ipw_files = filter(lambda x: x[:3] == 'IPW',domain_list)

        radar_files = filter(lambda x: x[:5] == 'Radar',domain_list)
    

        ipw_files = filter(lambda x: x[-7:-4] == doy_set,ipw_files)

        radar_files = filter(lambda x: x[-7:-4] == doy_set,radar_files)
    
        return ipw_files,radar_files
    
    def plot_domain(self,PixelPoints,marker = 'r*'):
        
        gridX = np.arange(-150.0,151.0,300.0/(100-1))
        gridY = np.arange(-150.0,151.0,300.0/(100-1))
        # Loop through each pair to plot on the grod
        for p in PixelPoints:
            plt.plot(gridX[p[0]],gridY[p[1]],marker)

        plt.xlabel('Easting')
    
        plt.ylabel('Northing')

        plt.xlim((-150.0,150.0))

        plt.ylim((-150.0,150.0))
        plt.grid()
    
    def make_prediction(self,doy):
        



