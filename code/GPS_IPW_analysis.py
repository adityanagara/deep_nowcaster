# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:48:20 2016

@author: adityanagarajan
This script plots a histogram of the difference in ipw values firstly. 
One of the things we need to do is filter out stations with a very large difference 
in the ipw values and then take the monthly averages. 
"""

import numpy as np
from matplotlib import pyplot as plt
import DFWnet


def main():
    x = DFWnet.CommonData()
    
    ipw_2014 = x.IPWvals_2014
    
    print ipw_2014.shape
    
#    plt.figure()
    
    
    
#    plt.plot(ipw_2014[:,0,1:].reshape(365*48))
    
    for i in range(2):
        stn_diff = np.diff(ipw_2014[:,i,1:].reshape(365*48).astype('float'))
    
        stn_diff[np.isnan(stn_diff)] = 0.
    
        stn_diff = [abs(num) for num in stn_diff]
        
        counts,bins = np.histogram(stn_diff)
        
        bins1 = bins[:-1] + np.diff(bins) / 2.
        
        print bins1,counts/(counts.sum()*1.)
        plt.figure()
        plt.bar(bins1,counts/(counts.sum()*1.),width = np.diff(bins1)[0])
        plt.show()

#    plt.show()

    


if __name__ == '__main__':
    main()
    