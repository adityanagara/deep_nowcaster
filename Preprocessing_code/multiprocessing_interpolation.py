# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 09:53:00 2016

@author: adityanagarajan
"""

import multiprocessing
import time
import numpy as np
import DFWnet
from matplotlib import pyplot as plt


DFW = DFWnet.CommonData()

IPWvals = np.load('../data/2014IPW_data.npy')
IPWvals_day = IPWvals[127,:,1:].astype('float')

def get_GPS_cartesian(IPWsites):
    
    lat0 = (np.pi/180.0)*DFW.KFWSlat
    long0 = (np.pi/180.0)*DFW.KFWSlong

    gpsX = np.zeros((IPWsites.shape[0],1))
    gpsY = np.zeros((IPWsites.shape[0],1))
    R = 6378.137

    for si in range(IPWsites.shape[0]):
        lat1 = (np.pi/180.0)*float(IPWsites[si,1])
        long1 = (np.pi/180.0)*float(IPWsites[si,2])
    # Get cartesian x distance from KFWS to GPS site
    
    
        gpsX[si] = R * np.arccos(np.cos(np.pi/2 - lat0) * np.cos(np.pi/2 - lat0) + 
                np.sin(np.pi/2-lat0) * np.sin(np.pi/2-lat0) * np.cos(long1-long0))
    
        # Get cartesian y distance from KFWS to GPS site
        gpsY[si] = R * np.arccos(np.cos(np.pi/2 - lat0) * np.cos(np.pi/2 - lat1) + 
        np.sin(np.pi/2 -lat0) * np.sin(np.pi/2 - lat1) * np.cos(long0 - long0))
    
        if (lat1 - lat0) > 0 and (long1 - long0 ) > 0:
            # First quadrant
            continue
        elif (lat1 - lat0) < 0 and (long1 - long0 ) > 0:
            gpsY[si] = - gpsY[si]
        elif (lat1 - lat0) < 0 and (long1 - long0 ) < 0:
            gpsY[si] = - gpsY[si]
            gpsX[si] = - gpsX[si]
        else:
            gpsX[si] = - gpsX[si]
        
    return gpsX,gpsY,IPWsites



def get_Mulit_Quadratic_Weights(gpsX,gpsY,IPWsites,doy,yr,saveWeights = True):
    n = IPWsites.shape[0]
    D = np.zeros((n,n))
    m = 100
    for i in range(n):
        xi = gpsX[i]
        yi = gpsY[i]
        for j in range(n):
            xj = gpsX[j]
            yj = gpsY[j]
            D[i,j] = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    from numpy.linalg import inv
    
    invD = inv(D)
    # Make the 150x150 km2 grid
    gridX = np.arange(-150.0,151.0,300.0/(m-1))
    gridY = np.arange(-150.0,151.0,300.0/(m-1))

    W = np.zeros((m,m,n))

    print 'Inverse done'

    start_time = time.time()
    
    for gx in xrange(m):
        x0 = gridX[gx]
        for gy in xrange(m):
            y0 = gridY[gy]
            for j in xrange(n):
                wj = 0.0
                for i in xrange(n):
                    xi = gpsX[i]
                    yi = gpsY[i]
                    d0i = np.sqrt((x0 - xi) ** 2 + (y0 - yi) ** 2)
                    wj += invD[i,j]*d0i
                W[gx,gy,j] = wj
        
    end_time = time.time()
    print 'Time taken to generate weights 100x100x44 weights matrix  = %.2fs'%(end_time-start_time)
    np.save('weights.npy',W)
    return W   
m = 100

gridX = np.arange(-150.0,151.0,300.0/(m-1))
gridY = np.arange(-150.0,151.0,300.0/(m-1))
n = DFW.sites.shape[0]

D = np.zeros((n,n))
gpsX,gpsY,IPWsites,IPWvals_day = get_GPS_cartesian(IPWvals_day,DFW.sites)

for i in range(n):
    xi = gpsX[i]
    yi = gpsY[i]
    for j in range(n):
        xj = gpsX[j]
        yj = gpsY[j]
        D[i,j] = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
from numpy.linalg import inv
    
invD = inv(D)
W = np.zeros((m,m,n))

def inner_loop((gx,gy)):
    x0 = gridX[gx]
    y0 = gridY[gy]
    W_temp = np.zeros(n)
    for j in xrange(n):
        wj = 0.0
        for i in xrange(n):
            xi = gpsX[i]
            yi = gpsY[i]
            d0i = np.sqrt((x0 - xi) ** 2 + (y0 - yi) ** 2)
            wj += invD[i,j]*d0i
            W_temp[j] = wj
    return W_temp


def get_Mulit_Quadratic_Weights_MultiCore(gpsX,gpsY,IPWsites,doy,yr,num_cores):    
    print 'Inverse done'
    start_time = time.time()
    grid_points = [(gx,gy) for gx in xrange(m) for gy in xrange(m)]
    p = multiprocessing.Pool(num_cores)
    W_total = p.map(inner_loop,grid_points)
    end_time = time.time()
    print len(W_total)
#    np.save('multi_processor_weights.npy',W)
    print 'Time taken to generate weights 100x100x44 weights matrix  = %.2fs'%(end_time-start_time)
    return W_total

yr = 14

#W = get_Mulit_Quadratic_Weights(gpsX,gpsY,IPWsites,128,yr)

for core in range(2,9):
    print 'Running number of cores %d '%core
    W_total = get_Mulit_Quadratic_Weights_MultiCore(gpsX,gpsY,IPWsites,128,yr,core)




'''
    def plot_domain(self,PixelPoints,marker = 'r*'):
        PixelPoints = PixelPoints.reshape(-1,2)
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

data = range(500)

def mp_worker_2(x):
    return x**2

def mp_worker((inputs, the_time)):
    print " Processs %s\tWaiting %s seconds" % (inputs, the_time)
    time.sleep(int(the_time))
    print " Process %s\tDONE" % inputs

def mp_handler():
    p = multiprocessing.Pool(8)
    p.map(mp_worker, data)
    
def mp_handler_2():
#    print data
    p = multiprocessing.Pool(8)
    new_data = p.map(mp_worker_2 , tuple(data))
    print new_data
    
'''

#if __name__ == '__main__':
    
#    mp_handler_2()
