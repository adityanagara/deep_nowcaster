# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:36:12 2015

@author: adityanagarajan
"""

import matplotlib
matplotlib.use('Agg')

import DFWnet
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from matplotlib import pyplot as plt
import time



DFW = DFWnet.CommonData()
yr = 15    

def cart2pol(x,y):
    r = np.sqrt(np.power(x,2) + np.power(y,2))
    theta = np.degrees(np.arctan2(y,x))
    return theta,r


def get_IPW_vals(doy):
    global DFW 
    print 'Obtain IPW values for DOY: ' + str(doy)
    
    IPWvals = DFW.IPWvals_2015
    
    return IPWvals[doy -1,:,:]

# Function which returns a matrix of normalized values (Normalization done wrt to month)
def NormalizeIPW_Normal(DAT,doy):
    global DFW
    
    DFW.doytodate(15,doy)
    
    SummaryStats = pd.read_csv('output/SitesSummary.csv',index_col=0)
    
    print DFW.mon
    IPWdataN = np.array(map(lambda x: (x[1:].astype('float') - SummaryStats[SummaryStats.site.values == x[0]][DFW.mon + '_Avg'].values)/SummaryStats[SummaryStats.site.values == x[0]][DFW.mon + '_Std'].values,DAT))
    
    return IPWdataN.astype('float')

def NormalizeIPW_CAMR(DAT,doy):
    global DFW
    Prvals = DFW.Prvals[doy-1,:,:]
    Prvals = Prvals[np.in1d(Prvals[:,0],DAT[:,0])]
    return (1000 * DAT[:,1:].astype('float'))/ Prvals[:,1:].astype('float') * 0.102
    return Prvals[:,1:].astype('float') * DAT[:,1:].astype('float')
    

def get_sites_without_data(doy):
    global DFW
    
    IPWvals = get_IPW_vals(doy)
    
    missing_data = []
#    missing_data.append('txsr')
    missing_data.append('txth')
    if doy == 129:
        missing_data.append('zfw1')
    if doy == 132:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txck')
        missing_data.append('txhi')
    elif doy == 133:
        missing_data.append('txmn')
        missing_data.append('txwa')
    elif doy == 145:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txgl')
        missing_data.append('txja')
    elif doy == 151:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txmw')
        missing_data.append('txwe')
        
    elif doy == 160:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txno')
        missing_data.append('txgl')
        missing_data.append('txhm')
        missing_data.append('txdc')
    elif doy == 169:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txmw')
        missing_data.append('txdc')
    elif doy == 170:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txmw')
        missing_data.append('txwe')
        missing_data.append('txja')
        missing_data.append('txke')
    elif doy == 198:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txgl')
        missing_data.append('txhm')
        missing_data.append('txco')
        missing_data.append('txde')
    elif doy == 199:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txka')
        missing_data.append('txmn')
        missing_data.append('txwa')
    elif doy == 223:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txja')
    elif doy == 228:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txhm')
    elif doy == 229:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txka')
    elif doy == 230:
        print 'We hit -----> ' + str(doy)
        missing_data.append('txno')
    
    for x in IPWvals:
        if x[x == str(np.nan)].size > 0:
            missing_data.append(x[0])
    return missing_data


def get_GPS_cartesian(sites):
    
    print 'The following sites have nan values'
    print sites
    global DFW
    IPWsites = DFW.sites
    
    for temp in sites:
        IPWsites = IPWsites[np.logical_not(IPWsites[:,0] == temp)]
    
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


def getNEXRADFile(doy,t):
    global DFW
    Months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    
    DFW.doytodate(14,doy)
    NEXRAD_Folder = 'data/RadarData/NEXRAD/2015/' + Months[int(DFW.mon) - 1] + DFW.day
    files = os.listdir(NEXRAD_Folder)
    files = filter(lambda x: x[-3:] == '.nc',files)
    files.sort(key = lambda x: int(x[-7:-3]))
    return files[t]


def gridIPWfields(t,IPWvals,W,doy):
    print 'Obtain the IPW grid values for: %d'%t 

    
    ipwData = IPWvals
    N = 100

    n = ipwData.shape[0]
    
    gridIPW = np.zeros((N,N))
    for gx in range(N):
        for gy in range(N):
            h0 = 0
            for j in range(n):
                hj = ipwData[j,t]
                wj = W[gx,gy,j]
                h0 = h0 + wj*hj
            gridIPW[gx,gy] = h0
    gridIPW = gridIPW.T
    return gridIPW

def get_Mulit_Quadratic_Weights(gpsX,gpsY,IPWsites,doy,saveWeights = True):
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

    for gx in range(m):
        x0 = gridX[gx]
        for gy in range(m):
            y0 = gridY[gy]
            for j in range(n):
                wj = 0.0
                for i in range(n):
                    xi = gpsX[i]
                    yi = gpsY[i]
                
                    d0i = np.sqrt((x0 - xi) ** 2 + (y0 - yi) ** 2)
                    wj = wj + invD[i,j]*d0i
                W[gx,gy,j] = wj


    end_time = time.time()
    print 'Time taken to generate weights 100x100x44 weights matrix  = %.2fs'%(end_time-start_time)
    if saveWeights:
        np.save('output/Weights/Weights_' + str(doy) + '.npy',W)
    
    return W   

def plot_radarIPW(t,doy,IPWvals,W,dirToMake):
    global DFW
    
    time_index = ['{0}{1}'.format(str(x).zfill(2),str(y).zfill(2)) for x in range(24) for y in [0,30]]
    # This code converts the radar reflectivity from polar coordinaates to cartesian coordinates

    # Initialize the resolution of the array 
    m = 100
    # Initialize an empty array to hold the reflectivity values in the cartesian coordinates
    gridZ = np.empty((m,m))
    gridZ.fill(np.nan)

    # Make the 150x150 km2 grid
    gridX = np.arange(-150.0,151.0,300.0/(m-1))
    gridY = np.arange(-150.0,151.0,300.0/(m-1))

    xMesh,yMesh = np.meshgrid(gridX,gridY)

    # Convert this grid to polar coordinates to match the values with obtained from the netcdf file
    gridA,gridR = cart2pol(xMesh,yMesh)

    # Convert from [-180.0,180] to [0,360.0]
    gridA[gridA < 0.0] = 360.0 + gridA[gridA < 0.0]
    
    # Get the file corresponding to the time stamp
#    temp_file = find_closest_radar_data(time_index[t])
    temp_file = getNEXRADFile(doy,t)
    
    print temp_file
    DFW.doytodate(14,doy)
    
    # Read the netcdf file from (Dataset part of netcdf4 library)
    rad = Dataset(r'data/RadarData/NEXRAD/2015/' + dirToMake + os.sep + temp_file)


    # Get the vector of azimuth angles
    azimuthVector = rad.variables['azimuth'][:]

    # Get the range gates
    rangeVector = rad.variables['gate'][:]

    startRange = rangeVector[0]

    gateWidth = np.median(np.diff(rangeVector))

    startRange = startRange /1000.0
    gateWidth = gateWidth / 1000.0

    # Get the level 3 products
    Z = rad.variables['BaseReflectivity'][:]

    # Transpose level 3 products to make (230,360)
    Z = Z.T
    
    for a in range(azimuthVector.size):
    
        I = np.less(np.abs(gridA - azimuthVector[a]),1.0)
    
        J = np.floor(((gridR[np.abs(gridA - azimuthVector[a]) < 1.0] - startRange)/gateWidth ))
    
        gridZ[I] = Z[tuple(J),a]
    
    # GET THE GPS X,Y CARTESIAN COORDINATES
    no_data = get_sites_without_data(doy)
    gpsX,gpsY,IPWsites = get_GPS_cartesian(no_data)

    # Transpose back to the original array

    gridZ = gridZ.T
    np.save('data/TrainTest/RadarRefl' + str(yr) + str(doy) + '_' + str(t) +  '.npy',gridZ)
    
    #Plot values greater than 30 dbZ
    gridZ[gridZ < 30.0] = np.nan
    
    
    
    
    gridIPW = gridIPWfields(t,IPWvals,W,doy)
    
    # Mask all values with nan
    gridZ = np.ma.array(gridZ, mask=np.isnan(gridZ))
    plt.figure()
    
    plt.pcolor(gridX,gridY,gridIPW,cmap='gist_ncar', vmin=-3.0, vmax=3.0)

    np.save('data/TrainTest/IPWdata' +str(yr) + str(doy) + '_' + str(t) + '.npy',gridIPW)
#    cbar = plt.colorbar()
#    cbar.set_label('Normalized IPW vals')
    
    plt.pcolor(gridX,gridY,gridZ,cmap='jet', vmin=10, vmax=60)
#    cbar = plt.colorbar()
#    cbar.set_label('Radar reflectivity')
    plt.grid()
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))
    
    plt.xlabel('Easting (km)')
    plt.ylabel('Northing (km)')
    plt.title('Reflectivity for: ' + DFW.mon + '/' + DFW.day + ' Time: ' + time_index[t] + ' (UTC)')
    
    
    plt.show()
    plt.savefig('output/RadarIPWfields_2015/' + dirToMake +'/Plot_'  + str(t) + '.png')
    

#Anomalies = np.load('data/Anomalies.npy')
Months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

#AnomaliesExtra = np.array((['05','09'],['05','13'],['05','24'],['05','26']))

#Anomalies = np.concatenate((Anomalies,AnomaliesExtra))

Anomalies = np.array((['05','08'],['05','10']))

for a in Anomalies[:1]:
    
    DFW.date2doy(yr,int(a[0]),int(a[1]))
    
    IPWvals = get_IPW_vals(DFW.doy)
    
    no_data = get_sites_without_data(DFW.doy)
    
    gpsX,gpsY,IPWsites = get_GPS_cartesian(no_data)
    
    print 'Following sites are not available'
    print no_data
    
    
    for temp in no_data:
        IPWvals = IPWvals[np.logical_not(IPWvals[:,0] == temp)]
    
    dirToMake = Months[int(a[0]) - 1] + a[1]
    
    IPWvals = NormalizeIPW_Normal(IPWvals,DFW.doy)
    
    if os.path.exists('output/Weights/Weights_' +str(yr) + str(DFW.doy) + '.npy'):
        W = np.load('output/Weights/Weights_' + str(yr) + str(DFW.doy) + '.npy')
    else:
        print 'Making Multiquadratic weights'
        W = get_Mulit_Quadratic_Weights(gpsX,gpsY,IPWsites,DFW.doy)
    
    
#    if not os.path.exists('output/IPWRadarProposal/ReflOnly/' + dirToMake):
#        os.mkdir('output/IPWRadarProposal/ReflOnly/' + dirToMake)
    
    print 'Making plots for ' + dirToMake
    for t in range(48):
        plot_radarIPW(t,DFW.doy,IPWvals,W,dirToMake)
        print 'Plot Made: ' + 'output/RadarIPWfields_2015/' + dirToMake +'/Plot_'  + str(t) + '.png'
    
            

    
    
    
        
    
    
    
    



    







