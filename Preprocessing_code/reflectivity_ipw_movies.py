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
import subprocess

DFW = DFWnet.CommonData()
Months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

def cart2pol(x,y):
    r = np.sqrt(np.power(x,2) + np.power(y,2))
    theta = np.degrees(np.arctan2(y,x))
    return theta,r

#def get_IPW_vals(doy):
#    global DFW 
#    print 'Obtain IPW values for DOY: ' + str(doy)
#    
#    IPWvals = DFW.IPWvals
#    
#    return IPWvals[doy -1,:,:]

# Function which returns a matrix of normalized values (Normalization done wrt to month)
def NormalizeIPW_Normal(SummaryStats,IPWvals,doy,missing_value_dict,sites_list,yr):
    DFW.doytodate(yr,doy)
    sites = sites_list[:,0]
#    sites = sites[np.logical_not(np.in1d(sites,missing_value_dict[int(doy) -1]))]
    IPWdataN = np.zeros(IPWvals.shape)
    for x in range(sites.shape[0]):
        IPWdataN[x,:] = (IPWvals[x,:] - SummaryStats[SummaryStats.site.values == sites[x]][DFW.mon + '_Avg'].values) / SummaryStats[SummaryStats.site.values == sites[x]][DFW.mon + '_Std'].values
#    IPWdataN = np.array(map(lambda x: (x[1:].astype('float') - SummaryStats[SummaryStats.site.values == x[0]][DFW.mon + '_Avg'].values)/SummaryStats[SummaryStats.site.values == x[0]][DFW.mon + '_Std'].values,IPWvals))
    return IPWdataN


def NormalizeIPW_Seasonal(SummaryStats,IPWvals,doy,missing_value_dict,sites_list,yr):
    DFW.doytodate(yr,doy)
    sites = sites_list[:,0]
#    sites = sites[np.logical_not(np.in1d(sites,missing_value_dict[int(doy) -1]))]
    IPWdataN = np.zeros(IPWvals.shape)
    for x in range(sites.shape[0]):
        IPWdataN[x,:] = (IPWvals[x,:] - SummaryStats[SummaryStats.site.values == sites[x]]['storm_season_avg'].values) / SummaryStats[SummaryStats.site.values == sites[x]]['storm_season_std'].values
        
    return IPWdataN

#def get_sites_without_data(doy):
#    global DFW
#    
#    IPWvals = get_IPW_vals(doy)
#    
#    missing_data = []
#    # This station seems to be constant through out
#    missing_data.append('txsr')
#    # Uncomment this and comment above for 2015 data set
##    missing_data.append('txth')
#    if doy == 129:
#        missing_data.append('zfw1')
#    if doy == 132:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txck')
#        missing_data.append('txhi')
#    elif doy == 133:
#        missing_data.append('txmn')
#        missing_data.append('txwa')
#    elif doy == 145:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txgl')
#        missing_data.append('txja')
#    elif doy == 151:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txmw')
#        missing_data.append('txwe')
#        
#    elif doy == 160:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txno')
#        missing_data.append('txgl')
#        missing_data.append('txhm')
#        missing_data.append('txdc')
#    elif doy == 169:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txmw')
#        missing_data.append('txdc')
#    elif doy == 170:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txmw')
#        missing_data.append('txwe')
#        missing_data.append('txja')
#        missing_data.append('txke')
#    elif doy == 198:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txgl')
#        missing_data.append('txhm')
#        missing_data.append('txco')
#        missing_data.append('txde')
#    elif doy == 199:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txka')
#        missing_data.append('txmn')
#        missing_data.append('txwa')
#    elif doy == 223:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txja')
#    elif doy == 228:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txhm')
#    elif doy == 229:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txka')
#    elif doy == 230:
#        print 'We hit -----> ' + str(doy)
#        missing_data.append('txno')
#    
#    for x in IPWvals:
#        if x[x == str(np.nan)].size > 0:
#            missing_data.append(x[0])
#    return missing_data


def get_GPS_cartesian(nan_sites,IPWvals,IPWsites):
    
    # Remove IPW vals and sites that have nan values in them at any point during the 
    # day
    IPWvals = IPWvals[np.logical_not(np.in1d(IPWsites[:,0],nan_sites))]
    IPWsites = IPWsites[np.logical_not(np.in1d(IPWsites[:,0],nan_sites))]
    
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
    return gpsX,gpsY,IPWsites,IPWvals


def getNEXRADFile(doy,t,yr):
    
    DFW.doytodate(yr,int(doy))
    NEXRAD_Folder = '../data/RadarData/NEXRAD/20'+ str(yr) + os.sep  + Months[int(DFW.mon) - 1] + DFW.day
    files = os.listdir(NEXRAD_Folder)
    files = filter(lambda x: x[-3:] == '.nc',files)
    files.sort(key = lambda x: int(x[-7:-3]))
    rad = Dataset(NEXRAD_Folder  + os.sep + files[t])
    return files[t],rad

def gridIPWfields(t,IPWvals,W,doy):
#    print 'Obtain the IPW grid values for: %d'%t 
    ipwData = IPWvals
    N = 100
    gridIPW = np.zeros((N,N))
    for gx in range(N):
        for gy in range(N):
            h0 = 0
            for j in range(ipwData.shape[0]):
                hj = ipwData[j,t]
                wj = W[gx,gy,j]
                h0 += wj*hj
            gridIPW[gx,gy] = h0
    gridIPW = gridIPW.T
    return gridIPW

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
    if saveWeights:
        np.save('../output/Weights/20' +str(yr) + '/Weights_'  + str(int(doy)) + '.npy',W)
    return W   

def refl_polar_cartesian(rad,m):
    gridX = np.arange(-150.0,151.0,300.0/(m-1))
    gridY = np.arange(-150.0,151.0,300.0/(m-1))
    xMesh,yMesh = np.meshgrid(gridX,gridY)
    gridZ = np.empty((m,m))
    gridZ.fill(np.nan)
    gridA,gridR = cart2pol(xMesh,yMesh)
    # Convert from [-180.0,180] to [0,360.0]
    gridA[gridA < 0.0] = 360.0 + gridA[gridA < 0.0]

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
    
    for a in range(azimuthVector.size):
    
        I = np.less(np.abs(gridA - azimuthVector[a]),1.0)
    
        J = np.floor(((gridR[np.abs(gridA - azimuthVector[a]) < 1.0] - startRange)/gateWidth ))
    
        gridZ[I] = Z[a,tuple(J)]
    
    return gridZ

def get_refl_array(t,doy,yr):
    base_path = '../data/RadarData/Decimated/'
    file_to_get = str(yr) + str(doy) + 'refl_decimated.npy'
    Z = np.load(base_path + file_to_get)
    return Z[t,...]

#KFWS_ring=plt.Circle((0.0,0.0),230,color='b',fill=False)
#
#plt.figure(figsize=(10,10))
#
#plt.title('GPS sites inside the ring of KFWS radar (cartesian)')
#plt.xlabel('Easting (km)')
#plt.ylabel('Northing (km)')
#plt.plot((0.0,0.0),'k*')
#plt.gca().add_artist(KFWS_ring)
#plt.plot(gpsX,gpsY,'ro')
#plt.xlim((-250.0,250.0))
#plt.ylim((-250.0,250))
#for s in range(DFW.sites.shape[0]):
#    plt.annotate(DFW.sites[s,0],(gpsX[s],gpsY[s]))
#
#plt.annotate('KFWS',(0.0,0.0))
#    
#plt.grid()
    
def plot_sites(gpsX,gpsY,IPWsites):
    plt.plot((0.0,0.0),'k*')
    plt.plot(gpsX,gpsY,'ro')
    for s in range(IPWsites.shape[0]):
        plt.annotate(IPWsites[s,0],(gpsX[s],gpsY[s]))

    
    
    
def plot_radarIPW(t,doy,IPWvals,W,no_data,yr,gpsX,gpsY,IPWsites,processes_from_netcdf = True):
    # define a list with the time index 00:00, 00:30, 01:00, 01:30,...
    doy = int(doy)
    time_index = ['{0}{1}'.format(str(x).zfill(2),str(y).zfill(2)) for x in range(24) for y in [0,30]]
    Months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    # This code converts the radar reflectivity from polar coordinaates to cartesian coordinates

    # Initialize the resolution of the array 
    m = 100
    # Initialize an empty array to hold the reflectivity values in the cartesian coordinates

    # Make the 150x150 km2 grid
    gridX = np.arange(-150.0,151.0,300.0/(m-1))
    gridY = np.arange(-150.0,151.0,300.0/(m-1))
    # Convert this grid to polar coordinates to match the values with obtained from the netcdf file
        
    # Get the file corresponding to the time stamp
    # return the file and the netcdf object variable
    if processes_from_netcdf:
        temp_file,rad = getNEXRADFile(doy,t,yr)   
        print temp_file
        gridZ = refl_polar_cartesian(rad,m)
        gridZ = gridZ.T
    else:
        gridZ = get_refl_array(t,doy,yr)
        
    DFW.doytodate(yr,int(doy))

    # Transpose back to the original array
    
    np.save('../data/dataset/20' + str(yr) + '/RadarRefl' + str(yr) +'_'  + str(int(doy)) + '_' + str(t) +  '.npy',gridZ)
    
    gridIPW = gridIPWfields(t,IPWvals,W,int(doy))
    
    #Plot values greater than 30 dbZ
    gridZ[gridZ < 30.0] = np.nan
    # Mask values with nan
    gridZ = np.ma.array(gridZ, mask=np.isnan(gridZ))
    
    # Plot the reflectivity fields overlapped over the ipw fields
    plt.figure()
#    plot_sites(gpsX,gpsY,IPWsites)
    plt.pcolor(gridX,gridY,gridIPW,cmap='gist_ncar', vmin=-3.0, vmax=3.0)
    np.save('../data/dataset/20' + str(yr) + '/IPWdata' +str(yr) +'_' + str(int(doy)) + '_' + str(t) + '.npy',gridIPW)
#    cbar = plt.colorbar()
#    cbar.set_label('Normalized IPW values', size = 16)
    plt.pcolor(gridX,gridY,gridZ,cmap='jet', vmin=10, vmax=60)
#    cbar = plt.colorbar()
#    cbar.set_label('Reflectivity (dBz)',size = 16)
    plt.grid()
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))
    plt.xlabel('Easting (km)',size = 18)
    plt.ylabel('Northing (km)', size = 18)
    plt.title('Reflectivity NIPW fields for: ' + DFW.mon + '/' + DFW.day + ' Time: ' + time_index[t] + ' (UTC)', size = 18)
    if not os.path.exists('../output/reflectivity_ipw_fields/20' + str(yr)+ os.sep + Months[int(DFW.mon) -1] + str(DFW.day) + os.sep):
        os.mkdir('../output/reflectivity_ipw_fields/20' + str(yr)+ os.sep + Months[int(DFW.mon) -1] + str(DFW.day) + os.sep)
    plt.savefig('../output/reflectivity_ipw_fields/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + str(DFW.day) + os.sep +'Plot_'  + str(t) + '.png')

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        >>> y= array([1, 1, 1, NaN, NaN, 2, 2, NaN, 0])
        http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def bad_val_imputer(doy,stn,IPWvals,first_order,nan_sites):
    first_order_drops = np.where(first_order <= -0.5)[0]
    first_order_increase = np.where(first_order >= 1.0)[0]
    index_to_nan = []
    # cases to consider 
    # 1. Case where drop occurs ar second to last index
    # 2. case where only increase occurs at first index
    # 3. case for 2,3,4 plateu after drop before increase
    # Weird jumps where there is a steap increase before decrease
    # There can also be super high increase and then a drop so WTF
    if first_order_increase.shape == first_order_drops.shape:
        if first_order_increase[0] < first_order_drops[0]:
            # case to handle increase before decrease IPWvals[193,22,1:].reshape(-1,)
            for x in range(first_order_increase.shape[0]):
                ctr = first_order_increase[x]
                index_to_nan.append(ctr)
                ctr+=1
                if first_order_increase[x] < first_order_drops[x]:
                    while first_order_drops[x] !=  ctr:
                        index_to_nan.append(ctr)
                        ctr+=1
                else:
                    nan_sites.append(stn)
        else:
            # case to handle decrease before increase
            for x in range(first_order_drops.shape[0]):
                ctr = first_order_drops[x] + 1        
                index_to_nan.append(ctr)
                # check to see that a drop occurs before a rise if not
                # this time series data is so fucked!!!
                if first_order_drops[x] < first_order_increase[x]:
                    # check to see if the indeex are the same
                    while first_order_increase[x] != ctr:
                        ctr += 1
                        index_to_nan.append(ctr)
                else:
                    nan_sites.append(stn)
    else:
        # for now just kill that station please
        # fuck that station up, fuck it up 
        # it is useless to any of us no fucking point
        nan_sites.append(stn)
    
    IPWvals[index_to_nan] = np.nan
    nans,x = nan_helper(IPWvals)
    IPWvals[nans] = np.interp(x(nans),x(~nans),IPWvals[~nans])
    # impute dem values 

def drop_stations(yr,doy,IPWvals,nan_sites):
    '''We are going to make sure that no site shows us a 100% increase or 
    decrease in water vapor from the previous value
    Drop station algorithm: Take first order and second order diff
    Test case txbx(i = 5) doy = 215, i = 214 -- 2014
    txke (i = 22) doy = 211 i = 210'''
    if yr == 14:
        nan_sites.append('txsr')
    stations = DFW.sites[:,0]
    IPWvals = IPWvals[doy,:,1:].astype('float')
    IPWvals[IPWvals < 0.0] = 0.1
    for i,stn in enumerate(stations):
        if stn not in nan_sites:
            first_order = np.diff(IPWvals[i,:]) / IPWvals[i,:-1]
            # only impute for fluctuations less than 4            
            if np.any(first_order <= -0.5,axis = 0) and np.where(first_order <= -0.5)[0].shape[0] < 4:
                bad_val_imputer(doy,stn,IPWvals[i,:],first_order,nan_sites)
            elif np.any(first_order <= -0.5,axis = 0) and np.where(first_order <= -0.5)[0].shape[0] >= 4:
                nan_sites.append(stn)
    
    return IPWvals
    
                

def nan_stations(yr,IPWvals,date_range):
    '''Takes input the array of shape 365,44,49 array and returns
    a distionary of missing_data[doy - 1] = [station list], the 
    data_range is index not doy (for doy = index + 1)
    '''
    missing_data = {}
    for i in date_range:
        day_vals = IPWvals[i,...]
        if i not in missing_data.keys():
            missing_data[i] = []
        for x in day_vals:
            if x[x == str(np.nan)].size > 0:
                missing_data[i].append(x[0])
        newVals = drop_stations(yr,i,IPWvals,missing_data[i])
        IPWvals[i,:,1:] = newVals.astype('S32')
    return missing_data

# Old drop stations code
#def drop_stations(yr,doy,IPWvals,nan_sites):
#    '''We are going to make sure that no site shows us a 100% increase or 
#    decrease in water vapor from the previous value'''
#    if yr == 14:
#        nan_sites.append('txsr')
#    stations = IPWvals[doy,:,0]
#    IPWvals = IPWvals[doy,:,1:].astype('float')
#    for i,stn in enumerate(stations):
#        if stn not in nan_sites:
#            temp_vec = np.abs(np.diff(IPWvals[i,:])) / IPWvals[i,:][:-1]
#            if np.any(temp_vec >= 1.0,axis = 0):
#                nan_sites.append(stations[i])
#
#def nan_stations(yr,IPWvals):
#    '''Takes input the array of shape 365,44,49 array and returns
#    a distionary of missing_data[doy] = [station list]'''
#    missing_data = {}
#    for i in range(IPWvals.shape[0]):
#        day_vals = IPWvals[i,...]
#        if i not in missing_data.keys():
#            missing_data[i] = []
#        for x in day_vals:
#            if x[x == str(np.nan)].size > 0:
#                missing_data[i].append(x[0])
#        drop_stations(yr,i,IPWvals,missing_data[i])
#    return missing_data


def moving_window_averag(IPWvals):
    '''moving average is taken by convolving over a signal of 24 1/24 values
    shape of resulting signal with mode 'valid' max(M,N) - min(M,N) + 1. Convolution
    perfoemed over entire year for each station. Each station has 365*48 values thus
    with 'valid' option we get values where both signals overlap returning 17497. 
    '''
    smooth_IPWvals = np.zeros((364,44,48)) # we define for 364 days
    N = 24
    for stn in range(IPWvals.shape[1]):
        smooth_IPWvals[:,stn,:] = np.convolve(IPWvals[:,stn,1:].reshape(-1,).astype('float'),np.ones(N,)/N,mode='valid')[:17472].reshape(364,48)
    return smooth_IPWvals

def get_stn_month_stats(yr):
    mon_day_dict = {}
    for x in range(1,366):
        DFW.doytodate(yr,x)
        if DFW.mon not in mon_day_dict.keys():
            mon_day_dict[DFW.mon] = []
        mon_day_dict[DFW.mon].append(DFW.day)
    return mon_day_dict

def get_stats(si,IPWvals,sites,yr,missing_value_dict):
    mon_day_dict = get_stn_month_stats(yr)
    sites_list = sites[:,0]
    avg_list = []
    std_list = []
    # Find the yearly average and standard deviations for each station
    yr_vals = np.hstack(IPWvals[:,si,1:]).astype('float')
    yr_vals[yr_vals < 0.0] = np.nan
    yr_vals = yr_vals[~np.isnan(yr_vals)]
    yr_avg = np.average(yr_vals)
    yr_std = np.std(yr_vals)
    DFW.date2doy(14,5,1)
    start_day = DFW.doy - 1
    DFW.date2doy(14,9,1)
    end_day = DFW.doy -1 # since the index is doy - 1 anyways we dont need to -2 and we only iterate to -1 of the index anyways
    # seasonal avg
    seasonal_vals = np.hstack(IPWvals[start_day:end_day,si,1:]).astype('float')
    seasonal_vals = seasonal_vals[~np.isnan(seasonal_vals)]
    seasonal_vals = seasonal_vals[seasonal_vals != 0.0]
    seasonal_avg = np.average(seasonal_vals)
    seasonal_std = np.std(seasonal_vals)
    # find the monthly average and standard deviations for each station
    for mon in sorted(mon_day_dict.keys()):
        DFW.date2doy(yr,int(mon),int(mon_day_dict[mon][0] ))
        start = int(DFW.doy)
        DFW.date2doy(yr,int(mon),int(mon_day_dict[mon][-1]))
        end = int(DFW.doy)
        if (yr == 15 and mon == '08'):
            end = 242
        mon_vals = np.hstack(IPWvals[start - 1: end ,si,1:]).astype('float')
        mon_vals = mon_vals[~np.isnan(mon_vals)]
#        mon_vals = mon_vals[mon_vals != 0.0]
        avg = np.average(mon_vals)
        avg_list.append(avg)
        std = np.std(mon_vals)
        std_list.append(std)
    
    # make a list containing ['site','yr avg','yr std','']
    Summary = [sites_list[si],yr_avg,yr_std]
    Summary.extend([seasonal_avg,seasonal_std])
    Summary.extend(avg_list)
    Summary.extend(std_list)
    
    return Summary

def make_summary_dataframe(yr,IPWvals,missing_value_dict):
    cols = ['site']
    cols.extend(['Yearly_AVG','Yearly_STD','storm_season_avg','storm_season_std'])
    cols.extend(['{0}_Avg'.format(str(x).zfill(2)) for x in range(1,13)])
    cols.extend(['{0}_Std'.format(str(x).zfill(2)) for x in range(1,13)])
    itr=0
    SummaryStats = pd.DataFrame(columns = cols)
    for si in range(44):
        temp_list = get_stats(si,IPWvals,DFW.sites,yr,missing_value_dict)
        SummaryStats.loc[itr] = temp_list
        itr+=1
    return SummaryStats

def set_ipw_nan(IPWvals,missing_value_dict):
    stations = list(DFW.sites[:,0])
    for day in missing_value_dict.keys():
        for stn in missing_value_dict[day]:
            IPWvals[day,stations.index(stn),1:] = np.array([str(np.nan)]*48)
    
def main(yr,storm_dates,make_summary_stats = True,normalize = 'monthly'):        
    DFW_network = DFWnet.CommonData()
    if yr == 14:
        date_range = range(120,273) # DOY 122 - 244 -- 121,244
        IPWvals = DFW_network.IPWvals_2014
    elif yr == 15:
        date_range = range(120,273) # DOY 122 - 244 -- 121,244
        IPWvals = DFW_network.IPWvals_2015
        IPWvals[IPWvals == '-9.9'] = str(np.nan)
    elif yr == 16:
        date_range = range(122,213) # DOY 122 - 244 -- index 122,244
        IPWvals = np.load('../data/2016IPW_data.npy')
    # Get stations with nan values in them. Resurnes the dictionary
#    missing_value_dict = nan_stations(yr,IPWvals)
    missing_value_dict = nan_stations(yr,IPWvals,date_range)
    # set the missing values to nan for days with corresponding stations
    # with corrupt values    
    set_ipw_nan(IPWvals,missing_value_dict)
    if make_summary_stats:
        # Define a data frame to hold monthly average and std
        SummaryStats = make_summary_dataframe(yr,IPWvals,missing_value_dict)
        SummaryStats.to_csv('../data/DFWsites_summary_20'+ str(yr) + '.csv')
        SummaryStats = pd.read_csv('../data/DFWsites_summary_20'+ str(yr) + '.csv',index_col=0)
    else:
        # average out the saved summary stats
        SummaryStats_2014 = pd.read_csv('../data/DFWsites_summary_20'+ str(14) + '.csv',index_col=0)
        SummaryStats_2015 = pd.read_csv('../data/DFWsites_summary_20'+ str(15) + '.csv',index_col=0)
        cols = ['site']
        cols.extend(['Yearly_AVG','Yearly_STD'])
        cols.extend(['{0}_Avg'.format(str(x).zfill(2)) for x in range(1,13)])
        cols.extend(['{0}_Std'.format(str(x).zfill(2)) for x in range(1,13)])
        SummaryStats = pd.DataFrame(columns = cols)
        SummaryStats.ix[:,'site'] = SummaryStats_2014.site.values
        # txsr and okar stations are fucked so impute one from an another
        SummaryStats_2014.ix[34,:] = SummaryStats_2015.ix[34,:]
        SummaryStats_2015.ix[0,['06_Avg','07_Avg','08_Avg']] = SummaryStats_2014.ix[0,['06_Avg','07_Avg','08_Avg']]
        SummaryStats_2015.ix[0,['06_Std','07_Std','08_Std']] = SummaryStats_2014.ix[0,['06_Std','07_Std','08_Std']]
        for key in ['{0}_Avg'.format(str(x).zfill(2)) for x in range(5,9)]:
            SummaryStats.ix[:,key] = np.mean((SummaryStats_2015[key].values,SummaryStats_2014[key].values),axis = 0)
        
        for key in ['{0}_Std'.format(str(x).zfill(2)) for x in range(5,9)]:
            SummaryStats.ix[:,key] = np.sqrt(
                                            np.mean((
                                                    SummaryStats_2014[key].apply(lambda x: x*x).values,
                                                    SummaryStats_2015[key].apply(lambda x: x*x).values),
                                                    axis = 0))
            
        SummaryStats.to_csv('../data/DFWsites_summary.csv')
        SummaryStats = pd.read_csv('../data/DFWsites_summary.csv',index_col=0)
        
        

    for d in storm_dates[:1]:
        # add to the above dictionary the dates with bad vals
#        drop_stations(yr,d[0],IPWvals,missing_value_dict[d[0] - 1])
        print d,missing_value_dict[d[0] - 1]
        IPWvals_day = IPWvals[int(d[0]) - 1,:,1:].astype('float')
        gpsX,gpsY,IPWsites,IPWvals_day = get_GPS_cartesian(missing_value_dict[int(d[0]) -1],IPWvals_day,DFW_network.sites)
        if normalize == 'monthly':
            IPWvals_day = NormalizeIPW_Normal(SummaryStats,IPWvals_day,int(d[0]),missing_value_dict,IPWsites,yr)
        elif normalize == 'seasonal':
            IPWvals_day = NormalizeIPW_Seasonal(SummaryStats,IPWvals_day,int(d[0]),missing_value_dict,IPWsites,yr)
        if os.path.exists('../output/Weights/20' +str(yr) + '/Weights_'  + str(int(d[0])) + '.npy'):
            print 'Weights exist loading weights...'
            W = np.load('../output/Weights/20' +str(yr) + '/Weights_'  + str(int(d[0])) + '.npy')
        else:
            print 'Making Multiquadratic weights'
            W = get_Mulit_Quadratic_Weights(gpsX,gpsY,IPWsites,int(d[0]),yr)
        for t in range(48):
            plot_radarIPW(t,d[0],IPWvals_day,W,missing_value_dict[int(d[0]) -1],yr,gpsX,gpsY,IPWsites)
    
if __name__ == '__main__':
#    np.seterr(invalid='raise')
    # July 24 2014, NEXRAD data not available for full day so we are going to kill 
    # this site
    for yr in [14]:
        print '-'*40
        if yr == 14:
            storm_dates = np.load('../data/storm_dates_2014.npy').astype('int')
            # The following 3 dates are going to be removed because we do not
            # have NEXRAD files for the entire day            
            idx1 = np.where(np.all(storm_dates == [205,  14,   7,  24],axis=1))[0][0]
            storm_dates = np.delete(storm_dates,idx1,axis = 0)
            idx2 = np.where(np.all(storm_dates == [176,  14,   6,  25],axis=1))[0][0]
            storm_dates = np.delete(storm_dates,idx2,axis = 0)
            idx3 = np.where(np.all(storm_dates == [204 , 14 ,  7 , 23],axis=1))[0][0]
            storm_dates = np.delete(storm_dates,idx3,axis = 0)
        elif yr == 15:
            storm_dates = np.load('../data/storm_dates_2015.npy').astype('int')
        elif yr == 16:
            storm_dates = np.load('../data/storm_dates_2016.npy').astype('int')    
        main(yr,storm_dates)  

'''
Stations to drop observer on 08/19/2016
2016
July 4th [txke] - kill this date entirely data is too noisy
[210,  16,   7,  28] - kill
[208,  16,   7,  26] feels like some of the stations here did not pass
well thru my filter

2014
May 22
May 23
May 28
May 31
June 9th is messy decide what to do
June 19th Noisy
June 22 Noisy
June 24 noisy. we left off here
June 25 
June 28 no examples
July 2 no examples
July 18 no examples
July 23 no examples

2015
May 12
May 18 no examples
May 20 no examples
May 22 no examples
May 27 no examples
May 28 no examples
June 19 no examples
June 19 no examples
'''
