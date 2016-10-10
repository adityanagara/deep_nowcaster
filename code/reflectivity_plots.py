# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:23:00 2016

@author: adityanagarajan
"""

from matplotlib import pyplot as plt
import numpy as np
import os
from netCDF4 import Dataset
import DFWnet
import BuildDataSet



DFW = DFWnet.CommonData()
Months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']



def getNEXRADFile(doy,t,yr):
    
    DFW.doytodate(yr,int(doy))
    NEXRAD_Folder = '../data/RadarData/NEXRAD/20'+ str(yr) + os.sep  + Months[int(DFW.mon) - 1] + DFW.day
    files = os.listdir(NEXRAD_Folder)
    files = filter(lambda x: x[-3:] == '.nc',files)
    files.sort(key = lambda x: int(x[-7:-3]))
    rad = Dataset(NEXRAD_Folder  + os.sep + files[t])
    return files[t],rad
    
def cart2pol(x,y):
    r = np.sqrt(np.power(x,2) + np.power(y,2))
    theta = np.degrees(np.arctan2(y,x))
    return theta,r

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
    

def plot_reflectivity(t,doy,yr):
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
    temp_file,rad = getNEXRADFile(doy,t,yr)   
    print temp_file
    gridZ = refl_polar_cartesian(rad,m)
    gridZ = gridZ.T
    DFW.doytodate(yr,int(doy))
    gridZ[gridZ < 30.0] = np.nan
    gridZ = np.ma.array(gridZ, mask=np.isnan(gridZ))
    plt.figure()
    plt.pcolor(gridX,gridY,gridZ,cmap='jet', vmin=10, vmax=60)
    cbar = plt.colorbar()
    cbar.set_label('Reflectivity (dBz)',size = 18)
    plt.grid()
    plt.xlim((-150.0,150.0))
    plt.ylim((-150.0,150.0))
    plt.title('Reflectivity fields for: ' + DFW.mon + '/' + DFW.day + ' Time: ' + time_index[t] + ' (UTC)',size = 18)
    plt.xlabel('Easting (km)', size = 18)
    plt.ylabel('Northing (km)', size = 18)
#    plt.show()
    if not os.path.exists('../output/ReflOnly/20' + str(yr)+ os.sep + Months[int(DFW.mon) -1] + str(DFW.day) + os.sep):
        os.mkdir('../output/ReflOnly/20' + str(yr)+ os.sep + Months[int(DFW.mon) -1] + str(DFW.day) + os.sep)
    plt.savefig('../output/ReflOnly/20' + str(yr) + os.sep +  Months[int(DFW.mon) -1] + str(DFW.day) + os.sep +'Plot_'  + str(t) + '.png')
    
    # plt.savefig('../output/reflectivity_ipw_fields/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + str(DFW.day) + os.sep +'Plot_'  + str(t) + '.png')


def main():
    print 'Plot reflectivity'
    data_builder = BuildDataSet.dataset(num_points = 500)
    yr = 14
    storm_dates = data_builder.load_storm_days(yr)
    for d in storm_dates[:1]:
        for t in range(48):
            plot_reflectivity(t,d[0],yr)

if __name__ == '__main__':
    main()

#    for d in storm_dates[-1:]:
#        # add to the above dictionary the dates with bad vals
##        drop_stations(yr,d[0],IPWvals,missing_value_dict[d[0] - 1])
#        print d,missing_value_dict[d[0] - 1]
#        IPWvals_day = IPWvals[int(d[0]) - 1,:,1:].astype('float')
#        gpsX,gpsY,IPWsites,IPWvals_day = get_GPS_cartesian(missing_value_dict[int(d[0]) -1],IPWvals_day,DFW_network.sites)
#        if normalize == 'monthly':
#            IPWvals_day = NormalizeIPW_Normal(SummaryStats,IPWvals_day,int(d[0]),missing_value_dict,IPWsites,yr)
#        elif normalize == 'seasonal':
#            IPWvals_day = NormalizeIPW_Seasonal(SummaryStats,IPWvals_day,int(d[0]),missing_value_dict,IPWsites,yr)
#        if os.path.exists('../output/Weights/20' +str(yr) + '/Weights_'  + str(int(d[0])) + '.npy'):
#            print 'Weights exist loading weights...'
#            W = np.load('../output/Weights/20' +str(yr) + '/Weights_'  + str(int(d[0])) + '.npy')
#        else:
#            print 'Making Multiquadratic weights'
#            W = get_Mulit_Quadratic_Weights(gpsX,gpsY,IPWsites,int(d[0]),yr)
#        for t in range(48):
#            plot_radarIPW(t,d[0],IPWvals_day,W,missing_value_dict[int(d[0]) -1],yr,gpsX,gpsY,IPWsites)


#def plot_radarIPW(t,doy,IPWvals,W,no_data,yr,gpsX,gpsY,IPWsites,processes_from_netcdf = True):
#    # define a list with the time index 00:00, 00:30, 01:00, 01:30,...
#    doy = int(doy)
#    time_index = ['{0}{1}'.format(str(x).zfill(2),str(y).zfill(2)) for x in range(24) for y in [0,30]]
#    Months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
#    # This code converts the radar reflectivity from polar coordinaates to cartesian coordinates
#
#    # Initialize the resolution of the array 
#    m = 100
#    # Initialize an empty array to hold the reflectivity values in the cartesian coordinates
#
#    # Make the 150x150 km2 grid
#    gridX = np.arange(-150.0,151.0,300.0/(m-1))
#    gridY = np.arange(-150.0,151.0,300.0/(m-1))
#    # Convert this grid to polar coordinates to match the values with obtained from the netcdf file
#        
#    # Get the file corresponding to the time stamp
#    # return the file and the netcdf object variable
#    if processes_from_netcdf:
#        temp_file,rad = getNEXRADFile(doy,t,yr)   
#        print temp_file
#        gridZ = refl_polar_cartesian(rad,m)
#        gridZ = gridZ.T
#    else:
#        gridZ = get_refl_array(t,doy,yr)
#        
#    DFW.doytodate(yr,int(doy))
#
#    # Transpose back to the original array
#    
#    np.save('../data/dataset/20' + str(yr) + '/RadarRefl' + str(yr) +'_'  + str(int(doy)) + '_' + str(t) +  '.npy',gridZ)
#    
#    gridIPW = gridIPWfields(t,IPWvals,W,int(doy))
#    
#    #Plot values greater than 30 dbZ
#    gridZ[gridZ < 30.0] = np.nan
#    # Mask values with nan
#    gridZ = np.ma.array(gridZ, mask=np.isnan(gridZ))
#    
#    # Plot the reflectivity fields overlapped over the ipw fields
#    plt.figure()
##    plot_sites(gpsX,gpsY,IPWsites)
#    plt.pcolor(gridX,gridY,gridIPW,cmap='gist_ncar', vmin=-3.0, vmax=3.0)
#    np.save('../data/dataset/20' + str(yr) + '/IPWdata' +str(yr) +'_' + str(int(doy)) + '_' + str(t) + '.npy',gridIPW)
#    cbar = plt.colorbar()
#    cbar.set_label('Normalized IPW values')
#    plt.pcolor(gridX,gridY,gridZ,cmap='jet', vmin=10, vmax=60)
#    cbar = plt.colorbar()
#    cbar.set_label('Reflectivity (dBz)')
#    plt.grid()
#    plt.xlim((-150.0,150.0))
#    plt.ylim((-150.0,150.0))
#    plt.xlabel('Easting (km)')
#    plt.ylabel('Northing (km)')
#    plt.title('Reflectivity NIPW fields for: ' + DFW.mon + '/' + DFW.day + ' Time: ' + time_index[t] + ' (UTC)')
#    if not os.path.exists('../output/reflectivity_ipw_fields/20' + str(yr)+ os.sep + Months[int(DFW.mon) -1] + str(DFW.day) + os.sep):
#        os.mkdir('../output/reflectivity_ipw_fields/20' + str(yr)+ os.sep + Months[int(DFW.mon) -1] + str(DFW.day) + os.sep)
#    plt.savefig('../output/reflectivity_ipw_fields/20' + str(yr) + os.sep + Months[int(DFW.mon) -1] + str(DFW.day) + os.sep +'Plot_'  + str(t) + '.png')

