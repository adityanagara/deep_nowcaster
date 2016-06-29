# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:58:57 2016

@author: adityanagarajan
"""

#from mpl_toolkits.basemap import Basemap
#
#from matplotlib import pyplot as plt
#import numpy as np
##import cartopy
import DFWnet
#
#
#
##def new_map(fig, lon, lat):
##    # Create projection centered on the radar. This allows us to use x
##    # and y relative to the radar.
##    proj = cartopy.crs.LambertConformal(central_longitude=lon, central_latitude=lat)
##
##    # New axes with the specified projection
##    ax = fig.add_subplot(1, 1, 1, projection=proj)
##
##    # Add coastlines
##    ax.coastlines('50m', 'black', linewidth=2, zorder=2)
##
##    # Grab state borders
##    state_borders = cartopy.feature.NaturalEarthFeature(
##        category='cultural', name='admin_1_states_provinces_lines',
##        scale='50m', facecolor='none')
##    ax.add_feature(state_borders, edgecolor='black', linewidth=1, zorder=3)
##    
##    return ax
#
##def plot_map():
#
#
## setup lambert conformal basemap.
## lat_1 is first standard parallel.
## lat_2 is second standard parallel (defaults to lat_1).
## lon_0,lat_0 is central point.
## rsphere=(6378137.00,6356752.3142) specifies WGS4 ellipsoid
## area_thresh=1000 means don't plot coastline features less
## than 1000 km^2 in area.
#
#DFW = DFWnet.CommonData()
#m = Basemap(width=12000000,height=9000000,
#            rsphere=(6378137.00,6356752.3142),\
#            resolution='l',area_thresh=100.,projection='lcc',\
#            lat_1=45.,lat_2=55,lat_0=DFW.KFWSlat,lon_0=DFW.KFWSlong)
#m.drawcoastlines()
#m.fillcontinents(color='coral',lake_color='aqua')
## draw parallels and meridians.
#m.drawparallels(np.arange(-80.,81.,20.))
#m.drawmeridians(np.arange(-180.,181.,20.))
#m.drawmapboundary(fill_color='aqua')
#m.drawcounties()
#m.drawgreatcircle()
#m.ma
## draw tissot's indicatrix to show distortion.
#ax = plt.gca()
##for y in np.linspace(m.ymax/20,19*m.ymax/20,9):
##    for x in np.linspace(m.xmax/20,19*m.xmax/20,12):
##        lon, lat = m(x,y,inverse=True)
##        poly = m.tissot(lon,lat,1.5,100,\
##                        facecolor='green',zorder=10,alpha=0.5)
#plt.title("Lambert Conformal Projection")
#plt.show()
    

#def main():
#    fig = plt.figure(figsize=(10, 10))
#    
#    #ax = new_map(fig, data.StationLongitude, data.StationLatitude)
#    ax = new_map(fig, DFW.KFWSlong, DFW.KFWSlat)
#    print 'Why is there no plot??'
    
#    ax.pcolormesh(x, y, ref, cmap=ref_cmap, norm=ref_norm, zorder=0)
    


#if __name__ == '__main__':
#    main()

from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
#from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt

DFW = DFWnet.CommonData()

# plot rainfall from NWS using special precipitation
# colormap used by the NWS, and included in basemap.

#nc = NetCDFFile('../../../examples/nws_precip_conus_20061222.nc')
# data from http://water.weather.gov/precip/
#prcpvar = nc.variables['amountofprecip']
#data = 0.01*prcpvar[:]
#latcorners = nc.variables['lat'][:]
#loncorners = -nc.variables['lon'][:]
#lon_0 = -nc.variables['true_lon'].getValue()
#lat_0 = nc.variables['true_lat'].getValue()
# create figure and axes instances
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
# create polar stereographic Basemap instance.
m = Basemap(projection='stere',lon_0=DFW.KFWS.long,lat_0=DFW.KFWSlat,lat_ts=DFW.KFWSlat,\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[2],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[2],\
            rsphere=6371200.,resolution='l',area_thresh=10000)
# draw coastlines, state and country boundaries, edge of map.
m.drawcoastlines()
m.drawstates()
m.drawcountries()
# draw parallels.
parallels = np.arange(0.,90,10.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(180.,360.,10.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
ny = data.shape[0]; nx = data.shape[1]
lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
x, y = m(lons, lats) # compute map proj coordinates.
# draw filled contours.
clevs = [0,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750]
cs = m.contourf(x,y,data,clevs,cmap=cm.s3pcpn)
# add colorbar.
cbar = m.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title(prcpvar.long_name+' for period ending '+prcpvar.dateofdata)
plt.show()