# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:12:51 2015

@author: adityanagarajan
"""

import numpy as np
import DFWnet
import os

import pandas as pd

DFW = DFWnet.CommonData()

Months = ['JAN','FEB','MAR','APR','May','June','July','August','SEP','OCT','NOV','DEC']

def get_stn_month_stats(yr):
    mon_day_dict = {}
    for x in range(1,366):
        DFW.doytodate(yr,x)
        if DFW.mon not in mon_day_dict.keys():
            mon_day_dict[DFW.mon] = []
        mon_day_dict[DFW.mon].append(DFW.day)
    return mon_day_dict


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
        avg = np.average(mon_vals)
        avg_list.append(avg)
        std = np.std(mon_vals)
        std_list.append(std)
    # make a list containing ['site','yr avg','yr std','']
    Summary = [sites_list[si],yr_avg,yr_std]
    Summary.extend(avg_list)
    Summary.extend(std_list)
    
    return Summary

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


def NormalizeIPW_Normal(SummaryStats,IPWvals,doy,missing_value_dict,sites_list,yr):
    DFW.doytodate(yr,doy)
    sites = sites_list[:,0]
#    sites = sites[np.logical_not(np.in1d(sites,missing_value_dict[int(doy) -1]))]
    IPWdataN = np.zeros(IPWvals.shape)
    for x in range(sites.shape[0]):
        IPWdataN[x,:] = (IPWvals[x,:] - SummaryStats[SummaryStats.site.values == sites[x]][DFW.mon + '_Avg'].values) / SummaryStats[SummaryStats.site.values == sites[x]][DFW.mon + '_Std'].values
#    IPWdataN = np.array(map(lambda x: (x[1:].astype('float') - SummaryStats[SummaryStats.site.values == x[0]][DFW.mon + '_Avg'].values)/SummaryStats[SummaryStats.site.values == x[0]][DFW.mon + '_Std'].values,IPWvals))
    return IPWdataN


def make_summary_dataframe(yr,IPWvals,missing_value_dict):
    cols = ['site']
    cols.extend(['Yearly_AVG','Yearly_STD'])
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
    stations = list(IPWvals[128,:,0])
    for day in missing_value_dict.keys():
        for stn in missing_value_dict[day]:
            IPWvals[day,stations.index(stn),1:] = np.array([str(np.nan)]*48)


def pad_storm_dates(storm_dates,yr):
    doy_list = map(lambda x: x[0], storm_dates)
    new_list = []
    print doy_list
    for d in doy_list:
        if d + 1 not in doy_list and d + 1 not in new_list:
            new_list.append(d + 1)
            DFW.doytodate(yr,d + 1)
            storm_dates.append((d + 1,DFW.yr,DFW.mon,DFW.day))
        if d - 1 not in doy_list and d - 1 not in new_list:
            new_list.append(d - 1)
            DFW.doytodate(yr,d - 1)
            storm_dates.append((d - 1,DFW.yr,DFW.mon,DFW.day))
    
    storm_dates.sort(key = lambda x: x[0])
    
            
            
    
    
def determine_storm_dates(yr = 16):
    if yr == 14:
        IPWvals = np.load('../data/2014IPW_data.npy')
        date_range = range(121,244)
    elif yr == 15:
        IPWvals = np.load('../data/2015IPW_data.npy')
        IPWvals[IPWvals == '-9.9'] = str(np.nan)
        date_range = range(121,244)
    elif yr == 16:
        IPWvals = np.load('../data/2016IPW_data.npy')
        date_range = range(122,245)        
    missing_value_dict = nan_stations(yr,IPWvals,date_range)
    SummaryStats = make_summary_dataframe(yr,IPWvals,missing_value_dict)
    set_ipw_nan(IPWvals,missing_value_dict)
    storm_dates = []
    ctr = 0
    for d in date_range:
        IPWvals_day = IPWvals[d,:,1:].astype('float')
        gpsX,gpsY,IPWsites,IPWvals_day = get_GPS_cartesian(missing_value_dict[d],IPWvals_day,DFW.sites)
        IPWvals_day = NormalizeIPW_Normal(SummaryStats,IPWvals_day,d,missing_value_dict,IPWsites,yr)
        tempArr = np.where(IPWvals_day > 2.0)
        if tempArr[0].size > 30:
            DFW.doytodate(yr,d + 1)
#            print 'Weather anomaly on ' + DFW.mon + '/' + DFW.day + '/' + '2016'
            storm_dates.append((d + 1,DFW.yr,DFW.mon,DFW.day))
            ctr+=1
#    print np.array(storm_dates,dtype = 'int')
#    pad_storm_dates(storm_dates,yr)
    return np.array(storm_dates,dtype = 'int')
    
#    np.save('../data/storm_dates_calculated_1_20' + str(yr),np.array(storm_dates,dtype = 'int'))

if __name__ == '__main__':
    yr_list = [14,15,16]
    storm_years = []
    for y in yr_list:
        print '-'*30
        storms = determine_storm_dates(y)
        print '-'*50
        print storms
#        np.save('../data/storm_dates_computed_1_20' + str(y),np.array(storms,dtype = 'int'))
        
#        storm_years.append(storms)


#for mon in [5,6,7,8]:
#    print Months[mon - 1] + ' & \t' + ','.join([str(x) for x in storm_years[0][storm_years[0][:,2] == mon,3]]) + '\t & \t' + ','.join([str(x) for x in storm_years[1][storm_years[1][:,2] == mon,3]]) + '\t' +  '\\' + '\\' + '\t' + '\\' + 'hline'

        


        


