# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:30:41 2016

@author: adityanagarajan
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import DFWnet

'''
>>> y= array([1, 1, 1, NaN, NaN, 2, 2, NaN, 0])
>>>
>>> nans, x= nan_helper(y)
>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
>>>
>>> print y.round(2)
[ 1.    1.    1.    1.33  1.67  2.    2.    1.    0.  ]
'''

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
                    print 'case unhandled...' + ' ' + str(doy) + stn
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
                    print 'case unhandled...'
                    print 'There is something wrong with ' + stn + ' ' + str(doy)
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
    stations = IPWvals[doy,:,0]
#    IPWvals = IPWvals[:,:,1:].astype('float')
    IPWvals = IPWvals[doy,:,1:].astype('float')
    IPWvals[IPWvals < 0.0] = 0.1
    for i,stn in enumerate(stations):
        if stn not in nan_sites:
            first_order = np.diff(IPWvals[i,:]) / IPWvals[i,:-1]
            # only impute for fluctuations less than 4            
            if np.any(first_order <= -0.5,axis = 0) and np.where(first_order <= -0.5)[0].shape[0] < 4:
                print IPWvals[i,:]
                bad_val_imputer(doy,stn,IPWvals[i,:],first_order,nan_sites)
                print IPWvals[i,:]
            elif np.any(first_order <= -0.5,axis = 0) and np.where(first_order <= -0.5)[0].shape[0] >= 4:
                print 'Real Bad...Drop this shit ' + stn
                print doy,stn,np.where(first_order < -0.5)[0].shape
                nan_sites.append(stn)

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
        drop_stations(yr,i,IPWvals,missing_data[i])
    return missing_data


def main():
    yr = 16
    date_range = range(121,214) # DOY 122 - 244 -- 121,244
    IPWvals = np.load('../data/2016IPW_data.npy')
    missing_value_dict = nan_stations(yr,IPWvals,date_range)
    print IPWvals
    print missing_value_dict
    
if __name__ == '__main__':
    main()