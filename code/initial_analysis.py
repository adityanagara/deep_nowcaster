# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:01:13 2016

@author: adityanagarajan
"""

import numpy as np
import pandas as pd

import os
import DFWnet

DFW = DFWnet.CommonData()

def NormalizeIPW_Normal(SummaryStats,IPWvals,doy,missing_value_dict,sites_list,yr):
    DFW.doytodate(yr,doy)
#    print DFW.mon
    sites = sites_list[:,0]
    sites = sites[np.logical_not(np.in1d(sites,missing_value_dict[int(doy) -1]))]
    IPWdataN = np.zeros(IPWvals.shape)
    for x in range(sites.shape[0]):
        IPWdataN[x,:] = (IPWvals[x,:] - SummaryStats[SummaryStats.site.values == sites[x]][DFW.mon + '_Avg'].values) / SummaryStats[SummaryStats.site.values == sites[x]][DFW.mon + '_Std'].values
    return IPWdataN

def get_stn_month_stats(yr):
    mon_day_dict = {}
    for x in range(1,366):
        DFW.doytodate(14,x)
        if DFW.mon not in mon_day_dict.keys():
            mon_day_dict[DFW.mon] = []
        mon_day_dict[DFW.mon].append(DFW.day)
    return mon_day_dict


def get_stats(si,IPWvals,sites,yr):
    mon_day_dict = get_stn_month_stats()
    sites_list = sites[:,0]
    avg_list = []
    std_list = []
    # Find the yearly average and standard deviations for each station
    yr_vals = np.hstack(IPWvals[:,si,1:]).astype('float')
    yr_avg = np.average(yr_vals[~np.isnan(yr_vals)])
    yr_std = np.std(yr_vals[~np.isnan(yr_vals)])
    # find the monthly average and standard deviations for each station
    for mon in sorted(mon_day_dict.keys()):
        DFW.date2doy(yr,int(mon),int(mon_day_dict[mon][0] ))
        start = int(DFW.doy)
        DFW.date2doy(yr,int(mon),int(mon_day_dict[mon][-1]))
        end = int(DFW.doy)
        mon_vals = np.hstack(IPWvals[start - 1: end ,si,1:])
        avg = np.average(mon_vals[~np.isnan(mon_vals.astype('float'))].astype('float'))
        avg_list.append(avg)
        std = np.std(mon_vals[~np.isnan(mon_vals.astype('float'))].astype('float'))
        std_list.append(std)
    
    # make a list containing ['site','yr avg','yr std','']
    Summary = [sites_list[si],yr_avg,yr_std]
    Summary.extend(avg_list)
    Summary.extend(std_list)
    return Summary

def nan_stations(IPWvals):
    '''Takes inpyt the array of shape 365,44,49 array and returns
    a distionary of missing_data[doy] = [station list]'''
    missing_data = {}
    for i in range(IPWvals.shape[0]):
        day_vals = IPWvals[i,...]
        if i not in missing_data.keys():
            missing_data[i] = []
        for x in day_vals:
            if x[x == str(np.nan)].size > 0:
                missing_data[i].append(x[0])
    return missing_data

def drop_stations(yr,doy,IPWvals,nan_sites):
    '''We are going to make sure that no site shows us a 100% increase or 
    decrease in water vapor from the previous value.'''
    stations = IPWvals[doy - 1,:,0]
    IPWvals = IPWvals[doy - 1,:,1:].astype('float')
    for i,x in enumerate(IPWvals):
        temp_vec = np.abs(np.diff(x)) / x[:-1]
        if np.any(temp_vec >= 1.0,axis = 0):
            nan_sites.append(stations[i])

# the total summary function gets the summary between 2014 and 2015
#def total_summary(summaries):
def make_summary_dataframe(yr,IPWvals):
    cols = ['site']
    cols.extend(['Yearly_AVG','Yearly_STD'])
    cols.extend(['{0}_Avg'.format(str(x).zfill(2)) for x in range(1,13)])
    cols.extend(['{0}_Std'.format(str(x).zfill(2)) for x in range(1,13)])
    itr=0
    SummaryStats = pd.DataFrame(columns = cols)
    for si in range(44):
        temp_list = get_stats(si,IPWvals,DFW.sites,yr)
        SummaryStats.loc[itr] = temp_list
        itr+=1
    return SummaryStats
# Drop site OKAR from 2014 and 2015 data set. RINXE not available
    
def get_anomalous_days(yr,IPW,summaries,nan_sites):
    Anomalies = []
    # DOY 121 - 243 is the storm season DOY
    for doy in range(121,243):
        IPWvals = IPW[doy,:,1:].astype('float')
    
        IPWvals = NormalizeIPW_Normal(summaries[yr],IPWvals,doy,nan_sites,DFW.sites,yr)
    
        tempArr = np.where(IPWvals > 2.0)
        if tempArr[0].size > 30:
        
            DFW.doytodate(yr,doy)
        
            print 'Weather anomaly on ' + DFW.mon + '/' + DFW.day + '/' + '2014'
            Anomalies.append([doy,yr,int(DFW.mon),int(DFW.day)])
    np.save('../data/storm_dates_20'+ str(yr)+'_.npy',np.array(Anomalies))
    return np.array(Anomalies)


def main():
        
    IPW_2014 = DFW.IPWvals_2014
    nan_2014 = nan_stations(IPW_2014)
    
    IPW_2015 = DFW.IPWvals_2015
    IPW_2015[IPW_2015 == '-9.9'] = str(np.nan)
    nan_2015 = nan_stations(IPW_2015)
    storm_dates = np.load('../data/storm_dates_2015.npy').astype('int')
    for d in storm_dates:
        drop_stations(15,d[0],IPW_2015,nan_2015[d[0] - 1])
        
        print d,nan_2015[d[0] -1]
    

#    IPW = {}
#    IPW[14] = IPW_2014
#    IPW[15] = IPW_2015
#    print IPW_2014.shape,IPW_2015.shape
#    print IPW_2014.shape,IPW_2015.shape
#    IPW_all = np.stack((IPW_2014,IPW_2015),axis=0)
#    summaries = {}
#    # Get summaries for 2014 and 2015 seperately
#    for i in [14,15]:
#        summaries[i] = make_summary_dataframe(i,IPW[i])
#    print summaries[14][['site','05_Avg','06_Avg']]        

if __name__ == '__main__':
    main()

'''
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

'''




