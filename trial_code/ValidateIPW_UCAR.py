# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:43:02 2015

@author: adityanagarajan
"""

import urllib2 
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



baseURL = 'http://www.suominet.ucar.edu/data/pwvConusHourly/'

#sites = np.loadtxt('/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/data/KFWS_GPS_ASOS_locations.csv',dtype='S',delimiter = ',',skiprows = 1)

'''
SUOh_2014.128.00.00.PWV
SUOh_2014.128.01.00.PWV
SUOh_2014.128.02.00.PWV
SUOh_2014.128.03.00.PWV
SUOh_2014.128.04.00.PWV
SUOh_2014.128.05.00.PWV
SUOh_2014.128.06.00.PWV
SUOh_2014.128.07.00.PWV
SUOh_2014.128.08.00.PWV
SUOh_2014.128.09.00.PWV
SUOh_2014.128.10.00.PWV
SUOh_2014.128.11.00.PWV
SUOh_2014.128.12.00.PWV
SUOh_2014.128.13.00.PWV
SUOh_2014.128.14.00.PWV
SUOh_2014.128.15.00.PWV
SUOh_2014.128.16.00.PWV
SUOh_2014.128.17.00.PWV
SUOh_2014.128.18.00.PWV
SUOh_2014.128.19.00.PWV
SUOh_2014.128.20.00.PWV
SUOh_2014.128.21.00.PWV
SUOh_2014.128.22.00.PWV
SUOh_2014.128.23.00.PWV
'''

files_list = ['SUOh_2014.128.{0}.00.PWV'.format(str(x).zfill(2)) for x in range(24) ]

out_file = '/Users/adityanagarajan/Summer_2015/ConvectiveInitiation/output/IPW20140508.csv'

#response = urllib2.urlopen(baseURL + files_list[3])
#html = response.read().split('\n')
#html = map(lambda x: x.split(),html)

UCARvals = pd.DataFrame(columns = ['site','time','IPW','Press','Temp','Rhum'])
ctr = 0

sites = ['txda','txdc','txde','txco','txty','txhm','txwa']

for f in files_list:
    response = urllib2.urlopen(baseURL + f)
    html = response.read().split('\n')
    html = map(lambda x: x.split(),html)
#    print 'Parsing site ' + f
    for s in sites:
        print 'Looking for ' + s.upper()
        
    
        vals = filter(lambda x: x[0] == s.upper(),html[2:-1])
        if vals:
            for v in vals:
                UCARvals.loc[ctr] = [s,v[1],v[3],v[9],v[10],v[11]]
                ctr+=1
        else:
            for i in range(2):
                UCARvals.loc[ctr] = [s,np.nan,np.nan,np.nan,np.nan,np.nan]
#                UCARvals.loc[ctr] = [0.0,np.nan,np.nan,np.nan,np.nan,np.nan]
                print 'No data for ' + s + ' in file ' + f
                ctr+=1

vals = np.loadtxt(out_file,delimiter=',',skiprows=1,dtype='S')

plt.figure()
plt.plot(UCARvals[UCARvals.site.values == 'txda'].IPW.values,'r-')
plt.plot(vals[vals[:,0] == 'txda',1:].reshape(48,).astype('float'),'b-')

plt.figure()

plt.plot(UCARvals[UCARvals.site.values == 'txde'].IPW.values,'r-')
plt.plot(vals[vals[:,0] == 'txde',1:].reshape(48,).astype('float'),'b-')


        














