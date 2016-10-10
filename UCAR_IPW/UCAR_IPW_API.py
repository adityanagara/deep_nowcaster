# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 16:44:10 2016

@author: adityanagarajan
"""

import os
import numpy as np
import urllib2
import pandas as pd
import DFWnet
from matplotlib import pyplot as plt

base_url = 'http://www.suominet.ucar.edu/data/pwvConusHourly/'
sites = ['txda','txdc','txde','txco','txty','txhm','txwa','txwf','txno','txpa']
DFW = DFWnet.CommonData()
pressure = DFW.Prvals


def parse_UCAR_files(doy,yr):
    UCARvals = pd.DataFrame(columns = ['site','time','IPW','Press','Temp','Rhum'])
    ctr = 0
    files_list = ['SUOh_20' + str(yr) + '.{0}.{1}.00.PWV'.format(doy,str(x).zfill(2)) for x in range(24)]
    
    for f in files_list:
        response = urllib2.urlopen(base_url + f)
        html = response.read().split('\n')
        html = map(lambda x: x.split(),html)
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
                    print 'No data for ' + s + ' in file ' + f
                    ctr += 1
    return UCARvals
    
def stn_index_converter(stn,ar):
    return np.where(ar[128,:,0] == stn)[0][0]
    

def main(ipw,yr = 16):
    doy = 150
    UCARdata = parse_UCAR_files(str(doy),yr)
    print ipw.shape
    print UCARdata.shape
    for s in sites:
        plt.figure()
        plt.plot(UCARdata[UCARdata.site.values == s].IPW.values,'r-',label = 'UCAR published')
        idx = stn_index_converter(s,ipw)
        plt.plot(ipw[doy - 1,idx,1:].reshape(48,).astype('float'),'b-', label = 'Locally processed')
        plt.title('Verification for site %s and day %s'%(s,str(doy)))
        plt.legend()

if __name__ == '__main__':
    ipw = DFW.IPWvals_2016
    main(ipw)

    



