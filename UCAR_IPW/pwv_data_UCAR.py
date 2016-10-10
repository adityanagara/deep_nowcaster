# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:46:40 2015

@author: aditya
"""

import urllib2

import csv

import pandas as pd



'''
 Parent Directory
SUOh_2008.080.00.00.PWV
SUOh_2008.080.01.00.PWV
SUOh_2008.080.02.00.PWV
SUOh_2008.080.03.00.PWV
SUOh_2008.080.04.00.PWV
SUOh_2008.080.05.00.PWV
SUOh_2008.080.06.00.PWV
SUOh_2008.080.07.00.PWV
SUOh_2008.080.08.00.PWV
SUOh_2008.080.09.00.PWV
SUOh_2008.080.10.00.PWV

'''

UCAR_files = ['SUOh_2010.{0}.{1}.00.PWV'.format(str(x).zfill(3),str(y).zfill(2)) for x in range(1,366) for y in range(25)]

reader = csv.reader(open('../data/KFWS_230km_sites.csv','rb'))
KFWS_sites = [x[0] for x in reader]
temp_site = 'SUOh_2015.003.11.00.PWV'
base_url = 'http://www.suominet.ucar.edu/data/pwvConusHourly/'
temp_file = urllib2.urlopen(base_url + temp_site)
data = temp_file.read()

data = data.split('\n')

fr = pd.DataFrame(map(lambda x: x.split(),data[2:]),columns = data[0].split())

k = fr.Site.unique()
k = [x.lower() for x in list(k)[:-1]]
true_sites = [x for x in KFWS_sites if x in k]

print true_sites







