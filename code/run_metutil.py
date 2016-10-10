# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:25:49 2015

@author: adityanagarajan
"""

import subprocess
import os
import DFWnet
import sys

def call_metutil(net,doy,ofile):
    x = DFWnet.CommonData()
    
    base_path = '/home/aditya/UMASS/DFWnetwork/'
    tempdir = base_path + net + '/2016/' + doy + os.sep
    
    for s in x.sites[x.sites[:,-1] == net,0]:
        
        zfile = 'z' + s + '6.' + doy
        
        if os.path.exists(tempdir + zfile):
            os.chdir(tempdir)
            subprocess.call(['sh_metutil','-f',ofile,'-z',zfile,'-i','1800'])
        else:
            print 'Missing data: '+ doy + '/'+ s
      
def split_files():
    with open('MissingData2.txt','rb') as f:
        getFiles = f.read()
    getFiles = getFiles.split('\n')
    temp = [item.split(':') for item in getFiles[:4]]
    temp_dict = dict((k.strip(),v.strip()) for k,v in temp)
    for n in temp_dict.keys():
        
        temp_dict[n] = [k.strip() for k in temp_dict[n].strip('[]').replace("'"," ").strip().split(',')]
        
    return temp_dict
    
file_dict = split_files()
net = sys.argv[1]
doy_list = [str(x).zfill(3) for x in range(122,213)]

file_path = '/home/aditya/UMASS/DFWnetwork/'

dict1 = {}

def dict_func(this,b): 
    this[b] = 0

map(lambda x: dict_func(dict1,x),doy_list)

doy_list = [str(x) for x in range(122,213)]

for n in sys.argv[1:]:
    FATAL_DOY = 0
    DOYS = []
    for d in doy_list:
        if os.path.exists(file_path + n+ '/2016' + os.sep + d):
            ofile = 'o' + n + 'a.' + d
            
            if os.path.exists(file_path + n+ '/2016' + os.sep + d + os.sep + ofile):
                num = call_metutil(n,d,ofile)
            else:
                FATAL_DOY+=1
                DOYS.append(d)
        else:
            dict1[d]+=1

Missing_DOY = []
for x in dict1.keys():
    if dict1[x] > 0:
        Missing_DOY.append(x)

Missing_DOY.sort()


print 'Number of missing files: %s'%len(Missing_DOY)
print 'Missing DOY Files: '
print Missing_DOY

print 'No o files: %s'%FATAL_DOY

print 'Missing o files: '

print DOYS
        
        


'''


/home/aditya/UMASS/DFWnet/net3/2014

sh_metutil -f o* -z z* -i 1800
'''

    
