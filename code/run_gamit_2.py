# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 23:18:01 2015

@author: adityanagarajan
"""

import os
import sys
import subprocess

net = sys.argv[1]
doy = sys.argv[2]

def run_gamit(doy,net):
    
    os.chdir('/home/aditya/UMASS/DFWnetDB' + os.sep + net + '/2015')
    
    subprocess.call(['sh_gamit','-expt',net,'-d','2015',doy,'-orbit','IGSF','-met'])
    print '%%%% Processing for  ' + doy + '  is complete %%%%'

doy_list = [str(x).zfill(3) for x in range(351,365)]

initial = os.getcwd()

run_gamit(doy,net)

#for doy in doy_list:
#    run_gamit(doy,net)

os.chdir(initial)
    




