# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 14:49:50 2014

@author: aditya
"""


#import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot
import os
import time


def validate(day):
    cnvl_file='/home/aditya/UMASS/Validation/WV/met_cnvl.14' + day
    zfw1_file='/home/aditya/UMASS/Validation/WV/met_zfw1.14' + day
    print cnvl_file
    print zfw1_file
    read_file_cnvl=open(cnvl_file)
    read_file_zfw1=open(zfw1_file)
    lines_cnvl=read_file_cnvl.readlines()
    lines_zfw1=read_file_zfw1.readlines()
    last=len(lines_cnvl)
    print last
    WV_Values_cnvl=[]
    WV_Values_zfw1=[]
    time_list=[]
    axis_time_dict={}
    for x in range(4,last):
        WV_Values_cnvl.append(lines_cnvl[x][50:55])
        WV_Values_zfw1.append(lines_zfw1[x][50:55])
        time_list.append(lines_cnvl[x][10:18])
    time_list.pop()
    #print time_list
    hour_axis_time=[]
    for i in range(0,288):
        temp=float(i)/12.0
        hour_axis_time.append(temp)
    #print hour_axis_time
    #print time_list
    for i in range(0,288):
        #print time_list[i]
        #print hour_axis_time[i]
        axis_time_dict[time_list[i]] = hour_axis_time[i]
    print WV_Values_cnvl
    print WV_Values_zfw1
#    WV_Values_cnvl.pop()
#    WV_Values_zfw1.pop()
    print len(WV_Values_cnvl)
    print len(WV_Values_zfw1)
    print len(hour_axis_time)
    print hour_axis_time
    #pyplot.plot(hour_axis_time,WV_Values_cnvl,'r')
    #pyplot.plot(hour_axis_time,WV_Values_zfw1,'b')
    #pyplot.show()
    save_plot='/home/aditya/UMASS/Validation/WV/cnvl_plot_123.png'
    print save_plot
    #pyplot.savefig(save_plot)
    return WV_Values_cnvl,WV_Values_zfw1
    

#day=['302','303','304','305','306','307']
#day_dict_cnvl ={}
#list_1=[]
#list_2=[]
#for d in day:
#    list_1,list_2=validate(d)
#    print list_1
#    print list_2

validate('302')

    
    
#WV_cnvl=[]
#WV_zfw1=[]
#x=[]
#
#for d in day:
#    init=0.0
#    validate(d)
#    for dec in range(0,288):
#        k = float(d) + init
#        x.append(k)
#        init=init+0.003472222222222222
#        print str(val) + str(dec)



