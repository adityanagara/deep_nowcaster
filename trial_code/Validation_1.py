# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 13:32:39 2014

@author: aditya
"""

from matplotlib import pyplot
import os
import time


WV_cnvl_final=[]
WV_zfw1_final=[]
time_cnvl_final=[]
time_zfw1_final=[]


def time_list():
    temp_file='/home/aditya/UMASS/Validation/WV/time_list.txt'
    read_temp=open(temp_file)
    temp_lines=read_temp.readlines()
    last_temp=len(temp_lines)
    time_list=[]
    for x in range(4,last_temp):
        time_list.append(temp_lines[x][10:18])
    print time_list
    return time_list

k=time_list()

def validate(day):
    global WV_cnvl_final
    global WV_zfw1_final
    global time_cnvl_final
    global time_zfw1_final
    cnvl_file='/home/aditya/UMASS/Validation/WV/met/met_2/met_cnvl.14' + day
    zfw1_file='/home/aditya/UMASS/Validation/WV/met/met_2/met_zfw1.14' + day
    #print cnvl_file
    #print zfw1_file
    read_file_cnvl=open(cnvl_file)
    read_file_zfw1=open(zfw1_file)
    lines_cnvl=read_file_cnvl.readlines()
    lines_zfw1=read_file_zfw1.readlines()
    WV_Values_cnvl=[]
    WV_Values_zfw1=[]
    WV_cnvl_time={}
    WV_zfw1_time={}
    k=float('nan')
    plot_WV_vals_cnvl=[]
    real_time=time_list()
    lines_cnvl.pop()
    lines_zfw1.pop()
    last_cnvl=len(lines_cnvl)
    last_zfw1=len(lines_zfw1)
    for x in range(4,last_cnvl):
        WV_cnvl_time[lines_cnvl[x][10:18]]=lines_cnvl[x][50:55]
        #WV_zfw1_time[lines_zfw1[x][10:18]]=lines_zfw1[x][50:55]
        WV_Values_cnvl.append(lines_cnvl[x][50:55])
        #WV_Values_zfw1.append(lines_zfw1[x][50:55])
    for x in range(4,last_zfw1):
        WV_zfw1_time[lines_zfw1[x][10:18]]=lines_zfw1[x][50:55]
        WV_Values_zfw1.append(lines_zfw1[x][50:55])
        #time_list.append(lines_cnvl[x][10:18])
    for i in real_time:
        print i
        if i in WV_cnvl_time.iterkeys():
            plot_WV_vals_cnvl.append(WV_cnvl_time[i])
            WV_cnvl_final.append(WV_cnvl_time[i])
        else:
            plot_WV_vals_cnvl.append(float('nan'))
            WV_cnvl_final.append(float('nan'))
        if i in WV_zfw1_time.iterkeys():
            WV_zfw1_final.append(WV_zfw1_time[i])
        else:
            WV_zfw1_final.append(float('nan'))
            
            #print k + ' ' + v
    #print plot_WV_vals_cnvl
    #print len(plot_WV_vals_cnvl)
    init=0.0
    for dec in range(0,288):
        temp_time = float(day) + init
        time_cnvl_final.append(temp_time)
        init=init+float(1.0/288.0)
        #print str(day) + str(dec)
    #print time_cnvl_final
    #print len(time_cnvl_final)
'''
    Compute the x axis time
'''
#days=['302','303','304','305','306','307']
days = ['308','309','310','311']
for d in days:
    validate(d)

print time_cnvl_final

pyplot.plot(time_cnvl_final,WV_cnvl_final,'r',label='cnvl')
pyplot.plot(time_cnvl_final,WV_zfw1_final,'b',label='zfw1')
#print WV_cnvl_final
#print WV_zfw1_final
pyplot.title('Integrated Precipitable Water for cnvl and zfw1 for days 308-312')
pyplot.xlabel('Day number 2014')
pyplot.ylabel('IPW (mm)')
pyplot.legend(loc='upper right',prop={'size':16})
save_plot='/home/aditya/UMASS/Validation/WV/cnvl_zfw1_val_20141223.jpg'
#print save_plot
#pyplot.savefig(save_plot)
pyplot.show()
#print len(k)
    
        

    