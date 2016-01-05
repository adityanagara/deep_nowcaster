# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:44:44 2015

@author: adityanagarajan
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import random
import cPickle

import nowcast
import BuildDataSet

region_dict = {}

region_dict['region1'] = (range(33,50),range(50,66),'r*')
region_dict['region2'] = (range(50,66),range(50,66),'g*')
region_dict['region3'] = (range(33,50),range(33,50),'b*')
region_dict['region4'] = (range(50,66),range(33,50),'y*')

now_caster = nowcast.BuildNowcaster()
data_builder = BuildDataSet.dataset()

#fill_domain = ([x for x in range(33,66)if x not in range(50,55)],[y for y in range(33,66) if y not in range(50,55)])


def plot_domains(PixelX,PixelY,marker):
    
    central_chunk = (range(46,54),range(46,54))
    
    #Pull out the noisy regions of the radar
    central_chunk_points = [(x_,y_) for x_ in central_chunk[0] for y_ in central_chunk[1]]
    
    PixelPoints = [(x,y) for x in PixelX for y in PixelY]
    
    PixelPoints = [pairs for pairs in PixelPoints if pairs not in central_chunk_points]
    
    random.seed(12345)
        
    PixelPoints = [PixelPoints[x] for x in random.sample(range(4292),1500)]
    
    PixelPoints = np.array(PixelPoints)
    
    print PixelPoints.shape
    
    gridX = np.arange(-150.0,151.0,300.0/(100-1))
    gridY = np.arange(-150.0,151.0,300.0/(100-1))
    for p in PixelPoints:
        plt.plot(gridX[p[0]],gridY[p[1]],marker)

    plt.xlabel('Easting')
    
    plt.ylabel('Northing')
    
    return PixelPoints
        #Plot domain

fill_domain = (range(17,83),range(17,83))

PixelPoints = plot_domains(fill_domain[0],fill_domain[1],region_dict['region1'][2])

sorted_days = data_builder.club_days()

days_in_sorted = sorted_days.keys()

days_in_sorted.sort()

print days_in_sorted

'''
>>> f = file('obj.save', 'wb')
>>> cPickle.dump(my_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
>>> f.close()

'''

IPW_Refl_points = []
save_ctr = 1

#for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
#    print 'Building data set for point: (%d,%d)'%(x_,y_)
#    for set_ in days_in_sorted:
#        temp_ipw_file_list = filter(lambda x: x[7:10] in sorted_days[set_],data_builder.IPWfiles)
#        temp_radar_file_list = filter(lambda x: x[9:12] in sorted_days[set_],data_builder.Radarfiles)
#        tmp_array = data_builder.build_features_and_truth(temp_ipw_file_list,temp_radar_file_list,x_,y_)
#        IPW_Refl_points.append(tmp_array)
#    if save_ctr % 100 == 0:
##        save_to_pkl_file = file(data_builder.TrainTestdir + 'RandomPoints/' + 'IPWpoints_%d.pkl'%save_ctr,'wb')
##        cPickle.dump(IPW_Refl_points,save_to_pkl_file,protocol=cPickle.HIGHEST_PROTOCOL)
##        save_to_pkl_file.close()
#        print 'Batch Done %d'%save_ctr
#        # Release the list
#        IPW_Refl_points = []
#    save_ctr+=1

            
        
#        np.save(data_builder.TrainTestdir + 'RandomPoints/' + 'IPWGritPoint_%d_%d_'%(x_,y_) +set_ +'.npy',tmp_array[0])
#        
#        np.save(data_builder.TrainTestdir + 'RandomPoints/' + 'RadarGritPoint_%d_%d_'%(x_,y_) +set_ +'.npy',tmp_array[1])


#Pull out the noisy regions of the radar


#plot_domains(fill_domain[0],fill_domain[1],region_dict['region1'][2])
#
plt.grid()

plt.xlim((-150.0,150.0))

plt.ylim((-150.0,150.0))

plt.title('1500 Random Points')

plt.show()



