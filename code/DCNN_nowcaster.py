# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:48:23 2016

@author: adityanagarajan
"""

import numpy as np
import os
from theano import tensor as T
import BuildDataSet
import nowcast

import random

from matplotlib import pyplot as plt

now_caster = nowcast.BuildNowcaster()
data_builder = BuildDataSet.dataset()

# We are going to choose a sub domain in our total 300x300 domain in DFW
fill_domain = (range(17,83),range(17,83))
#PixelPoints = plot_domains(fill_domain[0],fill_domain[1],region_dict['region1'][2])

PixelX = fill_domain[0]
PixelY = fill_domain[1]

# Pull the central chunk of points out
central_chunk = (range(46,54),range(46,54))

central_chunk_points = [(x_,y_) for x_ in central_chunk[0] for y_ in central_chunk[1]]

# Pair up the pixel points
PixelPoints = [(x,y) for x in PixelX for y in PixelY]

# Remove all central points from the pair
PixelPoints = [pairs for pairs in PixelPoints if pairs not in central_chunk_points]

# Randomely sample 1500 pairs of points
random.seed(12345)   
PixelPoints = [PixelPoints[x] for x in random.sample(range(4292),1500)]

PixelPoints = np.array(PixelPoints)

# Plot the domain to check the points
data_builder.plot_domain(PixelPoints)
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

sorted_days = data_builder.club_days()

days_in_sorted = sorted_days.keys()

days_in_sorted.sort()

print days_in_sorted

def make_mini_batches()








