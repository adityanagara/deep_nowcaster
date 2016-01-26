# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:19:51 2016

@author: adityanagarajan
"""

import os
import BuildDataSet
import cPickle

data_builder = BuildDataSet.dataset(Threshold = None)
PixelPoints = data_builder.sample_random_pixels()

train_set = data_builder.make_points_frames(PixelPoints[:40])

print 'Done making training set'


X_train,Y_train = data_builder.arrange_frames(train_set)




