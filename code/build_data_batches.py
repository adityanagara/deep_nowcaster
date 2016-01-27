# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:19:51 2016

@author: adityanagarajan
"""

import BuildDataSet
import cPickle

data_builder = BuildDataSet.dataset(Threshold = None)
PixelPoints = data_builder.sample_random_pixels()


num_points = 1280
prev_pt_tr = 1200

for tr_pt in range(1200,num_points,40):
    print 'Building set %d '%tr_pt
    train_set = data_builder.make_points_frames(PixelPoints[prev_pt_tr:tr_pt + 40])
    print 'Done making training set'
    X_Y_train = data_builder.arrange_frames(train_set)
    temp_file = open('data/dataset2/points_val_' + str(prev_pt_tr) + '_'+ str(tr_pt + 40) + '.pkl','wb')
    cPickle.dump(X_Y_train,temp_file,protocol=cPickle.HIGHEST_PROTOCOL)
    temp_file.close()
    prev_pt_tr += 40



