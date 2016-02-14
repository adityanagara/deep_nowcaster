# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:19:51 2016

@author: adityanagarajan
"""

import BuildDataSet
import cPickle
import os
import sys

data_builder = BuildDataSet.dataset(Threshold = 'bin')
PixelPoints = data_builder.sample_random_pixels()

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

def build_points_fields():
    num_points = 1200
    prev_pt_tr = 0

    for tr_pt in range(0,num_points,40):
        print 'Building set %d '%tr_pt
        train_set = data_builder.make_points_frames(PixelPoints[prev_pt_tr:tr_pt + 40])
        print 'Done making training set'
        X_Y_train = data_builder.arrange_frames(train_set)
        temp_file = open('data/dataset2/points_' + str(prev_pt_tr) + '_'+ str(tr_pt + 40) + '.pkl','wb')
        cPickle.dump(X_Y_train,temp_file,protocol=cPickle.HIGHEST_PROTOCOL)
        temp_file.close()
        prev_pt_tr += 40

def load_points_fields():
    base_path = 'data/dataset2/'
    file_list = os.listdir(base_path)
    file_list = filter(lambda x: x[-4:] == '.pkl',file_list)
    massive_data = []
    print(file_list)
    for f in file_list:
        print f
        open_file = open(base_path + f,'rb')
        load_file = cPickle.load(open_file)
        open_file.close()
        massive_data.append(load_file)
#        print('Done loading file %s '%f)
    for data in massive_data:
        print(data[0].shape,data[1].shape)
    print('Successfully Loaded 50gigs of data!! on da cluster')
    
load_points_fields()

    
    