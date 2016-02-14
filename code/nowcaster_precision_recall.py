# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:25:06 2016

@author: adityanagarajan
"""

import numpy as np
import sys
import os
import cPickle as pkl
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import average_precision_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score
#import DCNN_network
import lasagne

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
network_file = sys.argv[1]

def convert_gpu_cpu(network_file):
    print(network_file)
    from lasagne.layers import dnn
    print('Starting to convert gpu network to cpu network')
    n_file = file('output/' + network_file,'rb')
    network = pkl.load(n_file)
    n_file.close()
    params = lasagne.layers.get_all_param_values(network)
    cpu_n_file = file('output/' + 'CPU_' + network_file,'rb')
    pkl.dump(params,protocol = pkl.HIGHEST_PROTOCOL)
    cpu_n_file.close()
    print('Done!')
#def load_network_parameters():

convert_gpu_cpu(network_file)
