# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:58:15 2016

@author: adityanagarajan
Note: This code plots the results of training and validation error from the 
output csv files
"""

from matplotlib import pyplot as plt
import numpy as np
import os

#file_list = ['exp_1000_10_gpu2.csv','exp_1000_20_gpu2.csv','exp_1000_30_gpu2.csv','exp_1000_40_gpu2.csv']
#line_color = ['r','g','b','k']
#model = ['10 points','20 points','30 points','40 points']
#
#for f,c,m in zip(file_list,line_color,model):
#    results_array = np.loadtxt('output/' + f,dtype = 'S',delimiter = ',',skiprows = 1)
#    results_array = results_array[:-1].astype('float')
#    
#    plt.plot(results_array[:,0],results_array[:,1],c + '--',label = m + ' training mean squared error')
#    
#    plt.plot(results_array[:,0],results_array[:,2],c, label = m + ' validation mean squared error')
#    
#
#plt.xlabel('epochs')
#plt.ylabel('Mean Squared Error')
#plt.title('2 convolution layers and 1 fully connected layer')
#plt.legend()
#
#plt.grid()
#plt.show()

#print os.getcwd()
nn_experiments = np.load('../output/1_CNN_experiments/performance_metrics_1.pkl')
cnn_experiments = np.load('../output/1_CNN_experiments/performance_metrics_max_pool1.pkl')


print cnn_experiments

PODs_nn = []
PODs_cnn = []
CSIs_nn = []
FARs_nn = []
FARs_cnn = []
CSIs_cnn = []

for i in range(len(cnn_experiments)):
    PODs_cnn.append(cnn_experiments[i][0][1])
    PODs_nn.append(nn_experiments[i][0][1])
    FARs_nn.append(nn_experiments[i][0][2])
    CSIs_nn.append(nn_experiments[i][0][3])
    FARs_cnn.append(cnn_experiments[i][0][2])
    CSIs_cnn.append(cnn_experiments[i][0][3])

plt.figure()
plt.plot(PODs_cnn,label = 'POD CNN')
plt.plot(PODs_nn,label = 'POD NN')
plt.plot(FARs_cnn,label = 'FAR CNN')
plt.plot(FARs_nn,label = 'FAR NN')
#plt.plot(CSIs_cnn,label = 'CSI CNN')
#plt.plot(CSIs_nn,label = 'CSI NN')
plt.title('Test scores for month June 2014')
plt.legend()

#def ground_truth_distribution():
#    '''This script plots the distribution between rain cases and no rain cases
#    '''
#    base_path = '../data/TrainTest/points/'
#    file_list = os.listdir(base_path)
#    file_list = filter(lambda x: x[-4:] == '.pkl',file_list)
#    arr = np.load(base_path + file_list[0])
#    print len(arr)
#    ground_truthes = []
#    for a in range(len(arr)):
#        print arr[a][2].shape
##        ground_truthes.append(arr[a][2])
#    
#    return ground_truthes
#        
##    print file_list
#
#ground_truthes = ground_truth_distribution()
#
#print len(ground_truthes)





'''
For neural nets 
performance_metrics[ep].append([val_acc / ctr_val,
                                val_pod / ctr_val,
                                val_far / ctr_val,
                                val_csi / ctr_val])

For CNN
performance_metrics[ep].append([val_acc / val_batches_ctr,
                                val_POD / val_batches_ctr,
                                val_FAR / val_batches_ctr,
                                val_CSI / val_batches_ctr])                        
'''








