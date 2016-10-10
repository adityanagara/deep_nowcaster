# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:45:02 2016

@author: adityanagarajan
"""

import numpy as np
from matplotlib import pyplot as plt
import cPickle as pkl
plt.ion()
base_path = '../output/thesis_results/'


def make_plots_classification(lab,file_name):
        
    f1 = file(base_path + file_name)
    results = pkl.load(f1)
    f1.close()
    print file_name
    val_loss = []
    val_acc = []
    x_axis = range(200)
    train_loss = []
    for i in range(1,201):
        val_loss.append(results[i][0]['val_loss'])
        val_acc.append(results[i][0]['val_acc'])
        train_loss.append(results[i][0]['train_loss'])
    plt.plot(x_axis,val_loss[:],lab[2],label = lab[0] + ' validation loss')
    plt.plot(x_axis,train_loss[:],lab[1],label = lab[0] + ' training loss')
#    plt.title('Training and validation loss for 200 epochs')
    plt.ylim((0.08,0.20))
    plt.xlabel('epoches',size = 18)
    plt.ylabel('Cross Entropy Loss',size = 18)
    plt.legend(handlelength=5, borderpad=1.2, labelspacing=1.2,fontsize = 'x-large')
    print len(results)

def make_plots_regression(lab,file_name):
    f1 = file(base_path + file_name)
    results = pkl.load(f1)
    f1.close()
    print file_name
    val_loss = []
    x_axis = range(160)
    train_loss = []
    for i in range(1,161):
        val_loss.append(results[i][0]['val_loss'])
        train_loss.append(results[i][0]['train_loss'])
    plt.plot(x_axis,val_loss[:],lab[2],label = lab[0] + ' validation loss')
    plt.plot(x_axis,train_loss[:],lab[1],label = lab[0] + ' training loss')
#    plt.title('Training and validation loss for 200 epochs')
    plt.ylim((20.0,60.0))
    plt.xlabel('epoches')
    plt.ylabel('Squared error loss')
    plt.legend(handlelength=5, borderpad=1.2, labelspacing=1.2)
    print len(results)
    
def call_make_plots_classification(list_of_files):
    label_list = [ ('IPW + refl','r-','g-') , ('refl','b-','k-') ]
    
    plt.figure(figsize = (12,8))
    for l1,f1 in zip(label_list,list_of_files):
        make_plots_classification(l1,f1)
    plt.grid()
    plt.show()

def call_make_plots_regression(list_of_files):
    label_list = [ ('IPW + refl','r-','g-') , ('refl','b-','k-') ]
    
    plt.figure(figsize = (12,8))
    for l1,f1 in zip(label_list[:1],list_of_files[:1]):
        make_plots_regression(l1,f1)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # These files represent the training and validation loss measured with the 
    # split CNN IPW + refl with 8 filters each and the reflectivity alone with
    # 16 filters
    # performance_metrics_1CNN_0maxpool_2048neural_network_p20_special_0.pkl
    # performance_metrics_1CNN_0maxpool_2048neural_network_p20_refl_0.pkl
    # 1 CNN experiment for ipw + refl and refl
    list_of_files = {
                      'ipw_refl': ['performance_metrics_1CNN_0maxpool_2048neural_network_p20_special_{}.pkl'.format(x) for x in range(4)], 
                      'refl' : ['performance_metrics_1CNN_0maxpool_2048neural_network_p20_refl_{}.pkl'.format(x) for x in range(4)]
                      }
    # 2 CNN experiment for ipw + refl and refl
    # performance_metrics_2CNN_0maxpool_2048neural_network_p20_refl_3.pkl
    # performance_metrics_2CNN_0maxpool_2048neural_network_p20_special_3.pkl
#    list_of_files = {
#                      'ipw_refl': ['performance_metrics_2CNN_0maxpool_2048neural_network_p20_special_{}.pkl'.format(x) for x in range(4)], 
#                      'refl' : ['performance_metrics_2CNN_0maxpool_2048neural_network_p20_refl_{}.pkl'.format(x) for x in range(4)]
#                      }
    
    # comparison of different filter size for the two variables 
    # performance_metrics_1CNN_0maxpool_2048neural_network_p20_special_0.pkl
    # performance_metrics_1CNN_0maxpool_2048neural_network_p20_mod_special_0.pkl
    
#    list_of_files = {
#                      'ipw_refl': ['performance_metrics_1CNN_0maxpool_2048neural_network_p20_special_{}.pkl'.format(x) for x in range(4)], 
#                      'refl' : ['performance_metrics_1CNN_0maxpool_2048neural_network_p20_mod_special_{}.pkl'.format(x) for x in range(4)]
#                      }
    
    # Regression experiments 
    # performance_metrics_1CNN_0maxpool_2048neural_network_p20_mod_special_regression_adadelta_0.pkl
    # performance_metrics_1CNN_0maxpool_2048neural_network_p20_mod_special_refl8_regression_adadelta_0.pkl
#    list_of_files = {
#                        'ipw_refl' : ['performance_metrics_1CNN_0maxpool_2048neural_network_p20_mod_special_regression_adadelta_0.pkl'],
#                        'refl' : ['performance_metrics_1CNN_0maxpool_2048neural_network_p20_mod_special_refl8_regression_adadelta_0.pkl']
#                        }
    for ipw_refl,refl in zip(list_of_files['ipw_refl'],list_of_files['refl']):
        if 'regression' in ipw_refl and 'regression' in refl:
            call_make_plots_regression([ipw_refl,refl])
        else:
            call_make_plots_classification([ipw_refl,refl])
        
    
        
        
        

#    file_name = 'performance_metrics_1CNNneural_network0.pkl'