# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:00:50 2015

@author: adityanagarajan
"""

'''
13 additional metrics suggested by MECIKALSKI et al. 2015
'''

import numpy as np
import cPickle

from matplotlib import pyplot as plt

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc

'''
kf_dict[kf_num] = [(Y_train[Y_train == 1.0].shape[0],Y_test[Y_test == 1.0].shape[0]),
(Y_hat,Y_hat_prob,Y_test),round(p_score,2),round(r_score,2),round(f1,2),mdl]

'''

names_list = ['ipw','refl','ipw+refl']
#file_names = ['model_metrics/RandomForest_model_ipw_.pkl',
#              'model_metrics/RandomForest_model_refl_.pkl',
#              'model_metrics/RandomForest_model_ipw_refl_.pkl',
#              'model_metrics/GaussianNB_model_ipw.pkl',
#              'model_metrics/GaussianNB_model_refl.pkl',
#              'model_metrics/GaussianNB_model_ipw_refl_.pkl']

file_names = ['model_metrics/RF_ipw_random_points_avg.pkl',
              'model_metrics/RF_refl_random_points_avg.pkl',
              'model_metrics/RF_ipw_refl_random_points_avg.pkl',
              'model_metrics/NB_ipw_random_points_avg.pkl',
              'model_metrics/NB_refl_random_points_avg.pkl',
              'model_metrics/NB_ipw_refl_random_points_avg.pkl']

def hits(obj_tuple):
    return np.sum(np.logical_and(obj_tuple[0] == obj_tuple[2],obj_tuple[2] == 1.,))

def misses(obj_tuple):
    return np.sum(np.logical_and(obj_tuple[0] != obj_tuple[2],obj_tuple[2] == 1.,))

def false_alarm(obj_tuple):
    return np.sum(np.logical_and(obj_tuple[0] != obj_tuple[2],obj_tuple[2] == 0.,))

def correct_negatives(obj_tuple):
    return np.sum(np.logical_and(obj_tuple[0] == obj_tuple[2],obj_tuple[2] == 0.,))

def plot_ROC(obj_tuple,file_name):
    fpr, tpr, thrashold = roc_curve(obj_tuple[2], obj_tuple[1][:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,label= file_name+ '(area = %0.2f)' % roc_auc)



plt.figure()
for i in range(3):
    print file_names[i]
    f = file_names[i]
    obj_file = file(f)
    obj = cPickle.load(obj_file)
    obj_file.close()
    obj_tuple = obj[0][1]
    
    POD = float(hits(obj_tuple)) / float((hits(obj_tuple) + misses(obj_tuple)))

    POFD = float(false_alarm(obj_tuple)) / float((false_alarm(obj_tuple) + correct_negatives(obj_tuple)))

    FAR = float(false_alarm(obj_tuple)) / float((false_alarm(obj_tuple) + hits(obj_tuple)))

    Acc = (float(hits(obj_tuple)) + float(correct_negatives(obj_tuple))) / float(hits(obj_tuple) + misses(obj_tuple) + false_alarm(obj_tuple) + correct_negatives(obj_tuple))
    # Critical Success Index
    CSI = (float(hits(obj_tuple))/(float(hits(obj_tuple)) + float(misses(obj_tuple)) + float(false_alarm(obj_tuple))))
    plot_ROC(obj_tuple,names_list[i])
    print 'Probability of detection %.2f'%POD
    print 'Probability of False Detection %.2f '%POFD
    print 'False alarm rate %.2f '%FAR
    print 'Accuracy %.2f'%Acc
    print 'Critical Success Index %.2f '%CSI

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend(loc = 4)

plt.title('Naive Bayes ROC Curves')

plt.grid()

plt.show()











