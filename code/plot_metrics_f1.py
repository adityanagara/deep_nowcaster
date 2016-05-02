# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:32:54 2015

@author: adityanagarajan
"""

import numpy as np
import cPickle

from matplotlib import pyplot as plt

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

'''
kf_dict[kf_num] = [(Y_train[Y_train == 1.0].shape[0],Y_test[Y_test == 1.0].shape[0]),
(Y_hat,Y_hat_prob,Y_test),round(p_score,2),round(r_score,2),round(f1,2),mdl]
#file_names = ['model_metrics/NB_ipw_random_points_avg.pkl','model_metrics/NB_refl_random_points_avg.pkl','model_metrics/NB_ipw_refl_random_points_avg.pkl']
file_names = ['model_metrics/RF_ipw_random_points_avg.pkl','model_metrics/RF_refl_random_points_avg.pkl','model_metrics/RF_ipw_refl_random_points_avg.pkl']


'''

file_names = ['model_metrics/RF_ipw_random_points_avg.pkl',
              'model_metrics/RF_refl_random_points_avg.pkl',
              'model_metrics/RF_ipw_refl_random_points_avg.pkl',
              'model_metrics/NB_ipw_random_points_avg.pkl',
              'model_metrics/NB_refl_random_points_avg.pkl',
              'model_metrics/NB_ipw_refl_random_points_avg.pkl']
              
#obj_file = file('model_metrics/RandomForest_model_ipw_.pkl')
#obj = cPickle.load(obj_file)
#obj_file.close()

plt.figure()

data_labels = ['IPW only','reflectivity only','IPW + reflectivity']
f1_scores = []
for f in file_names:
    
    obj_file = file(f)
    obj = cPickle.load(obj_file)
    obj_file.close()
    for oj in obj.keys():
        f1_scores.append(obj[oj][-2])

print f1_scores
x_axis = np.arange(1,8)

plt.plot()
plt.plot(x_axis,f1_scores[:7],'r--',label = 'Random forest with ipw')
plt.plot(x_axis,f1_scores[7:14],'r-.',label = 'Random forest with refl')
plt.plot(x_axis,f1_scores[14:21],'r-',label = 'Random forest with ipw and refl')

plt.plot(x_axis,f1_scores[21:28],'b--',label = 'Gaussian Naive Bayes with ipw')
plt.plot(x_axis,f1_scores[28:35],'b-.',label = 'Gaussian Naive Bayes with refl')
plt.plot(x_axis,f1_scores[35:42],'b-',label = 'Gaussian Naive Bayes with ipw and refl')

plt.ylim((0.0,1.0))

plt.title('f1 score across cross validation set for different trainers')

plt.ylabel('f1 score')
plt.xlabel('Cross Validation Set')
plt.grid()
plt.show()
plt.legend(loc="upper right",fontsize = 'x-small')


#plt.plot(recall, precision, label=l + ' average precision score = %.2f'%average_precision)
#    
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title('Precision-Recall curve for Gaussian Naive Bayes'.format(average_precision))
#
#
#
#plt.grid()
#plt.show()
#

'''
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
                                                        
'''

