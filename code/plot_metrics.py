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

'''

#file_names = ['model_metrics/GaussianNB_model_ipw.pkl','model_metrics/GaussianNB_model_refl.pkl','model_metrics/GaussianNB_model_ipw_refl_.pkl']

#file_names = ['model_metrics/NB_ipw_random_points_avg.pkl','model_metrics/NB_refl_random_points_avg.pkl','model_metrics/NB_ipw_refl_random_points_avg.pkl']
file_names = ['model_metrics/RF_ipw_random_points_avg.pkl','model_metrics/RF_refl_random_points_avg.pkl','model_metrics/RF_ipw_refl_random_points_avg.pkl']


plt.figure()

data_labels = ['IPW only','reflectivity only','IPW + reflectivity']

for f,l in zip(file_names,data_labels):
    print f
    obj_file = file(f)
    obj = cPickle.load(obj_file)
    obj_file.close()
    Y_test = obj[0][1][2]
    Y_score = obj[0][1][1]

    precision,recall,thresholds = precision_recall_curve(Y_test,Y_score[:,1])

    average_precision = average_precision_score(Y_test, Y_score[:, 1])

    plt.plot(recall, precision, label=l + ' average precision score = %.2f'%average_precision)
    
plt.xlabel('Recall',fontsize=16)
plt.ylabel('Precision',fontsize=16)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve for Random Forest',fontsize=16)

plt.legend(loc="upper Righr",fontsize = 'medium')

plt.grid()
plt.show()


'''
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
                                                        
'''

