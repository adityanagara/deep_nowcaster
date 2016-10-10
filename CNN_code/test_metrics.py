# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:05:43 2016

@author: adityanagarajan
"""

import theano
import numpy as np
from theano import tensor as T

test_prediction_ = T.ivector('inputs')
    
target_var = T.ivector('targets')


hits = T.sum(T.and_(T.eq(test_prediction_,target_var),T.eq(target_var,1))) #+ 1e-16
num_cases = T.sum(T.eq(target_var,1)) + 1e-16
#misses = T.sum(T.and_(T.neq(test_prediction_,target_var),T.eq(target_var,1))) +  1e-16
false_alarms = T.sum(T.and_(T.neq(test_prediction_,target_var),T.eq(target_var,0)))
    
val_accuracy = T.mean(T.eq(test_prediction_, target_var),
                      dtype=theano.config.floatX)
POD = hits / num_cases
FAR = false_alarms / (false_alarms + hits)
CSI = hits / (num_cases + false_alarms)


val_fn_temp = theano.function([test_prediction_, target_var], [hits,num_cases,false_alarms],allow_input_downcast=True)
val_fn = theano.function([test_prediction_, target_var], [val_accuracy,POD,FAR,CSI],allow_input_downcast=True)

Y_pred = np.array([0,0,0,0,0,0,0,0]).astype('float32')

Y_true = np.array([0,0,0,0,0,0,0,0]).astype('float32')

temp_arr = val_fn_temp(Y_pred,Y_true)

print temp_arr

