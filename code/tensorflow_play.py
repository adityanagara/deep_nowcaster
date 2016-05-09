# -*- coding: utf-8 -*-
"""
Created on Mon May  2 19:36:03 2016

@author: adityanagarajan
"""
from __future__ import division
import tensorflow as tf
import numpy as np


y_true = tf.placeholder(tf.float32, [None, 2])
y_pred = tf.placeholder(tf.float32, [None, 2])

ground_truth = np.array(([1,0],[1,0],[0,1],[1,0],[1,0],[0,1],[1,0],[0,1]))

prediction = np.array(([1,0],[0,1],[0,0],[0,1],[1,0],[0,1],[1,0],[0,0]))
print ground_truth
print prediction

A = np.arange(1,10)

B = np.arange(11,20)

#for a,b in zip(A,B):
#    print a,b


truth = tf.argmax(y_true, 1)

pred = tf.argmax(y_pred, 1)

#correct_prediction = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

#hits = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1)),tf.equal(tf.argmax(y_true, 1),1)),'float'))

hits = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(truth, pred),tf.equal(truth,1)),'float'))
misses = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(truth, pred),tf.equal(truth,1)),'float'))
false_alarms = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(truth, pred),tf.equal(truth,0)),'float'))


#hits = tf.logical_and(tf.equal(truth, prediction),tf.equal(truth,1))



#misses = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(truth, prediction),tf.equal(truth,1)),'float'))
#misses = tf.reduce_sum(tf.logical_and(tf.not_equal(truth, prediction),tf.equal(truth,1)))
#false_alarms = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(truth, prediction),tf.equal(truth,0)),'float'))
#correct_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(truth, prediction),tf.equal(truth,0)),'float'))

#tf.reduce_all()
#ADD = tf.add(hits,misses)

POD = tf.div(hits,tf.add(hits,misses))
FAR = tf.div(false_alarms,tf.add(false_alarms,hits))
CSI = tf.div(hits,tf.add_n([hits,misses,false_alarms]))

sess = tf.Session()
temp_accuracy = sess.run([CSI],
                   feed_dict={
                       y_true: ground_truth,
                       y_pred: prediction
                       })

print temp_accuracy

sess.close()





'''
class NOWCAST_performance():
    Takes in input a tuple of shape 3 containing 3 column vectors
    Column 0: predicted values (0 or 1)
    Column 1: ground truth (0 or 1)
    Column 2: predicted score between [0,1]
    def __init__(self,obj_tuple):
        # Probability of detection
        self.POD = float(self.hits(obj_tuple)) / float((self.hits(obj_tuple) + self.misses(obj_tuple)))
        # Probability of false detection
        self.POFD = float(self.false_alarm(obj_tuple)) / float((self.false_alarm(obj_tuple) + self.correct_negatives(obj_tuple)))
        # False Alarm Rate
        self.FAR = float(self.false_alarm(obj_tuple)) / float((self.false_alarm(obj_tuple) + self.hits(obj_tuple)))
        # Critical Success Index 
        self.CSI = (float(self.hits(obj_tuple))/(float(self.hits(obj_tuple)) + float(self.misses(obj_tuple)) + float(self.false_alarm(obj_tuple))))
        # precision recall and f1 scores evaluated at 0.5 threshold
        self.p_score,self.r_score,self.f1 = self.precision_recall_scores(obj_tuple)
        # get precision recall and threshold list to plot curves
        self.precision_recall_curves(obj_tuple)
        # save the predictions 
#        self.obj_typle = obj_tuple
        
    def hits(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] == obj_tuple[1],obj_tuple[1] == 1.))

    def misses(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] != obj_tuple[1],obj_tuple[1] == 1.))

    def false_alarm(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] != obj_tuple[1],obj_tuple[1] == 0.))

    def correct_negatives(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] == obj_tuple[1],obj_tuple[1] == 0.))
    
    def precision_recall_scores(self,obj_tuple):
        p_score = precision_score(obj_tuple[1],obj_tuple[0])
        r_score = recall_score(obj_tuple[1],obj_tuple[0])
        f1 = f1_score(obj_tuple[1],obj_tuple[0])
        return p_score,r_score,f1
    
    def precision_recall_curves(self,obj_tuple):
        precision,recall,thresholds = precision_recall_curve(obj_tuple[1],obj_tuple[2])
        self.precision_list = precision
        self.recall_list = recall
        self.threshold_list = thresholds
        average_precision = average_precision_score(obj_tuple[1],obj_tuple[2])
        self.average_precision = average_precision
   
'''
