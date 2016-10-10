# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:55:55 2016

@author: adityanagarajan
"""

import numpy as np

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class NOWCAST_performance():
    '''Takes in input a tuple of shape 3 containing 3 column vectors
    Column 0: predicted values (0 or 1)
    Column 1: ground truth (0 or 1)
    Column 2: predicted score between [0,1]'''
    def __init__(self,obj_tuple,model = None):
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
        self.mdl = model
#        self.obj_typle = obj_tuple
        
    def hits(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] == obj_tuple[1],obj_tuple[1].astype('float') == 1.))

    def misses(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] != obj_tuple[1],obj_tuple[1].astype('float') == 1.))

    def false_alarm(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] != obj_tuple[1],obj_tuple[1].astype('float') == 0.))

    def correct_negatives(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] == obj_tuple[1],obj_tuple[1].astype('float') == 0.))
    
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
        
        
        
        
        

        
        
