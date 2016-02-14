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
    
    def __init__(self,obj_tuple):
        # Probability of detection
        self.POD = float(self.hits(obj_tuple)) / float((self.hits(obj_tuple) + self.misses(obj_tuple)))
        # Probability of false detection
        self.POFD = float(self.false_alarm(obj_tuple)) / float((self.false_alarm(obj_tuple) + self.correct_negatives(obj_tuple)))
        # False Alarm Rate
        self.FAR = float(self.false_alarm(obj_tuple)) / float((self.false_alarm(obj_tuple) + self.hits(obj_tuple)))
        # Critical Success Index 
        self.CSI = (float(self.hits(obj_tuple))/(float(self.hits(obj_tuple)) + float(self.misses(obj_tuple)) + float(self.false_alarm(obj_tuple))))
        
    def hits(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] == obj_tuple[2],obj_tuple[2] == 1.,))

    def misses(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] != obj_tuple[2],obj_tuple[2] == 1.,))

    def false_alarm(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] != obj_tuple[2],obj_tuple[2] == 0.,))

    def correct_negatives(self,obj_tuple):
        return np.sum(np.logical_and(obj_tuple[0] == obj_tuple[2],obj_tuple[2] == 0.,))
        
        

        
        