# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 10:10:01 2016

@author: adityanagarajan
"""

import numpy as np
import cPickle as pkl
from matplotlib import pyplot as plt



def main():
    f1 = file('results_reflectivity.pkl','rb')
    arr = pkl.load(f1)
    f1.close()
#    print arr['acc']
    plt.plot(arr['val_loss'])
#    plt.plot(arr['val_acc'])
    
#    plt.plot(arr['loss'])
#    plt.plot(arr['val_loss'])

if __name__ == '__main__':
    main()
