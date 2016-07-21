# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 19:27:01 2016

@author: adityanagarajan
"""

import numpy as np
from sklearn.decomposition import PCA
import cPickle as pkl
from matplotlib import pyplot as plt
plt.ioff()

def plot_metrics():
    phase_list = [0,1,2,3,4,5,6,7]
    ipw_components = [2,3,4,5,6,7,8,9,10]
    refl_components = [2,3,4,5,6,7,8,9,10]

    f1 = file('PCA_CCA_results_3.pkl','rb')
    mdl_arrays = pkl.load(f1)
    f1.close()
    plt.figure()
    print len(mdl_arrays)
    for i,mdl in enumerate(mdl_arrays):
        temp_sum = []
        for m in mdl:
            temp_sum.append(np.sum(m.S))
#            print m.wilks_statistics()
            print m.p,m.q
        plt.plot(temp_sum,label='(%d,%d)'%(m.p,m.q))
    plt.legend()
    plt.grid()
    plt.title('Correlation coefficient vs time lag for various EOFs of (ipw,refl)')
    plt.ylabel('Total correlation coefficient')
    plt.xlabel('Time lag')
    plt.savefig('../output/CCA_results/fig3_.png')
            
            


def main():
    plot_metrics()
    
        
    


if __name__ == '__main__':
    main()