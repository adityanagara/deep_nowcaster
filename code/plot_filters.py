# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:14:48 2016

@author: adityanagarajan
"""

import numpy as np
from matplotlib import pyplot as plt
from itertools import product

network_file = np.load('../output/1_CNN_experiments/CPU_1_CNN_layer_max_pool_0_200.pkl')

print len(network_file)

# Let us first look at the min and max values of the filters

def plot_conv_weights(layer, figsize=(6, 6)):
    """Plot the weights of a specific layer.
    Only really makes sense with convolutional layers.
    Parameters
    ----------
    layer : lasagne.layers.Layer
    """
    W = layer
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    for feature_map in range(shape[1]):
        figs, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
            if i >= shape[0]:
                break
            axes[r, c].imshow(W[i, feature_map], cmap='gray',
                              interpolation='none')
    return plt

conv_plot = plot_conv_weights(network_file[0],figsize = (8,8))

conv_plot.show()

#f = 3
#
#for i in range(7):
#    print 'For index %d the min: %f max %f'%(i,np.min(network_file[0][f,i,...]),np.max(network_file[0][f,i,...]))




