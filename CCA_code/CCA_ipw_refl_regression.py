# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 18:52:20 2016

@author: adityanagarajan
"""

import numpy as np
import PCA_CCA
import os
import BuildDataSet
import re
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import CCA as CCA_sklearn
from sklearn.decomposition import PCA as PCA_sklearn


np.random.seed(12345)

def gen_slices(ipw_array,refl_array):
    X_11 = ipw_array[:50,:50]
    Y_11 = refl_array[:50,:50]
    X_12 = ipw_array[:50,50:]
    Y_12 = refl_array[:50,50:]
    X_21 = ipw_array[50:,:50]
    Y_21 = refl_array[50:,:50]
    X_22 = ipw_array[50:,50:]
    Y_22 = refl_array[50:,50:]
    
    return [X_11,X_12,X_21,X_22],[Y_11,Y_12,Y_21,Y_22]


def load_data_slices(phase = 1,images_fields = 'fields'):
    
    base_path = '../data/dataset/20'
    # We are going to visually look at each day to determine our storm 
    # dates. The storm dates are determined such that there is not 
    # too much noise and there is an evident storm. 
    data_builder = BuildDataSet.dataset(num_points = 500)
    
    storm_dates = np.array(([128,  14,   5,   8],
                            [129,  14,   5,   9],
                            [132,  14,   5,  12],
                            [133,  14,   5,  13],
                            [144,  14,   5,  24],
                            [145,  14,   5,  25],
                            [146,  14,   5,  26],
                            [164,  14,   6,  13],
                            [195,  14,   7,  14],
                            [196,  14,   7,  15],
                            [197,  14,   7,  16],
                            [198,  14,   7,  17],
                            [211,  14,   7,  30],
                            [212,  14,   7,  31],
                            [223,  14,   8,  11],
                            [241,  14,   8,  29]))

    
    doy_strings = data_builder.club_days(storm_dates)
    days_in_sorted = doy_strings.keys()
    days_in_sorted.sort()
    
    if images_fields == 'fields':
        ipw_files,refl_files = data_builder.sort_IPW_refl_files(14)
    else:
        ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(14)
    X = []
    Y = []
    for set_ in days_in_sorted:
        temp_ipw_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],ipw_files)
        temp_refl_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],refl_files)
        temp_ipw_files = map(lambda x: base_path + str(14) + os.sep + x,temp_ipw_files)
        temp_refl_files = map(lambda x: base_path + str(14) + os.sep + x,temp_refl_files)
        
        if phase > 0:
            # Drop n files from the beggining of the ipw file list
            # Drop n files fro the end of the refl list
            temp_refl_files = temp_refl_files[phase:]
            temp_ipw_files = temp_ipw_files[:-phase]
            
        for ipw,refl in zip(temp_ipw_files,temp_refl_files):
            ipw_array = np.load(ipw).astype('float')
            refl_array = np.load(refl).astype('float')
            
            refl_array[np.isnan(refl_array)] = 0.0
            refl_array[refl_array < 0.0] = 0.0
            temp_tup = gen_slices(ipw_array,refl_array)
            for X_,Y_ in zip(temp_tup[0],temp_tup[1]):
                
                X.append(X_.reshape(-1,))
                Y.append(Y_.reshape(-1,))
            
    
    X = np.array(X)
    Y = np.array(Y)
    print 'Input array has been loaded the size is as follows (N_examples,F_features)'
    print X.shape,Y.shape
    return X,Y

def permute_dataSet(X,Y):
    N = X.shape[0]
    p = X.shape[1]
    q = Y.shape[1]
    X_Y = np.concatenate((X,Y),axis = 1)
    X_Y = X_Y[np.random.permutation(N)]
    
    return X_Y[:,:p],X_Y[:,p:]

def reduce_dimensions(X_slices,Y_slices,p_components,q_components,plot=False):
    mdl1 = PCA_CCA.PCA(n_components = p_components)
    mdl1.fit(X_slices)
    ipw_explained = []
    for i in range(1,p_components):
#        print 'Explained variance for component %d IPW %.6f'%(i,np.sum(mdl1.explained_variance[:i]))
        ipw_explained.append(np.sum(mdl1.explained_variance[:i]))
    X_new = mdl1.transform(X_slices)        
    mdl2 = PCA_CCA.PCA(n_components = q_components)
    mdl2.fit(Y_slices)
    refl_explained = []
    for j in range(1,q_components):
#        print 'Explained variance for component %d Reflectivity %.6f'%(j,np.sum(mdl2.explained_variance[:j]))
        refl_explained.append(np.sum(mdl2.explained_variance[:j]))
    if plot:
        plt.figure()
        plt.subplot(121)
        plt.plot(ipw_explained)
#    plt.ylim(0.0,1.1)
        plt.subplot(122)
        plt.plot(refl_explained)
#    plt.ylim(0.0,1.1)
        plt.show()
    Y_new = mdl2.transform(Y_slices)
    return X_new,Y_new

def plot_field_slices(Y_):
    print Y_.shape
    for i in range(0,192,4):
        plt.figure()
        plt.subplot(223)
        plt.imshow(Y_[i].reshape(50,50),origin = 'lower', cmap = 'gray')
        plt.subplot(224)
        plt.imshow(Y_[i+1].reshape(50,50),origin = 'lower', cmap = 'gray')
        plt.subplot(221)
        plt.imshow(Y_[i+2].reshape(50,50),origin = 'lower', cmap = 'gray')
        plt.subplot(222)
        plt.imshow(Y_[i+3].reshape(50,50),origin = 'lower', cmap = 'gray')
        plt.savefig('../output/CCA_results/out_fields/Plot_' + str(i) + '.png')
#        plt.show()



def main():
    phase = 0
    X_slices,Y_slices = load_data_slices(int(phase),images_fields = 'images')
    X_slices,Y_slices = permute_dataSet(X_slices,Y_slices)
    ipw_components = 100
    refl_components = 200
    
    pca_mdl1 = PCA_sklearn(n_components = 0.99)
    pca_mdl2 = PCA_sklearn(n_components = 0.8)
    
    pca_mdl1.fit(X_slices)
    pca_mdl2.fit(Y_slices)
    
    X_new = pca_mdl1.transform(X_slices)
    Y_new = pca_mdl2.transform(Y_slices)
    
    print X_new.shape,Y_new.shape
    
    print pca_mdl1.explained_variance_ratio_,pca_mdl2.explained_variance_ratio_
    
    cca_mdl = CCA_sklearn(n_components = 4)
    
    cca_mdl.fit(X_new,Y_new)
    
    print cca_mdl.x_rotations_.shape
    
    print cca_mdl.y_rotations_.shape
    
#    S_12 = np.cov(X_new,Y_new,rowvar = 0)
    
    mdl_temp = PCA_CCA.CCA(n_components = 4)
    
    mdl_temp.fit(X_new,Y_new)
    
#    S_12
    
    print '-'*50
    print np.dot(np.dot(cca_mdl.x_rotations_.T,mdl_temp.S_12),cca_mdl.y_rotations_)
    
#    X,Y = reduce_dimensions(X_slices,Y_slices,ipw_components,refl_components)
#    print 'After PCA'
#    print X.shape,Y.shape
#    
#    CCA_mdl = PCA_CCA.CCA(n_components = 10)
##    CCA_mdl = CCA_sklearn(n_components = 10)
#    CCA_mdl.fit(X,Y)
#    # Load data slices again for testing
#    X_slices,Y_slices = load_data_slices(int(phase),images_fields = 'images')
#    
##    X_slices = X_slices[:384,:]
##    Y_slices = Y_slices[:384,:]
#    
#    PCA_mdl1 = PCA_CCA.PCA(ipw_components)
#    PCA_mdl1.fit(X_slices)
#    X_new = PCA_mdl1.transform(X_slices)
#    
#    PCA_mdl2 = PCA_CCA.PCA(refl_components)
#    PCA_mdl2.fit(Y_slices)
#    Y_new = PCA_mdl2.transform(Y_slices)
#    
#    Y_reduced = CCA_mdl.predict(X_new)
#    
#    Y_fields = PCA_mdl2.inverse_transform(Y_reduced)
#    
#    print Y_fields.shape
#    
#    plot_field_slices(Y_fields)
    
    
    
    
def test():
    X = np.random.multivariate_normal(np.random.randint(50,100,(30)).astype('float'),np.identity(30),500)
    Y = np.random.multivariate_normal(np.random.randint(80,200,(12)).astype('float'),np.identity(12),500)
    
    mdl1 = CCA_sklearn(n_components = 10)
    mdl2 = PCA_CCA.CCA(n_components = 10)
    
    mdl1.fit(X,Y)
    mdl2.fit(X,Y)
    
    print X.shape,Y.shape
    
#    PCA_mdl1 = PCA_CCA.PCA(20)
#    PCA_mdl1.fit(X)
#    
#    PCA_mdl2 = PCA_CCA.PCA(10)
#    PCA_mdl2.fit(Y)
#    X_ = PCA_mdl1.transform(X)
#    Y_ = PCA_mdl2.transform(Y)
#    
#    CCA_mdl = PCA_CCA.CCA(n_components = 2)
##    CCA_mdl = CCA_sklearn(n_components = 2)
#    CCA_mdl.fit(X_,Y_)
#    
#    X_test = np.random.multivariate_normal(np.random.randint(50,100,(30)).astype('float'),np.identity(30),10)
#    Y_test = np.random.multivariate_normal(np.random.randint(80,200,(12)).astype('float'),np.identity(12),10)
#    
#    print Y_test
#    X_new = PCA_mdl1.transform(X_test)
#    Y_new = PCA_mdl2.transform(Y_test)
#    
#    Y_hat = CCA_mdl.predict(X_new)
#    
#    Y_orig = PCA_mdl2.inverse_transform(Y_hat)
#    print '-'*50
#    print Y_orig
    
if __name__ == '__main__':
#    test()
    main()