# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:08:04 2016

@author: adityanagarajan
"""

import numpy as np
#import DFWnet
import BuildDataSet; reload(BuildDataSet)
import re
import os

from sklearn.cross_decomposition import CCA
#from sklearn.decomposition import PCA

from scipy.linalg import fractional_matrix_power
from matplotlib import pyplot as plt
import nowcast; reload(nowcast)
import sys
import cPickle as pkl
import PCA_CCA;reload(PCA_CCA)
import scipy.io

#plt.ioff()
plt.ion()
np.random.seed(451234)
storm_dates_train = np.array(([128,  14,   5,   8],
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

storm_dates_test = np.array(([128,  15,   5,   8],
                            [129,  15,   5,   9],
                            [137,  15,   5,  17],
                            [138,  15,   5,  18],
                            [139,  15,   5,  19],
                            [140,  15,   5,  20],
                            [147,  15,   5,  27],
                            [148,  15,   5,  28],
                            [164,  15,   6,  13],
                            [165,  15,   6,  14],
                            [166,  15,   6,  15],
                            [167,  15,   6,  16],
                            [168,  15,   6,  17]))


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
    

def load_data_slices(phase = 1,images_fields = 'fields',storm_dates = storm_dates_train):
    
    base_path = '../data/dataset/20'
    # We are going to visually look at each day to determine our storm 
    # dates. The storm dates are determined such that there is not 
    # too much noise and there is an evident storm. 
    data_builder = BuildDataSet.dataset(num_points = 500)
    
    doy_strings = data_builder.club_days(storm_dates)
    days_in_sorted = doy_strings.keys()
    days_in_sorted.sort()
    
    if images_fields == 'fields':
        ipw_files,refl_files = data_builder.sort_IPW_refl_files(storm_dates[0][1])
    else:
        ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(storm_dates[0][1])
    
    X = []
    Y = []
    for set_ in days_in_sorted:
        temp_ipw_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],ipw_files)
        temp_refl_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],refl_files)
        temp_ipw_files = map(lambda x: base_path + str(storm_dates[0][1]) + os.sep + x,temp_ipw_files)
        temp_refl_files = map(lambda x: base_path + str(storm_dates[0][1]) + os.sep + x,temp_refl_files)
        
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

def load_data(phase = 0,images_fields = 'fields'):
    
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
    time_step = []
    for set_ in days_in_sorted:
        temp_ipw_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],ipw_files)
        temp_refl_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings[set_],refl_files)
        temp_ipw_files = map(lambda x: '../data/dataset/20' + str(14) + os.sep + x,temp_ipw_files)
        temp_refl_files = map(lambda x: '../data/dataset/20' + str(14) + os.sep + x,temp_refl_files)
        
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
            X.append(ipw_array.reshape(-1,))
            Y.append(refl_array.reshape(-1,))
            time_step.append(re.findall('\d+',ipw.split('/')[-1]))
            
    
    X = np.array(X)
    Y = np.array(Y)
    print 'Input array has been loaded the size is as follows where rows'
    print 'represent the variables and the columns the examples'    
    print X.shape,Y.shape
    return time_step,X,Y

def compute_covariance(X,Y):
    m = X.shape[1]
    X_bar = X - np.mean(X,axis = 0)
    Y_bar = Y - np.mean(Y,axis=0)
    
    S_11 = (1./(m-1.)) * np.dot(X_bar.T,X_bar)
    S_22 = (1./(m-1.)) * np.dot(Y_bar.T,Y_bar)
    S_12 = (1./(m-1.)) * np.dot(X_bar.T,Y_bar)
    
    return S_11,S_12,S_22
    

    
def compute_CCA(X,Y):
    # define a variable for the number of examples
    m = X.shape[1]
    # Center the two arrays 
    X_bar = X - np.mean(X,axis = 1).reshape(-1,1)
    Y_bar = Y - np.mean(Y,axis=1).reshape(-1,1)
    
    # Obtain the covariance matricies
    S_11 = (1./(m-1.)) * np.dot(X_bar,X_bar.T)
    S_22 = (1./(m-1.)) * np.dot(Y_bar,Y_bar.T)
    S_12 = (1./(m-1.)) * np.dot(X_bar,Y_bar.T)
    print '-'*50
    print S_11
    print '-'*50
    print S_22
    print '-'*50
    print S_12.T
        
    print 'Inverse of S_11'
    U = fractional_matrix_power(S_11,-0.5)
    print 'Inverse of S_22'
    V = fractional_matrix_power(S_22,-0.5)
    
    T = np.dot(np.dot(U,S_12),V)
    
    print 'SVD...' 
    
    U_k,S_k,V_k = np.linalg.svd(T)
        
    A_1 = np.dot(U,U_k)
    B_1 = np.dot(V,V_k.T)
    
    
    return U_k,S_k,V_k,A_1,B_1
    
    
def plot_fields(T,X,Y):
    shape = (100,100)
    for i in range(X.shape[0]):
        fname = '../data/test_fields/Plot_'+ T[i][1] + '_'+ T[i][2] + '.png'
        plt.figure()
        plt.subplot(121)

        plt.imshow(X[i].reshape(shape),origin = 'lower', cmap = 'gray')
        plt.subplot(122)
        plt.imshow(Y[i].reshape(shape),origin = 'lower', cmap = 'gray')
        plt.savefig(fname)

def plot_field_slices(X_,Y_):
    X_ = X_.T
    Y_ = Y_.T
    print X_.shape,Y_.shape
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
        plt.show()
    

def plot_fields_overlapped():
    fields_plotter = nowcast.BuildNowcaster()
    
    data_builder = BuildDataSet.dataset(num_points = 500)
    files = data_builder.sort_IPW_refl_files(14)
    temp_array = np.zeros((len(files[0]),100,100))
    ctr = 0
    for ipw_file,refl_file in zip(files[0],files[1]):
        ipw_array = np.load('../data/dataset/2014/' + ipw_file )
        refl_array = np.load('../data/dataset/2014/' + refl_file)
        refl_array[refl_array < 0.0] = 0.0
        refl_array[refl_array < 30.0] = np.nan
#        fields_plotter.plot_ipw_refl_fields_overlap(ipw_array,refl_array)
        new_array = data_builder.convert_IPW_img(ipw_array)
        temp_array[ctr,...] = ipw_array
        ctr+=1
        print new_array.shape
    
    print 'Done!'
    print temp_array.shape
    

def CCA_analysis(X,Y,phase):
    S_11,S_12,S_22 = compute_covariance(X,Y)
    base_path = '../output/CCA_Weights/CCA_output_' + str(phase) + '.pkl'
    arr = np.load(base_path)
    A_ = arr[3]
    B_ = arr[4]
    
    lambdas = np.dot(A_.T,S_12,B_)
    
    print lambdas
    
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
    return X_new,Y_new,mdl1,mdl2


def permute_dataSet(X,Y):
    N = X.shape[0]
    p = X.shape[1]
    q = Y.shape[1]
    X_Y = np.concatenate((X,Y),axis = 1)
    X_Y = X_Y[np.random.permutation(N)]
    
    return X_Y[:,:p],X_Y[:,p:]
    
    
    
def train_CCA(phase,ipw_components,refl_components,probe_fields = False):
        
    X_slices,Y_slices = load_data_slices(int(phase),images_fields = 'images',storm_dates = storm_dates_train)
    X_slices,Y_slices = permute_dataSet(X_slices,Y_slices)
    #----------------------------------------------------#
#    Use the following data to test the implementation 
#    data from matlab verified using matlab
#    X = np.loadtxt('A_test_data.csv',delimiter=',')
#    Y = np.loadtxt('B_test_data.csv',delimiter=',')
#    X = X.T
#    Y = Y.T
#    cca_output = compute_CCA(X,Y)
#    print cca_output[3]
#    print cca_output[4]
    #----------------------------------------------------#
#    np.random.seed(12345)
#    X = np.random.multivariate_normal(np.random.randint(50,100,(10)).astype('float'),np.identity(10),200)
#    Y = np.random.multivariate_normal(np.random.randint(80,200,(6)).astype('float'),np.identity(6),200)
#    scipy.io.savemat('test_1.mat', dict(x=X, y=Y))
    #----------------------------------------------------#
    
    X,Y,pca_mdl1,pca_mdl2 = reduce_dimensions(X_slices,Y_slices,ipw_components,refl_components)
    scipy.io.savemat('test_wilks.mat', dict(x=X, y=Y))     
    mdl2 = PCA_CCA.CCA(n_components = 8)
    
    mdl2.fit(X,Y)
    print '-'*50
    print 'Covariance of training set'
    print mdl2.S
    print np.dot(np.dot(mdl2.A,mdl2.S_12),mdl2.B)
    
    return mdl2,pca_mdl1,pca_mdl2

def test_CCA(phase,cca_mdl,pca_mdl1,pca_mdl2):
    X_slices,Y_slices = load_data_slices(int(phase),images_fields = 'images',storm_dates = storm_dates_test)
    
    print X_slices.shape,Y_slices.shape
    
    X_new = pca_mdl1.transform(X_slices)
    Y_new = pca_mdl2.transform(Y_slices)
    
    S_11_test,S_12_test,S_22_test = compute_covariance(X_new,Y_new)
    
    print cca_mdl.A.shape, S_12_test.shape, cca_mdl.B.shape
    
    print np.dot(np.dot(cca_mdl.A,S_12_test),cca_mdl.B)
    

    
    
def plot_metrics(mdl_array):
    component_sum = []
    for mdl in mdl_array:
        component_sum.append(np.sum(mdl.S))
    print component_sum
    

def experiment_1():
    phase = 0
    ipw_components = 2
    refl_components = 2    
    phase_list = [0,1,2,3,4,5,6,7]
    ipw_components = [8]
    refl_components = [20,40,60,80,100,120,140,160,180,200]
    ipw_refl_component_models = []
    for ipw_c in ipw_components:
        for refl_c in refl_components:
            print 'Running ipw components = %d , refl components = %d'%(ipw_c,refl_c)
            mdl_array = []
            for phase in phase_list:
                print 'Running phase %d '%phase
                mdl = train_CCA(phase,ipw_c,refl_c)
                mdl_array.append(mdl)
            plot_metrics(mdl_array)
            ipw_refl_component_models.append(mdl_array)
    f1 = file('PCA_CCA_results_3.pkl','wb')
    pkl.dump(ipw_refl_component_models,f1)
    f1.close()
        
def experiment_2():
    phase = 0
    cca_mdl,pca_mdl1,pca_mdl2 = train_CCA(phase,8,80)
    test_CCA(phase,cca_mdl,pca_mdl1,pca_mdl2)
    

    
if __name__ == '__main__':
    experiment_2()
    
            
        
        

