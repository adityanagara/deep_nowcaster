# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:06:15 2016

@author: adityanagarajan
"""

import numpy as np

from scipy.linalg import svd,fractional_matrix_power

from scipy.stats import chi2


class PCA(object):
    def __init__(self,n_components):
        self.n_components = n_components
    
    def fit(self,X):
        self.N = X.shape[0]
        self.X_mean = np.mean(X,axis = 0)
        X_bar = X - self.X_mean
        U, S, V = svd(X_bar, full_matrices=False)
        self.components = V[:self.n_components]
        self.singular_values = np.power(S,2) / float(self.N)
        self.explained_variance = self.singular_values / np.sum(self.singular_values)
        
    
    def transform(self,Y):
        Y_bar = Y - self.X_mean
        return np.dot(Y_bar,self.components.T)
    
    def inverse_transform(self,Y):
        return np.dot(Y,self.components) + self.X_mean
    
class CCA(object):
    def __init__(self,n_components):
        self.k = n_components
        
    def _compute_covariance(self,X,Y):
        self.N = X.shape[0]
        self.p = X.shape[1]
        self.q = Y.shape[1]
        self.X_mean = np.mean(X,axis = 0)
        self.Y_mean = np.mean(Y,axis=0)
        X_bar = X - self.X_mean
        Y_bar = Y - self.Y_mean
    
        S_11 = (1./(self.N-1.)) * np.dot(X_bar.T,X_bar)
        S_22 = (1./(self.N-1.)) * np.dot(Y_bar.T,Y_bar)
        S_12 = (1./(self.N-1.)) * np.dot(X_bar.T,Y_bar)
        
        self.S_11 = S_11
        self.S_12 = S_12
        self.S_22 = S_22
        
    
    def fit(self,X,Y):
        self._compute_covariance(X,Y)
        S_11_ = fractional_matrix_power(self.S_11,-0.5)
        
        S_22_ = fractional_matrix_power(self.S_22,-0.5)
        
        T = np.dot(np.dot(S_11_,self.S_12),S_22_)
        
        U ,S ,V = np.linalg.svd(T)
        
        self.U = U[:,:self.k]
        self.S = S[:self.k]
        self.V = V[:self.k,:]
        
        self.A = np.dot(S_11_,U)
        self.B = np.dot(S_22_,V.T)
        self.A = self.A[:,:self.k]
        self.B = self.B[:,:self.k]
        
        self.coeff_ = np.dot(self.A,self.V)
        
        return self
    
    def transform(self,X,Y):
        X_bar = X - self.X_mean
        Y_bar = Y - self.Y_mean
        
        return (np.dot(X_bar,self.A[:,:self.k]),np.dot(Y_bar,self.B[:,:self.k]))
    
    def predict(self,X):
        '''Given the predictors X we will predict the output Y'''
        X_bar = X - self.X_mean
        return np.dot(X_bar,self.coeff_) + self.Y_mean
        
#        print np.dot(X_bar[0,:].reshape(1,-1),coeff_) + np.mean(Y,axis=0).reshape(1,-1)
    
    def wilks_statistics(self,S):
        logLambda = np.zeros(S.size)
        for s in range(S.size):
            logLambda[s] = np.sum(np.log(1 - np.power(S[s:],2)))
        k = np.arange(S.size)
        dp = self.p - k
        dq = self.q - k
        dstats = dp * dq
        delta = self.N - 1  - 0.5 * (self.p + self.q + 1)
        nue = np.cumsum(np.concatenate(([0],1./ np.power(S[:-1],2)))) - k
        L_k = - (delta + nue) * logLambda
        return chi2.cdf(L_k, dstats)


#def main():
#    mdl = PCA(5)
#    np.random.seed(12345)
#    X = np.random.multivariate_normal(np.random.randint(50,100,(10)).astype('float'),np.identity(10),200)
#    print X[0,:]
#    mdl.fit(X)
#    print X[0,:]
##    print mdl.explained_variance
#    Y = mdl.transform(X[:5,:])
#    print X[0,:]
##    print Y.shape
##    print Y
#    
#    X_ = mdl.inverse_transform(Y)
#    print X[0,:]
#    
#    print X_[0,:]
#
#def test_CCA():
#    np.random.seed(12345)
#    X = np.random.multivariate_normal(np.random.randint(50,100,(10)).astype('float'),np.identity(10),200)
#    Y = np.random.multivariate_normal(np.random.randint(80,200,(6)).astype('float'),np.identity(6),200)
#    
#    import scipy.io
#    scipy.io.savemat('test_predictions.mat', dict(x=X, y=Y))
#    
#    mdl3 = CCA(n_components = 1)    
#    mdl3.fit(X,Y)      
#    
#    X_,Y_ = mdl3.transform(X,Y)
#    
#    Y_predicted = mdl3.predict(X)
#    
#    print Y[5:10,:]
#    print '-'*50
#    print Y_predicted[5:10,:]
#    
#    from sklearn.cross_decomposition import CCA as CCA_sklearn
#    mdl2 = CCA_sklearn(n_components = 1)
#    mdl2.fit(X,Y)
#    
#    Y_predicted_2 =  mdl2.predict(X)
#    print '-'*50
#    print Y_predicted_2[5:10,:]
#    
#    
#    
#

def test_cca_implementation():
    X = np.random.multivariate_normal(np.random.randint(50,100,(10)).astype('float'),np.identity(10),200)
    Y = np.random.multivariate_normal(np.random.randint(80,200,(6)).astype('float'),np.identity(6),200)

    X_test = np.random.multivariate_normal(np.random.randint(50,100,(10)).astype('float'),np.identity(10),20)
    Y_test = np.random.multivariate_normal(np.random.randint(50,100,(6)).astype('float'),np.identity(6),20)
    
    mdl_test = CCA(n_components = 6)
    mdl_test.fit(X,Y)
    
    Y_pred = mdl_test.predict(X)
    
    print Y_pred
    print '-'*50
#    print Y_test

    from sklearn.cross_decomposition import CCA as CCA_sklearn
    
    mdl_actual = CCA_sklearn(n_components = 6)
    mdl_actual.fit(X,Y)
    
    print '-'*50
    Y_actual = mdl_actual.predict(X)
    print Y_actual
    
    
    
    
    
if __name__ == '__main__':
    test_cca_implementation()
#    main()
        
        
        
    
    
        
        
        
        
        
        
        
        

        

        
