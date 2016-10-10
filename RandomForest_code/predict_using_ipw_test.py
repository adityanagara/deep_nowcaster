# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:49:20 2015

@author: aditya
"""

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import GradientBoostingClassifier

import os
import numpy as np


# Create the training set

def _training():
    Base_dir = 'TRAIN_TEST_DATA_SET' + os.sep
    files1=os.listdir(Base_dir)
    files1.sort(key = lambda s: s[6:10])
    train_files = files1[:22]
    test_files = files1[22:]
    print train_files
    print test_files
    print len(train_files)
    print len(test_files)
    fout_train = open('TRAIN_TEST_DATA_SET/train.csv','a+')
    fout_test = open('TRAIN_TEST_DATA_SET/test.csv','a+')
    for f in train_files:
        fin = open('TRAIN_TEST_DATA_SET' + os.sep + f,'r')
        fin.next()
        for line in fin:
            fout_train.write(line)
        fin.close() # not really needed
    fout_train.close()
    for g in test_files:
        print g
        fin = open('TRAIN_TEST_DATA_SET' + os.sep + g,'r')
        fin.next()
        for line in fin:
            fout_test.write(line)
        fin.close()
    fout_test.close()   
            
    
_training()

train_mat=np.loadtxt('TRAIN_TEST_DATA_SET/train.csv',dtype='S',delimiter=',')

test_mat = np.loadtxt('TRAIN_TEST_DATA_SET/test.csv',dtype='S',delimiter=',')
#
#
train_mat = train_mat[:,0:25].astype(float)

test_mat = test_mat[:,0:25].astype(float)

print train_mat.shape

print test_mat.shape


 
#
print train_mat[:,1:].shape
#
print train_mat[:,0].shape

print train_mat[:,0]

clf = DecisionTreeClassifier(max_depth = 3)
#clf = GradientBoostingClassifier()

clf.fit(train_mat[:,1:],train_mat[:,0])


y_hat = clf.predict(test_mat[:,1:])
print y_hat


np.savetxt('TRAIN_TEST_DATA_SET/out.csv',y_hat.T,fmt='%d')



#
#clf = svm.SVC()
#clf.fit(train_mat[:,1:],train_mat[:,0])
#
#y_hat = clf.predict(test_mat[:,1:])
#
#print y_hat.shape
#
#np.savetxt('TRAIN_TEST_DATA_SET/out.csv',y_hat.T,fmt='%d')
#print test_mat[:,0].shape
#
test_acc = 0.0
for i in range(test_mat[:,0].shape[0]):
    if test_mat[i,0] == y_hat[i]:
        test_acc+=1.0

print test_acc/88.0
        








    
    