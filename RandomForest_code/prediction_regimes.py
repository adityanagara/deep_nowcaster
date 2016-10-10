# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 18:47:25 2015

@author: adityanagarajan
"""

import numpy as np
from matplotlib import pyplot as plt
import DFWnet
import os


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import KFold


DFW = DFWnet.CommonData()

TrainTestdir = 'data/TrainTest'

files = os.listdir(TrainTestdir)

IPWfiles = filter(lambda x: x[:7] == 'IPWdata' ,files)

Radarfiles = filter(lambda x: x[:9] == 'RadarRefl',files)

#Sort files DOY.hh
IPWfiles.sort(key = lambda x: float(x[7:10]) + float(x[x.index('_') + 1: x.index('.')]) * 0.01)


Radarfiles.sort(key = lambda x: float(x[9:12]) + float(x[x.index('_') + 1: x.index('.')])* 0.01)



# Pull out june data for now
IPWfiles = filter(lambda x: int(x[7:10]) < 152 or int(x[7:10]) > 181,IPWfiles)

Radarfiles = filter(lambda x: int(x[9:12]) < 152 or int(x[9:12]) > 181,Radarfiles)


# Pull out 205 = 07/24
IPWfiles = filter(lambda x: int(x[7:10]) != 205, IPWfiles)

Radarfiles = filter(lambda x: int(x[9:12]) != 205,Radarfiles)


temp_list = []
for f in IPWfiles:
    if f[7:10] not in temp_list:
        temp_list.append(f[7:10])
        
print temp_list


for t in temp_list:
    DFW.doytodate(14,int(t))
    print DFW.mon + '/' + DFW.day
    
thisList = ['IPWdata{0}_{1}.npy'.format(x,y) for x in temp_list for y in range(48)]

for L in thisList:
    if L not in IPWfiles:
        print L

# Set the thrashold for rain or not
Thrashold = 30.0




PixelX = range(20,50)
PixelY = range(50,80)

#r_PixelX = range(48,50)
#r_PixelY = range(50,52)
#
#remove_center = [(x,y) for x in r_PixelX for y in r_PixelY]

PixelPoints = [(x,y) for x in PixelX for y in PixelY]


PixelPoints = np.array(PixelPoints)


tempMtx = np.zeros((100,100))

RadarRegion1 = np.zeros((30,30))

gridX = np.arange(-150.0,151.0,300.0/(100-1))
gridY = np.arange(-150.0,151.0,300.0/(100-1))

pointX = []
pointY = []

for p in PixelPoints:
    print p[1]
    print p[0]
    
    pointX.append(gridX[p[0]])
    pointY.append(gridY[p[1]])

# These points have been verified

for tempIndex in range(900):
    print pointX[tempIndex],pointY[tempIndex]
    plt.plot(pointX[tempIndex],pointY[tempIndex],'r*')

plt.xlabel('Easting')
plt.ylabel('Northing')
plt.title('Experimental subdomain consisting of 30x30 grid')
    


plt.grid()
plt.xlim((-150.0,150.0))

plt.ylim((-150.0,150.0))


# Generate the output matrix for the 900 foints for each time step
#OutputMatrix = np.zeros((900,1104))
#
#Basefile = 'data/TrainTest/'
#
#for f in range(len(Radarfiles)):
#    
#    RadarMatrix = np.load(Basefile + Radarfiles[f])
#    
#    RadarMatrix[np.isnan(RadarMatrix)] = 0.0
#    
#    RadarMatrix[RadarMatrix < Thrashold] = 0.0
#    
#    RadarMatrix[RadarMatrix >= Thrashold] = 1.0
#    
#    # Determine ground truth for each of the 20 pixel points
#    for point in range(PixelPoints.shape[0]):
#        OutputMatrix[point,f] = RadarMatrix[PixelPoints[point,0],PixelPoints[point,1]]
#
#
#
#
#FeatureMatrix = np.zeros((1104,10000))
#
#for i in range(len(Radarfiles)):
#    
#    RadarMatrix = np.load(Basefile + Radarfiles[i])
#    
#    RadarMatrix[np.isnan(RadarMatrix)] = 0.0
#    
#    IPWMatrix = np.load(Basefile + IPWfiles[i])
#    
#    FeatureMatrix[i,:] = IPWMatrix.reshape(-1,)
#
#
#np.save(Basefile + 'FeatureMatrix.npy',FeatureMatrix)
#
#np.save(Basefile + 'OutputMatrix.npy',OutputMatrix)


#kf = KFold(1104, n_folds=6)
#
#
#
#
#
#ctr1 = 0
#
## Input the number of points we want to go over
#
#input_points = 100
#
#rocScores = np.zeros((6,input_points))
#
#for train_index, test_index in kf:
#    ctr = 0
#
#    for px in range(input_points):
#        
#        print 'Starting model for point : (%d, %d)'%(PixelPoints[px,0],PixelPoints[px,1])
#        
#        x = FeatureMatrix[:,:]
#    
#        y = OutputMatrix[px,:]
#        
#        x_train,x_test = x[train_index],x[test_index]
#        y_train,y_test = y[train_index],y[test_index]
#        
#        print x_train.shape,x_test.shape
#        print y_train.shape,y_test.shape
#        
#        
#        mdl = LogisticRegression(penalty = 'l2',C = 100.0)
#        
#        mdl.fit(x_train,y_train)
#        
#        y_cap = mdl.predict_proba(x_test)
#        
#        
#        fpr,tpr,thrsh = roc_curve(y_test,y_cap[:,1])
#        if np.where(np.isnan(fpr))[0].size > 0 or np.where(np.isnan(tpr))[0].size > 0:
#            print 'WARNING: No rain at all in this pixel (%d,%d) '%(PixelPoints[px,0],PixelPoints[px,1])
#            
#            roc_auc = np.nan
#        else:
#            roc_auc = auc(fpr,tpr)
#            print 'Train Test Split Number: %d %d) Area Under the curve for pixel point %d,%d = %f'%(ctr1,ctr,PixelPoints[px,0],PixelPoints[px,1],roc_auc)
#        rocScores[ctr1,ctr] = roc_auc
#        ctr+=1
#
#    ctr1+=1


#plt.figure()



#

# Generate the matrix for the 20 points for all time steps (1104)
#len(Radarfiles)

#for f in range(24,25):
#    
#    print Radarfiles[f]
#    
#    RadarMatrix = np.load(Basefile + Radarfiles[f])
#    
##    RadarMatrix[np.isnan(RadarMatrix)] = 0.0
#    
#    
#    RadarMatrix[RadarMatrix < 30.0] = np.nan
#    
#    RadarMatrix = np.ma.array(RadarMatrix, mask=np.isnan(RadarMatrix))
#    
#    plt.subplot(1,2,1)
#    
#    plt.pcolor(gridX,gridY,RadarMatrix,cmap='jet', vmin=10, vmax=60)
#    
#    plt.grid()
#    
#    plt.colorbar()
#    
#    plt.subplot(1,2,2)
    
##    plt.pcolor(gridX[PixelPoints[:,0]],gridY[PixelPoints[:,1]],RadarMatrix[],cmap='jet', vmin=10, vmax=60)
#    
#    
#    
#    plt.grid()
#    
#    plt.colorbar()
    
    
    
    
    
    
    
#    RadarMatrix[RadarMatrix < Thrashold] = 0.0
#    
#    RadarMatrix[RadarMatrix >= Thrashold] = 1.0
    
    # Determine ground truth for each of the 20 pixel points
#    for point in range(PixelPoints.shape[0]):
#        OutputMatrix[point,f] = RadarMatrix[PixelPoints[point,:][0],PixelPoints[point,:][1]]








