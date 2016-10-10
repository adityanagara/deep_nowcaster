# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 12:51:44 2016

@author: adityanagarajan
"""

import numpy as np
import cPickle as pkl
import matplotlib as mpl
from matplotlib import pyplot as plt
import os


def main():
    print os.getcwd()
    # 20140508_storm_mistake2CNN_2maxpool_neural_network_refl_run2_0_110.pkl
    # 20140508_storm_mistake1CNNneural_network_0_100.pkl
    # 20140508_storm_mistake1CNNneural_network_refl_0_100.pkl
    f1 = file('20140508_storm_mistake1CNNneural_network_refl_0_100.pkl','rb')
    arr = pkl.load(f1)
    f1.close()
    num_timesteps = 91
    domain_points = (range(17,83),range(17,83))
    
    PixelPoints = [(x,y) for x in domain_points[0]  
                    for y in domain_points[1]]
    
    PixelPoints = np.array(PixelPoints)
    
    Y_real = np.stack(map(lambda x: x[0], arr)).reshape(66,66,num_timesteps)
    
    Y_pred = np.stack(map(lambda x: x[1], arr)).reshape(66,66,num_timesteps,2)
    
    Y_pred = np.argmax(Y_pred,axis = 3)
    eval_array = np.zeros((66,66,num_timesteps))
    
    # Hits blue
    eval_array[np.logical_and(Y_real == 1,Y_real == Y_pred)] = 1.5
    # Misses green
    eval_array[np.logical_and(Y_real == 1,Y_real != Y_pred)] = 2.5
    # False Alarms
    eval_array[np.logical_and(Y_pred == 1,Y_real != Y_pred)] = 3.5
    
    cmap = plt.cm.jet
    
    cmaplist = [cmap(i) for i in [0,125,255]]
    
    bounds = np.linspace(1,4,4)
    
    cmap = cmap.from_list('Custom cmap', cmaplist, 3)
    
    norm = mpl.colors.BoundaryNorm(bounds, 3)
    
    gridX = np.arange(-150.0,151.0,300.0/(100-1))
    gridY = np.arange(-150.0,151.0,300.0/(100-1))
    # center of the domain
    x_ = 49.0
    y_ = 49.0

    x_range_start = gridX[x_] - 33.0*(300.0/99.0)
    y_range_start = gridY[y_] - 33.0*(300.0/99.0)

    x_range_end = gridX[x_] + 33.0*(300.0/99.0)
    y_range_end = gridY[y_] + 33.0*(300.0/99.0)
    gridX_ = np.arange(x_range_start,x_range_end,300./99.)
    gridY_ = np.arange(y_range_start,y_range_end,300./99.)
    time_index = ['{0}:{1}'.format(str(x).zfill(2),str(y).zfill(2)) 
                        for x in range(2,24) for y in [0,30]]

    
    time_index.extend(['{0}:{1}'.format(str(x).zfill(2),str(y).zfill(2)) 
                        for x in range(24) for y in [0,30]])
    
    print len(time_index)

    for i in range(num_timesteps):
        plt.figure()
        
#        pred_ = np.argmax(real_pred[:,:,i,1:],axis=2)
        
#        pred_ = np.ma.array(pred_, mask = pred_ == 0)
        
#        real_ = np.ma.array(real_pred[:,:,i,0], mask = real_pred[:,:,i,0] == 0.)
        real_ = np.ma.array(eval_array[:,:,i],mask = eval_array[:,:,i] == 0)
#        plt.subplot(121)
        plt.pcolor(gridX_,gridY_,real_.T,cmap = cmap,norm = norm)
        plt.xlim((-150.0,150.0))
        plt.ylim((-150.0,150.0))
        cbar = plt.colorbar(cmap=cmap, norm=norm,spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
        plt.grid()
        plt.title('Reflectivity prediction plot for ' + time_index[i + 1])
        plt.savefig('prediction_plots/reflcnn1/Plot_' + str(i) + '.png')
#        plt.subplot(122)
#        plt.pcolor(gridX_,gridY_,pred_.T,cmap='RdGy',vmin=0.0,vmax=1.0)
#        
#        plt.xlim((-150.0,150.0))
#        
#        plt.ylim((-150.0,150.0))
#        
#        plt.grid()
#        
#        plt.savefig('../output/prediction_movies_thesis/Plot_' + str(i) + '.png')
        
        
    
    
#    real_predictions = np.zeros((len(arr),91,3))
    
#    real_predictions[pt_ctr,:,0] = Y_test.reshape(91,)

if __name__ == '__main__':
    main()

'''
# setup the plot
fig, ax = plt.subplots(1,1, figsize=(6,6))

# define the data
x = np.random.rand(20)
y = np.random.rand(20)
tag = np.random.randint(0,20,20)
tag[10:12] = 0 # make sure there are some 0 values to showup as grey

# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.5,.5,.5,1.0)
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0,20,21)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# make the scatter
scat = ax.scatter(x,y,c=tag,s=np.random.randint(100,500,20),cmap=cmap, norm=norm)

# create a second axes for the colorbar
ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')

ax.set_title('Well defined discrete colors')
ax2.set_ylabel('Very custom cbar [-]', size=12)
'''



'''
def make_predictions(yr = 14):
    base_path = '../output/1_CNN_experiments/'
    # /Users/adityanagarajan/projects/nowcaster/output/1_CNN_experiments
    print os.path.exists(base_path + 'CPU_1_CNN_layer_max_pool_0_200.pkl')
    params = np.load(base_path + 'CPU_1_CNN_layer_max_pool_0_200.pkl')
    print len(params)
    
    input_var = T.tensor4('inputs')
    
    network,hidden_1 = build_DCNN_maxpool_softmax(input_var)
    
    lasagne.layers.set_all_param_values(network, params)
    
    prediction_ = lasagne.layers.get_output(network, deterministic=True)

    predict_function = theano.function([input_var], prediction_)
    
    data_builder = BuildDataSet.dataset(num_points = 1000)
    
    # This is the range of points in the central domain which can 
    # have a 33x33 around it.
    num_points = 4356
    
    domain_points = (range(17,83),range(17,83))
    
    PixelPoints = [(x,y) for x in domain_points[0]  
                    for y in domain_points[1]]
                        
    
    PixelPoints = np.array(PixelPoints)
    
    storm_dates_all = data_builder.load_storm_days(yr)
    
    doy_strings = data_builder.club_days(storm_dates_all)
    
    days_in_sorted = doy_strings.keys()
    
    days_in_sorted.sort()
    
    print doy_strings
    
    ipw_files,refl_files = data_builder.sort_IPW_refl_files_imgs(yr)
    
    temp_ipw_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings['129'],ipw_files)
#    
    temp_refl_files = filter(lambda x: re.findall('\d+',x)[1] in doy_strings['129'],refl_files)
    
    temp_ipw_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_ipw_files)
    temp_refl_files = map(lambda x: '../data/dataset/20' + str(yr) + os.sep + x,temp_refl_files)       
    # We define an array which contains the ground truth in column 1
    # and the prediction probabilities in column 2 and 3
    real_predictions = np.zeros((num_points,91,3))
    
    pt_ctr = 0
    
    for x_,y_ in zip(PixelPoints[:,0],PixelPoints[:,1]):
        
        print 'Predicting for point: (%d,%d)'%(x_,y_)
        
        temp_array = data_builder.build_features_and_truth_imgs(temp_ipw_files,temp_refl_files,x_,y_)
        
        ipw_refl_tensors = data_builder.arrange_frames_single(temp_array)
        
        X_test = ipw_refl_tensors[1]
        
        Y_test = ipw_refl_tensors[2]
        
        Y_pred = predict_function(X_test)
        
        real_predictions[pt_ctr,:,0] = Y_test.reshape(91,)
        
#        print Y_test.reshape(91,)
#        print np.argmax(Y_pred,axis=1)
        
        real_predictions[pt_ctr,:,1:] = Y_pred
        
        real_predictions.shape
        
        pt_ctr+=1
        
    return real_predictions
    
def plot_movies():
    real_pred = np.load('../output/CNN_real_pred_array.npy')
    real_pred =  real_pred.reshape(66,66,91,3)
    
    gridX = np.arange(-150.0,151.0,300.0/(100-1))
    gridY = np.arange(-150.0,151.0,300.0/(100-1))
    
    # center of the domain
    x_ = 49.0
    y_ = 49.0

    x_range_start = gridX[x_] - 33.0*(300.0/99.0)
    y_range_start = gridY[y_] - 33.0*(300.0/99.0)

    x_range_end = gridX[x_] + 33.0*(300.0/99.0)
    y_range_end = gridY[y_] + 33.0*(300.0/99.0)

    gridX_ = np.arange(x_range_start,x_range_end,300./99.)
    gridY_ = np.arange(y_range_start,y_range_end,300./99.)
    
#    i_start = x_ -33
#    i_end = x_ + 33
#    j_start = y_ - 33
#    j_end = y_ + 33
#    
#    time_index = ['{0}{1}'.format(str(x).zfill(2),str(y).zfill(2)) 
#                        for x in range(24) for y in [0,30]]
    num_time_steps = 91
    
    
    for i in range(num_time_steps):
        plt.figure()
        
        pred_ = np.argmax(real_pred[:,:,i,1:],axis=2)
        
        pred_ = np.ma.array(pred_, mask = pred_ == 0)
        
        real_ = np.ma.array(real_pred[:,:,i,0], mask = real_pred[:,:,i,0] == 0.)
        plt.subplot(121)
        plt.pcolor(gridX_,gridY_,real_.T,cmap='RdGy',vmin=0.0,vmax=1.0)
        
        plt.xlim((-150.0,150.0))
        
        plt.ylim((-150.0,150.0))
        
        plt.grid()
        
        plt.subplot(122)
        plt.pcolor(gridX_,gridY_,pred_.T,cmap='RdGy',vmin=0.0,vmax=1.0)
        
        plt.xlim((-150.0,150.0))
        
        plt.ylim((-150.0,150.0))
        
        plt.grid()
        
        plt.savefig('../output/prediction_movies_thesis/Plot_' + str(i) + '.png')
        

'''
