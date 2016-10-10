# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:46:19 2016

@author: adityanagarajan
"""

import cPickle as pkl
from matplotlib import pyplot as plt
import numpy as np
import os

plt.ion()
#plt.ioff()

yr_mon = {14: [5,6,7,8],
              15: [5,6],
                16: [5,6,7]}

blocks = [str(yr) + '_' + str(mon) for yr in [14,15,16] for mon in yr_mon[yr]]

#samples.append({'theta':theta.copy(), 'alpha':alpha.copy(), 'beta':beta.copy()})
def plot_PR_curve(data):
    train_val = 0
    plt.figure()
    ctr = 0
    POD_list = []
    AUC_list = []
    FAR_list = []
    CSI_list = []
    for each_ in data:
#        print 'Validation year = %s Month = %s'%tuple(blocks[ctr].split('_'))
#        print '-'*50
#        print 'POD %.4f'%each_[train_val].POD
#        print 'FAR %.4f'%each_[train_val].FAR
#        print 'CSI %.4f'%each_[train_val].CSI
#        print '\n'
        POD_list.append(each_[train_val].POD)
        AUC_list.append(each_[train_val].average_precision)
        FAR_list.append(each_[train_val].FAR)
        CSI_list.append(each_[train_val].CSI)
        plt.plot(each_[train_val].recall_list,each_[train_val].precision_list,label = blocks[ctr])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        ctr+=1
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(np.linspace(0.,1.,20),np.linspace(1.,0.,20),'b--')
    plt.grid()
#    plt.show()
#    print POD_list
#    print AUC_list    
    print 'Average AUC = %.4f '%(sum(AUC_list) / len(AUC_list))
    print 'Average POD = %.4f '%(sum(POD_list) / len(POD_list))
    print 'Average FAR = %.4f '%(sum(FAR_list) / len(FAR_list))
    print 'Average CSI = %.4f '%(sum(CSI_list) / len(CSI_list))
#    print AUC_list

def print_metrics(data):
    ctr = 0
    performance_metrics = []
    for performance in data:
        print blocks[ctr]
        print '-'*50
        print 'Precision = %.2f'%performance[0].p_score
        print 'Recall = %.2f'%performance[0].r_score
        print 'F1 score = %.2f'%performance[0].f1
        print 'Area under the curve = %.2f'%performance[0].average_precision
        
        print 'POD = %.2f'%performance[0].POD
        print 'FAR = %.2f'%performance[0].FAR
        print 'CSI = %.2f'%performance[0].CSI
        performance_metrics.append({'POD': performance[0].POD,
                                    'FAR' : performance[0].FAR,
                                    'CSI' : performance[0].CSI,
                                    'AUC': performance[0].average_precision,
                                    'F1': performance[0].f1})
        ctr+=1
    
    return performance_metrics
    
def get_feature_importance(data):
    '''
    feature 0 - 5 ipw gradients
    feature 5 - 10 refl gradients
    feature 10 - 14 ipw averages
    feature 14 - 18 ipw standard deviations
    feature 18 - 21 refl averages
    feature 21 - 25 refl standard deviations
    ../data/dataset/2014/IPWdata14_210_41_img.npy
    '''
#    feature_names = ['grad_IPW'.format(x) for x in range(5)]
    feature_names = ['$\Delta IPW(t_0 - t_1)$','$\Delta IPW(t_1 - t_2)$','$\Delta IPW(t_2 - t_3)$','$\Delta IPW(t_0 - t_2)$','$\Delta IPW(t_1 - t_3)$']
    feature_names.extend(['$\Delta refl(t_0 - t_1)$','$\Delta refl(t_1 - t_2)$','$\Delta refl(t_2 - t_3)$','$\Delta refl(t_0 - t_2)$','$\Delta refl(t_1 - t_3)$'])
    feature_names.extend(['$IPW_A(t_{})$'.format(x) for x in range(4)])
    feature_names.extend(['$IPW_S(t_{})$'.format(x) for x in range(4)])
    feature_names.extend(['$refl_A(t_{})$'.format(x) for x in range(4)])
    feature_names.extend(['$refl_S(t_{})$'.format(x) for x in range(4)])
    cv_sets_feature_importance = []
    for performance in data:
        
        feature_importance = performance[0].mdl.feature_importances_
        
        feature_importance =  100.0 * (feature_importance / feature_importance.max())
        
        cv_sets_feature_importance.append(feature_importance)
    
    cv_sets_feature_importance = np.array(cv_sets_feature_importance)
    feature_importance = np.mean(cv_sets_feature_importance,axis = 0)
    sorted_index = np.argsort(feature_importance)
    
    plt.figure()
    pos = np.arange(feature_importance.shape[0]) + 0.5
    plt.barh(pos, feature_importance[sorted_index], align='center')
    plt.yticks(pos,np.array(feature_names)[sorted_index],fontsize=16)
    plt.title('Random Forest variable importance', fontsize = 14)
    plt.ylabel('variables',fontsize = 14)
    plt.xlabel('relative importance', fontsize = 14)
#    plt.grid()
    plt.show()

'''
['$\del IPW(t_0 - t_1)$','$\del IPW(t_1 - t_2)$','$\del IPW(t_2 - t_3)$','$\del IPW(t_0 - t_2)$','$\del IPW(t_1 - t_3)$']
[$IPW_{avg}(t_{})$.format(x) for x in range(4)]
feature_names = ['I^{t^*-1}$','$I^{t^*-1.5}$','$I^{t^*-2}$','$I^{t^*-2.5}$','$R^{t^*-1}$','$R^{t^*-1.5}$','$R^{t^*-2}$','$R^{t^*-2.5}$']

#feature_names = ['$I^{t^*-1}$','$I^{t^*-1.5}$','$I^{t^*-2}$','$I^{t^*-2.5}$','$R^{t^*-1}$','$R^{t^*-1.5}$','$R^{t^*-2}$','$R^{t^*-2.5}$']
#
#
sorted_index = np.argsort(feature_importance)
pos = np.arange(sorted_index.shape[0]) +0.5
plt.barh(pos, feature_importance[sorted_index], align='center')
plt.title('Random Forest variable importance', fontsize = 14)
plt.ylabel('variables',fontsize = 14)
plt.xlabel('relative importance', fontsize = 14)
plt.yticks(pos,np.array(feature_names)[sorted_index],fontsize=16)
plt.grid()
plt.show()
'''


def performance_averages():
    f2 = file('../output/RF_experiments/metrics_summary.pkl','rb')
    per = pkl.load(f2)
    f2.close()
    ipw_refl = per[0]
    refl = per[1]
    ipw_refl_pod = map(lambda x: x['F1'],ipw_refl)
    refl_pod = map(lambda x: x['F1'],refl)
    plt.figure()
    plt.plot(ipw_refl_pod,label='ipw_refl POD')
    plt.plot(refl_pod,label='refl POD')
    plt.legend()
    plt.show()

def get_result_files(ipw_templet,refl_templet,x = 500):
#    param_list = [str(x) +  '_' +  str(y) for x in range(400,900) for y in range(6,13)]
    # 400RF_60prediction_ipw_refl_experimentlog2_max_depth12
    # 400RF_60prediction_refl_experimentlog2_max_depth12
    # 700RF_60prediction_ipw_refl_experiment_max_depth_ver12.pkl
    # 700RF_60prediction_refl_experiment_max_depth12.pkl
    base_path = '../output/RF_experiments/'
    ipw_refl_file_list = []
    refl_file_list = []
    for y in [6,9,12]:
        
        if os.path.exists(base_path + ipw_templet.format(x,y)):
            
            ipw_refl_file_list.append(base_path + ipw_templet.format(x,y))
            
        if os.path.exists(base_path + refl_templet.format(x,y)):
            
            refl_file_list.append(base_path + refl_templet.format(x,y))
            
    return ipw_refl_file_list,refl_file_list
#    print ipw_refl_file_list[:7]
#    print refl_file_list[:7]

blocks = [str(yr) + '_' + str(mon) for yr in [14,15,16] for mon in yr_mon[yr]]


def plot_loops():
    ipw_file_list,refl_file_list = get_result_files()
    print len(ipw_file_list),len(refl_file_list)
    slices = [slice(0,7)]
    ipw_POD_list = []
    refl_POD_list = []
    for ipw,refl in zip(ipw_file_list,refl_file_list):
        print ipw,refl
        f1 = file(ipw,'rb')
        ipw_data = pkl.load(f1)
        f1.close()
        f2 = file(refl,'rb')
        refl_data = pkl.load(f2)
        f2.close()
        temp_ipw_list = []
        temp_refl_list = []
#        ipw_POD_list.append(ipw_data[0][1].POD)
#        refl_POD_list.append(refl_data[0][1].POD)
        print 'IPW + refl POD %.4f'%ipw_data[1][0].POD
        print 'reflectivity POD %.4f'%refl_data[1][0].POD
    
#    print max(ipw_POD_list)
#    print max(refl_POD_list)
    
        ipw_POD_list.append({'train_POD' : map(lambda x: x[1].POD, ipw_data),
                            'validation_POD': map(lambda x: x[0].POD, ipw_data),
                            'train_AUC': map(lambda x: x[1].average_precision, ipw_data),
                            'validation_AUC': map(lambda x: x[0].average_precision, ipw_data)})
                            
                            
        refl_POD_list.append({'train_POD' : map(lambda x: x[1].POD, refl_data),
                            'validation_POD': map(lambda x: x[0].POD, refl_data),
                            'train_AUC': map(lambda x: x[1].average_precision, refl_data),
                            'validation_AUC': map(lambda x: x[0].average_precision, refl_data)})
        
        average_ipw_val = []
        average_refl_val = []
        average_ipw_train = []
        acerage_refl_train = []
        
        for i,r in zip(ipw_POD_list,refl_POD_list):
            average_ipw_val.append(sum(i['validation_AUC']) / len(i['validation_AUC']))
            
            average_refl_val.append(sum(r['validation_AUC']) / len(r['validation_AUC']))
            
            average_ipw_train.append(sum(i['train_AUC']) / len(i['train_AUC']))
            
            acerage_refl_train.append(sum(r['train_AUC']) / len(r['train_AUC']))
            
        print average_ipw_val
        print average_refl_val
        
#        x_axis = range(6,13)
        plt.figure()
        plt.plot(average_ipw_val, label = 'ipw + refl',marker = 'o')
        plt.plot(average_refl_val, label = 'refl',marker = 'o')
#        plt.plot(x_axis,average_ipw_train, label = 'ipw + refl',marker = 'o')
#        plt.plot(x_axis,acerage_refl_train, label = 'refl',marker = 'o')
        plt.xlabel('Max tree depth')
        plt.ylabel('AUC of Precision-Recall curve')
        plt.legend()
        plt.grid()
        plt.show()

'''
Description of whats in each file:
contains 9 objects where an object for each CV set
an object further contains 2 more objects index 
0 for validation and index 1 for training. 
POD
FAR
CSI
'''

def get_CV_2014():
    base_path = '../output/RF_experiments/'
    file_list = ['400RF_60prediction_ipw_refl_experiment_max_depth_ver6.pkl',
                 '400RF_60prediction_refl_experiment_max_depth6.pkl',
                 '400RF_60prediction_ipw_refl_experimentlog2_max_depth12.pkl',
                 '400RF_60prediction_refl_experimentlog2_max_depth12.pkl']
    
    mon_dict = {0 : 'May', 1: 'June',2 : 'July', 3 : 'August'}
    
    for f1 in file_list:
        print '-'*50
        print f1
        load_f1 = file(base_path + f1,'rb')
        arr = pkl.load(load_f1)
        load_f1.close()
        average_POD = []
        average_FAR = []
        average_CSI = []
        average_AUC = []
        for i in range(4):
            print '-'*50
            print 'POD for month %s = %.4f'%(mon_dict[i],arr[i][0].POD)
            print 'FAR for month %s = %.4f'%(mon_dict[i],arr[i][0].FAR)
            print 'CSI for month %s = %.4f'%(mon_dict[i],arr[i][0].CSI)
            print 'AUC for month %s = %.4f'%(mon_dict[i],arr[i][0].average_precision)
            
            average_POD.append(arr[i][0].POD)
            average_FAR.append(arr[i][0].FAR)
            average_CSI.append(arr[i][0].CSI)
            average_AUC.append(arr[i][0].average_precision)
        
        print 'Average POD %.4f '%(sum(average_POD) / len(average_POD))
        print 'Average FAR %.4f '%(sum(average_FAR) / len(average_FAR))
        print 'Average CSI %.4f '%(sum(average_CSI) / len(average_CSI))
        print 'Average AUC %.4f '%(sum(average_AUC) / len(average_CSI))
        
            
            
            
            
            
            
#    ipw_refl = '{0}RF_60prediction_ipw_refl_experiment_max_depth_ver{1}.pkl'
#    refl = '{0}RF_60prediction_refl_experiment_max_depth{1}.pkl'
        
def plot_results():
    train_test = 0
    x_label = [6,9,12]
    # 400RF_60prediction_ipw_refl_experimentlog2_max_depth6.pkl
    # 400RF_60prediction_refl_experimentlog2_max_depth9.pkl
    ipw_refl_templet = '{0}RF_60prediction_ipw_refl_experimentlog2_max_depth{1}.pkl'
    refl_templet = '{0}RF_60prediction_refl_experimentlog2_max_depth{1}.pkl'
    
    for mark,x in zip(['b','g','r'],[400,500,600]):
        ipw_CSI_list_train = []
        refl_CSI_list_train = []
        ipw_CSI_list_val = []
        refl_CSI_list_val = []
        ipw_file_list,refl_file_list = get_result_files(ipw_refl_templet,refl_templet,x)
        for ipw,refl in zip(ipw_file_list,refl_file_list):
            print ipw,refl
            f1 = file(ipw,'rb')
            ipw_data = pkl.load(f1)
            f1.close()
            f2 = file(refl,'rb')
            refl_data = pkl.load(f2)
            f2.close()
            
            ipw_CSI_list_train.append(map(lambda k: ipw_data[k][1].CSI,range(4)))
            refl_CSI_list_train.append(map(lambda k: refl_data[k][1].CSI,range(4)))
            
            ipw_CSI_list_val.append(map(lambda k: ipw_data[k][0].CSI,range(4)))
            refl_CSI_list_val.append(map(lambda k: refl_data[k][0].CSI,range(4)))
            
            
#        plt.figure()
#        plt.plot(x_label,map(lambda t : sum(t) / len(t),ipw_CSI_list_train),mark + '-', label = 'ipw + refl ' + str(x) + ' trees')
#        plt.plot(x_label,map(lambda t : sum(t) / len(t),refl_CSI_list_train),mark + '--', label = 'refl ' + str(x) + ' trees')
        
        plt.plot(x_label,map(lambda t : sum(t) / len(t),ipw_CSI_list_val),mark + '-', label = 'refl ' + str(x) + ' trees')
        plt.plot(x_label,map(lambda t : sum(t) / len(t),refl_CSI_list_val),mark + '--', label = 'ipw + refl ' + str(x) + ' trees')
        
        
#    plt.title('CSI metric across max tree depth and number of trees in the forest')
    
    plt.xlabel('Max tree depth')
    plt.ylabel('CSI score')
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.show()
#    plt.savefig('RF_result.png')
        
        
        
            
#                print 'IPW + refl metrics'
#                print '-'*50
#                print ipw_data[0][0].POD
#                print ipw_data[0][0].FAR
#                print ipw_data[0][0].CSI
#                print 'refl metrics'
#                print '-'*50
#                print refl_data[0][0].POD
#                print refl_data[0][0].FAR
#                print refl_data[0][0].CSI


# 400RF_60prediction_ipw_refl_experiment_modellog2_max_depth12.pkl
# 400RF_60prediction_ipw_refl_experiment_modelNone_max_depth6.pkl
# 400RF_60prediction_refl_experiment_modellog2_max_depth12.pkl
# 400RF_60prediction_refl_experiment_modelNone_max_depth6.pkl

def plot_feature_importance():
    base_path = '../output/RF_experiments/'    
    file_list = ['400RF_60prediction_ipw_refl_experiment_modellog2_max_depth12.pkl']    
    f1 = file(base_path + file_list[0],'rb')
    data = pkl.load(f1)
    f1.close()
    get_feature_importance(data)
    
if __name__ == '__main__':
    plot_feature_importance()
    
#    plot_results()
#    get_CV_2014()
    



'''
    POD_list = []
    AUC_list = []
    FAR_list = []
    CSI_list = []
    for each_ in data:
#        print 'Validation year = %s Month = %s'%tuple(blocks[ctr].split('_'))
#        print '-'*50
#        print 'POD %.4f'%each_[train_val].POD
#        print 'FAR %.4f'%each_[train_val].FAR
#        print 'CSI %.4f'%each_[train_val].CSI
#        print '\n'
        POD_list.append(each_[train_val].POD)
        AUC_list.append(each_[train_val].average_precision)
        FAR_list.append(each_[train_val].FAR)
        CSI_list.append(each_[train_val].CSI)
        plt.plot(each_[train_val].recall_list,each_[train_val].precision_list,label = blocks[ctr])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        ctr+=1
'''

#    file_path = ['../output/RF_experiments/400RF_60prediction_ipw_refl_experiment_max_depth6.pkl','../output/RF_experiments/400RF_60prediction_refl_experiment_max_depth6.pkl']
#    performance_metrics = []
#    for f in file_path:
#        print '*'*50
#        f1 = file(f,'rb')
#        data = pkl.load(f1)
#        f1.close()
#        print len(data)
#        plot_PR_curve(data)
#        feature_importance(data)
#        metrics = print_metrics(data)
#        performance_metrics.append(metrics)
    
#    f2 = file('../output/RF_experiments/metrics_summary.pkl','wb')
#    pkl.dump(performance_metrics,f2)
#    f2.close()
    
#    performance_averages()
        
#        feature_importance(data)

'''
file_names = ['model_metrics/RandomForest_model_ipw_.pkl',
              'model_metrics/RandomForest_model_refl_.pkl',
              'model_metrics/RandomForest_model_ipw_refl_.pkl',
              'model_metrics/GaussianNB_model_ipw.pkl',
              'model_metrics/GaussianNB_model_refl.pkl',
              'model_metrics/GaussianNB_model_ipw_refl_.pkl']

new_file = 'model_metrics/RF_ipw_refl_random_points_avg.pkl'

f1 = file(new_file)
model_metric = cPickle.load(f1)
f1.close()

CV_feature_importance = []
for i in model_metric.keys():
    feature_importance = model_metric[i][-1].feature_importances_


    feature_importance =  100.0 * (feature_importance / feature_importance.max())
    CV_feature_importance.append(feature_importance)

feature_importance = np.mean(np.array(CV_feature_importance),axis = 0)

print feature_importance

feature_names = ['$I^{t^*-1}$','$I^{t^*-1.5}$','$I^{t^*-2}$','$I^{t^*-2.5}$','$R^{t^*-1}$','$R^{t^*-1.5}$','$R^{t^*-2}$','$R^{t^*-2.5}$']

#feature_names = ['$I^{t^*-1}$','$I^{t^*-1.5}$','$I^{t^*-2}$','$I^{t^*-2.5}$','$R^{t^*-1}$','$R^{t^*-1.5}$','$R^{t^*-2}$','$R^{t^*-2.5}$']
#
#
sorted_index = np.argsort(feature_importance)
pos = np.arange(sorted_index.shape[0]) +0.5
plt.barh(pos, feature_importance[sorted_index], align='center')
plt.title('Random Forest variable importance', fontsize = 14)
plt.ylabel('variables',fontsize = 14)
plt.xlabel('relative importance', fontsize = 14)
plt.yticks(pos,np.array(feature_names)[sorted_index],fontsize=16)
plt.grid()
plt.show()
'''
        
        
