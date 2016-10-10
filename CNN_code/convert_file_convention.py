# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 23:11:18 2016

@author: adityanagarajan
"""

import os
import numpy as np
import re
import cPickle as pkl

# 
# 1CNNneural_network_0_100.pkl --> 1CNN_0maxpool_2048_neural_network_p20_V_E.pkl
# 1CNNneural_network_refl_3_100.pkl
# 1CNN_1maxpool_neural_network_ipw_refl_0_10.pkl

def main():
    base_path = '../output/thesis_results/'
    # convert this 1CNNneural_network_refl_0_10.pkl
    # to 1CNN_0maxpool_2048neural_network_p20_refl_0_110.pkl
    old_file_names = ['1CNNneural_network_refl_{0}_{1}.pkl'.format(x,y) for x in range(4) for y in range(10,110,10)]
    new_file_templet = '1CNN_0maxpool_2048neural_network_p20_refl_{0}_{1}'
    for f1 in old_file_names:
        if os.path.exists(base_path + f1):
            nums = re.findall('\d+',f1)
            new_file = new_file_templet.format(nums[1],nums[2])
            print base_path + f1 + '  -->  ' + base_path + new_file + '.pkl'
#            os.rename(base_path + f1,base_path + new_file + '.pkl')

def concat_loss_file():
    base_path_1 = '../output/output/' 
    base_path_2 = '../output/thesis_results/'
    
    file_list_1 = ['performance_metrics_1CNNneural_network_refl0.pkl','performance_metrics_1CNNneural_network_refl1.pkl','performance_metrics_1CNNneural_network_refl2.pkl','performance_metrics_1CNNneural_network_refl3.pkl']
    file_list_2 = ['performance_metrics_1CNN_0maxpool_2048neural_network_p20_refl_0.pkl','performance_metrics_1CNN_0maxpool_2048neural_network_p20_refl_1.pkl','performance_metrics_1CNN_0maxpool_2048neural_network_p20_refl_2.pkl','performance_metrics_1CNN_0maxpool_2048neural_network_p20_refl_3.pkl']
    
    for f1,f2 in zip(file_list_1,file_list_2):
        
        file_name1 = file(base_path_1 + f1,'rb')
        performance_list_1 = pkl.load(file_name1)
        file_name1.close()
        
        file_name2 = file(base_path_2 + f2,'rb')
        performance_list_2 = pkl.load(file_name2)
        file_name2.close()
        
#        print performance_list_1
        print '-'*50
#        print performance_list_2
        
        for i in range(1,101):
            performance_list_1[i + 100] = performance_list_2[i]
        
#        print '--'*50
        print performance_list_1
        out_file = file(base_path_2 + f2,'wb')
        pkl.dump(performance_list_1,out_file)
        out_file.close()
    
if __name__ == '__main__':
    concat_loss_file()
    
#    main()