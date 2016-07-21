# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 12:51:09 2016

@author: adityanagarajan
"""

def  maxLength(a, k):
    sum_list = []
    for sub_arr in range(1,len(a)):
        temp_list = []
        range_list = range(0,len(a) + 1,sub_arr + 1)
        for i in range(0,len(a)):
#            print i
            if sub_arr <= abs(len(a) - i):
                temp_list.append(a[i : i + sub_arr])
                sum_list.append(sub_arr + 1)
    print max(sum_list)

maxLength([1,2,3,4,5],6)

