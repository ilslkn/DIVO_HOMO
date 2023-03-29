#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:00:32 2023

@author: laikongning
"""

# author:lyz
# Date:2022.04.22
import os
import numpy as np
import pandas

folder = 'test'
num = 10
root_path = '/data/srtp/Multi_view_tracking/labels_with_ids/' +folder
new_root = '/data/srtp/Multi_view_tracking/DIVO_{}/'.format(str(num))+folder
if not os.path.exists(new_root):
    os.makedirs(new_root)        

def read_tablemethod(filename):
    data = pandas.read_table(filename, header=None, delim_whitespace=True)
    return data
total_files = 0

#lst = ['Circle', 'Shop', 'Moving', 'Park', 'Ground', 'Gate1', 'Floor', 'Side', 'Square', 'Gate2']
lst = ['Ground', 'Floor']
for i in lst:
    tot_src = []
    tot_dst = []
    src_path = os.path.join(root_path, i)
    src_path = src_path + '_View1' + '/img1'
    dst_path = os.path.join(root_path, i)
    dst_path = dst_path + '_View2' + '/img1'
    src_lst = os.listdir(src_path)
    dst_lst = os.listdir(dst_path)
    len_lst = len(src_lst)
    for j in range(len_lst):
        path1 = os.path.join(src_path, src_lst[j])
        path2 = os.path.join(dst_path, dst_lst[j])
        src_data = read_tablemethod(path1)
        dst_data = read_tablemethod(path2)
        src = np.zeros(src_data.shape)
        dst = np.zeros(dst_data.shape)
        for x in range(src_data.shape[0]):
            for y in range(src_data.shape[1]):
                src[x, y] = src_data[y][x]
        for x in range(dst_data.shape[0]):
            for y in range(dst_data.shape[1]):
                dst[x, y] = dst_data[y][x]
        src_new = np.zeros((src_data.shape[0],2))
        dst_new = np.zeros((dst_data.shape[0],2))
        
        k = 0
        m = 0
        l = 0
        while k < src.shape[0] and m < dst.shape[0]:

            if src[k][1] == dst[m][1]:
                src_new[l][0] = src[k][2] + 0.5 * src[k][4]
                dst_new[l][0] = dst[m][2] + 0.5 * dst[m][4]
                src_new[l][1] = src[k][3] + 0.5 * src[k][5]
                dst_new[l][1] = dst[m][3] + 0.5 * dst[m][5]
                m=m+1
                l=l+1
                k=k+1
            elif src[k][1] > dst[m][1]:
                m=m+1
            else:
                k=k+1
        if l >= num:
            pair_path = os.path.join(new_root, src_lst[j])[:-4]
            if not os.path.exists(pair_path):
                os.makedirs(pair_path)            
            src_pd = pandas.DataFrame(src_new[:num])
            dst_pd = pandas.DataFrame(dst_new[:num])
            src_pd.to_csv(os.path.join(pair_path, '1.txt'), sep='\t', header=False, index=False)
            dst_pd.to_csv(os.path.join(pair_path, '3.txt'), sep='\t', header=False, index=False)
            total_files = total_files + 1

print("Successfully created dataset DIVO_{}, with total files ={}".format(str(num), str(total_files)))


        