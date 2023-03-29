#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:49:41 2022

@author: laikongning
"""
import torch
import cv2
import numpy as np
import torch.nn.functional as f
import pandas
import os
import datetime
import logging
import sys
import configparser

config = configparser.ConfigParser()
config.read('conf.ini',encoding='utf-8')
result_path = config.get('PATH', 'result_path')
max_n_pts = config.getint('PARA','Max Points')



def calculate_pred_pts(H, src_pts):
    batch = H.shape[0]
    ones = torch.ones((batch,1))
    H_ = torch.cat((H,ones),dim=1).reshape(batch,3,3)
    
    ones_2 = torch.ones((batch,1,max_n_pts))
    src_pts_ = torch.cat((src_pts.transpose(1,2),ones_2),dim=1)

    temp = torch.bmm(H_.float(),src_pts_.float())
    
    p = temp[:,2]
    p1 = torch.stack((p,p,p)).transpose(0,1)
    pred_pts = (temp/p1)[:,:2].transpose(1,2)
    return pred_pts


def read_tablemethod(filename):
    data = pandas.read_table(filename, header=None, delim_whitespace=True)
    return data

def DIVO_get_points(path1,path2):
    src_data = read_tablemethod(path1)
    dst_data = read_tablemethod(path2)
    src = np.zeros(src_data.shape)
    dst = np.zeros(dst_data.shape)
    for x in range(src_data.shape[0]):
        for y in range(src_data.shape[1]):
            src[x, y] = src_data[y][x]
            dst[x, y] = dst_data[y][x]
            
    retval = {"src_pts": torch.tensor(src).float(), "dst_pts": torch.tensor(dst).float()}
    return retval

def beijing(sec, what):
    beijing_time = datetime.datetime.now() 
    return beijing_time.timetuple()



def logger_init(log_file_name = 'monitor',
                log_level = logging.DEBUG,
                log_dir = result_path + '/logs/',
                only_file = False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),
                                      logging.StreamHandler(sys.stdout)]
                            )

