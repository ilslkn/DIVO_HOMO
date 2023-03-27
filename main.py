#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:23:54 2023

@author: laikongning
"""

import argparse
import configparser

parser = argparse.ArgumentParser()

parser.add_argument('--p', dest = 'result_path', type = str, help = 'The path to store the results', required=True)
parser.add_argument('--dx', dest = 'DX_USE', type = str, choices=['y','n'], default= 'n')
parser.add_argument('--resume', dest = 'RESUME', type = str, choices = ['y', 'n'], default = 'n')
parser.add_argument('--epoch', dest = 'EPOCH', type = int, default=100)

args = parser.parse_args()

config = configparser.ConfigParser()
config['PATH'] = {
        'root_dir' : '../../../../data/srtp/Multi_view_tracking/DIVO_4',
        'train_dataset_path' : '../../../../data/srtp/Multi_view_tracking/DIVO_4/Train',
        'test_dataset_path' : '../../../../data/srtp/Multi_view_tracking/DIVO_4/Test'        
}

config['PATH']['result_path'] = args.result_path
with open('conf.ini', 'w') as configfile:
    config.write(configfile)
