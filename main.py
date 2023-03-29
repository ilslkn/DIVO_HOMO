#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:23:54 2023

@author: laikongning
"""

import argparse
import configparser
import os


parser = argparse.ArgumentParser()

parser.add_argument('--p', dest = 'result_path', type = str, help = 'The path to store the results', required=True)
parser.add_argument('--dx', dest = 'DX_USE', type = str, choices=['y','n'], default= 'n')
parser.add_argument('--resume', dest = 'RESUME', type = str, choices = ['y', 'n'], default = 'n')
parser.add_argument('--epoch', dest = 'EPOCH', type = str, default='100')
parser.add_argument('--n', dest = 'n_pts', type = str, default='10')

args = parser.parse_args()

config = configparser.ConfigParser()
root = '/data/srtp/Multi_view_tracking'
config.add_section('PATH')
config['PATH']['root_dir'] = os.path.join(root,'DIVO_{}'.format(args.n_pts))
config['PATH']['train_dataset_path'] = config['PATH']['root_dir']+'/train'
config['PATH']['test_dataset_path'] = config['PATH']['root_dir']+'/test'
config['PATH']['result_path'] = os.path.join(root,args.result_path)

config.add_section('PARA')
config['PARA']['EPOCH'] = args.EPOCH
config['PARA']['Learning Rate'] = '1e-6'
config['PARA']['Batch Size'] = '4'
config['PARA']['Max Points'] = args.n_pts

config.add_section('SWITCH')
config['SWITCH']['DX_USE'] = args.DX_USE
config['SWITCH']['RESUME'] = args.RESUME

with open('conf.ini', 'w') as configfile:
    config.write(configfile)

from train import train
#train()
from train import test_opencv
test_opencv()