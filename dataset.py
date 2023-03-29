#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:16:19 2023

@author: laikongning
"""

from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import os
import torchvision
from torch import nn
from model import *
from utils import *

import pandas

class TrainDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_list = os.listdir(self.root_dir)

    def __getitem__(self, idx):
        img_label = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_label)

        img1_path = os.path.join(img_path, '1.txt')
        img2_path = os.path.join(img_path, '3.txt')
        datas = DIVO_get_points(img1_path, img2_path)
        
        return datas  

    def __len__(self):
        return len(self.img_list)