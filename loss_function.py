#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 15:23:17 2022

@author: laikongning
"""

import torch
import cv2
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('conf.ini',encoding='utf-8')
DX_USE = config.get('SWITCH', 'DX_USE')
max_n_pts = config.getint('PARA','Max Points')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyMSELoss(nn.Module):
    def __init__(self):
        super(MyMSELoss, self).__init__()

    def forward(self, H_out, dx, pts1, pts2):
        batch = pts1.shape[0]
        ones = torch.ones((batch, 1)).to(device)
        H = torch.cat((H_out, ones), dim=1).reshape(batch, 3, 3)

        ones_2 = torch.ones((batch, 1, max_n_pts)).to(device)
        if(DX_USE):
            src_pts = torch.cat(((pts1 + dx).transpose(1, 2), ones_2), dim=1)
        else:
            src_pts = torch.cat((pts1 .transpose(1, 2), ones_2), dim=1)
        temp = torch.bmm(H, src_pts)

        p = temp[:, 2]
        p1 = torch.stack((p, p, p)).transpose(0, 1)
        pred_pts = (temp / p1)[:, :2].transpose(1, 2)
        #loss = f.mse_loss(pred_pts, pts2) 
        
        
        w = torch.exp(-((pred_pts[:,:,0] - pts2[:,:,0]) ** 2 + (pred_pts[:,:,1] - pts2[:,:,1]) ** 2)).detach() #without grad
        
        #距离
        #w = torch.sqrt(w / torch.sum(w))
        w = w / torch.sum(w) #后百分之十=0
        des_sorted, index = torch.sort(w, dim = 1)
        for b in range(batch):
            for i in range(int(0.1*max_n_pts)):
                w[b][index[b][i]] = 0
                
        w2 = torch.stack((w,w),dim=2).detach()
        if(DX_USE):
            dx_res = torch.zeros((batch, max_n_pts, 2))
            for b in range(batch):
                for i in range(max_n_pts):
                    dx_res[b][i][0] = max(0, abs(dx[b][i][0] + pts1[b][i][0])-1)
                    dx_res[b][i][1] = max(0, abs(dx[b][i][1] + pts1[b][i][1])-1)
            
            
            loss = f.mse_loss(w2 * pred_pts, w2 * pts2) + torch.sum(dx_res,dim=(0,1,2))/max_n_pts # 区间惩罚 hinge loss max(0,m-dx)
        else:
            loss = f.mse_loss(w2 * pred_pts, w2 * pts2)
        return loss


