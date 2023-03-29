#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:03:23 2022

@author: laikongning
"""
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

class TrainModel(torch.nn.Module):
    def __init__(self, size, dim=512, num_layers=8):
        super(TrainModel, self).__init__()
        self.num_layers = num_layers
        self.nhead = 8
        self.encoder_layers_main = torch.nn.ModuleList([])
        self.encoder_layers_H = torch.nn.ModuleList([])
        self.encoder_layers_dX = torch.nn.ModuleList([])
        self.mask_layers = torch.nn.ModuleList([])
        for n in range(num_layers):
            self.encoder_layers_main.append(
                torch.nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=1024, nhead=self.nhead,
                                                 batch_first=True, dropout=0.))
        for i in range(3):
            self.encoder_layers_H.append(
                torch.nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=1024, nhead=self.nhead,
                                                 batch_first=True, dropout=0.))
            self.encoder_layers_dX.append(
                torch.nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=1024, nhead=self.nhead,
                                                 batch_first=True, dropout=0.))
            
            #self.mask_layers.append(
                #torch.nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=256, nhead=self.nhead,
                                                   #batch_first=True, dropout=0.))
                                                
#        self.pooling = torch.nn.AvgPool2d(4)
#        self.linear = torch.nn.Linear(50, 8)
        self.linear1 = torch.nn.Linear(4,dim)
        #self.F = torch.nn.Flatten()
        self.pooling_H = torch.nn.AvgPool2d((size,1),stride=1)
        self.linear_H = torch.nn.Linear(dim, 8)
        self.linear_dX = torch.nn.Linear(dim,2)


    def forward(self, X):
        # X = [batch, seq_len, emb]

        #X_shape = X.shape
        #M = (X == 0)[:,:,0]
        X = self.linear1(X)
        for n in range(self.num_layers):
            #X = self.encoder_layers[n](X, src_mask=None, src_key_padding_mask=M)
            X = self.encoder_layers_main[n](X, src_mask=None, src_key_padding_mask=None)

        H = self.encoder_layers_H[0](X, src_mask=None, src_key_padding_mask=None)
        dX = self.encoder_layers_dX[0](X, src_mask=None, src_key_padding_mask=None)
        for n in range(2):
            #X = self.encoder_layers[n](X, src_mask=None, src_key_padding_mask=M)
            H = self.encoder_layers_H[n+1](H, src_mask=None, src_key_padding_mask=None)
            dX = self.encoder_layers_dX[n+1](dX, src_mask=None, src_key_padding_mask=None)
   
#        X = self.pooling(X)
        H = self.pooling_H(H)
        H = F.relu(H)
        H = self.linear_H(H).squeeze()
        
        dX = self.linear_dX(dX)
        # X = torch.reshape(X, (X_shape[0], X_shape[1], -1))
        return {
            'H': H,
            'dX': dX
        }
