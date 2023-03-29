#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:12:53 2023

@author: laikongning
"""

import torch
from loss_function import MyMSELoss
from model import TrainModel
from dataset import TrainDataset
from utils import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import configparser

config = configparser.ConfigParser()
config.read('conf.ini',encoding='utf-8')
train_dataset_path = config.get('PATH','train_dataset_path')
test_dataset_path = config.get('PATH','test_dataset_path')
result_path = config.get('PATH', 'result_path')
BATCH = config.getint('PARA', 'Batch Size')
start_learning_rate = config.getfloat('PARA','Learning Rate')
max_n_pts = config.getint('PARA','Max Points')
EPOCH = config.getint('PARA', 'EPOCH')
resume = config.get('SWITCH','RESUME')
DX_USE = config.get('SWITCH', 'DX_USE')
ckpt_path = os.path.join(result_path, 'ckpt_model.pth')
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_train_step = 0
    START_EPOCH = 0
    min_test_loss = 10


    
    train_dataset = TrainDataset(train_dataset_path)
    test_dataset = TrainDataset(test_dataset_path)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    
    model = TrainModel(size=max_n_pts).to(device)
    
    loss_fn = MyMSELoss().to(device)
    
    learning_rate = start_learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1,patience=5,verbose=True)
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    writer = SummaryWriter(result_path+'/logs')
    
    logging.Formatter.converter = beijing
    
    logger_init()

    if resume == 'y':
        if os.path.isfile(ckpt_path):
            logging.info("Resume from checkpoint...")
            print(ckpt_path)
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            START_EPOCH = checkpoint['epoch'] + 1
            total_train_step = checkpoint['train_step'] + 1
            logging.info("=====> loaded checkpoint epoch{}".format(checkpoint['epoch']+1))
        else:
            logging.info("=====> no checkpoint found")
    
    if DX_USE == 'y' :
        logging.info("///////dx is now used in loss function/////")
    else:
        logging.info("///////dx is now not used in loss function/////")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    for epoch in range(START_EPOCH,EPOCH):
        logging.info("------------ epoch {} ------------".format(epoch + 1))
        
        #train
        model.train()
        for datas in train_dataloader:
            src_pts = datas['src_pts'].to(device)
            dst_pts = datas['dst_pts'].to(device)
            pts = torch.cat((src_pts,dst_pts),dim=2)
            
            optimizer.zero_grad()
            output = model(pts.to(device))
            H_out = output['H'].to(device)
            dx = output['dX'].to(device)
            loss = loss_fn(H_out, dx, src_pts, dst_pts)
            loss.backward()
            optimizer.step()
    
            total_train_step = total_train_step + 1
            writer.add_scalar('train_loss', loss.item(),total_train_step)
            if total_train_step % 50 == 0:
                logging.info("step: {}, Train Loss: {}".format(total_train_step, loss.item()))
    
    
        model.eval()
        total_test_loss = 0
        total_test_loss_cv2 = 0
        with torch.no_grad():
            
            for datas in test_dataloader:
                src_pts = datas['src_pts'].to(device)
                dst_pts = datas['dst_pts'].to(device)
                pts = torch.cat((src_pts,dst_pts),dim=2)
                
                output = model(pts.to(device))
                H_out = output['H'].to(device)
                
                    
                dx = output['dX'].to(device)
                loss = loss_fn(H_out, dx, src_pts, dst_pts)
    
    
                total_test_loss += loss.item()
                
            scheduler.step(total_test_loss)
            if total_test_loss < min_test_loss:
                min_test_loss = total_test_loss
                checkpoint = {"model_state_dict":model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch,
                              "train_step": total_train_step}
                torch.save(checkpoint, ckpt_path)
                logging.info("ckpt_model in epoch {} has been saved".format(epoch))
            logging.info("Test Loss: {}".format(total_test_loss))    
            writer.add_scalar('test_loss', total_test_loss,epoch+1)

    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch,
                  "train_step": total_train_step}
    if DX_USE == 'y':
        torch.save(checkpoint, result_path + '/ckpt_model_dx.pth')
    else:
        torch.save(checkpoint, result_path + '/ckpt_model_.pth')

def test_opencv():
    logger_init()
    logging.info("------------ Testing the performance of OpenCV ------------")
    train_dataset = TrainDataset(train_dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    step = 0
    for datas in train_dataloader:
        step+=1
        src_pts = datas['src_pts']
        dst_pts = datas['dst_pts']
        batch = src_pts.shape[0]
        H_cv2 = torch.zeros((batch,3,3))
        for i in range(batch):
            s = src_pts[i].numpy()
            d = dst_pts[i].numpy()
            H, mask = cv2.findHomography(s,d,cv2.RANSAC,5.0)
            H_cv2[i] = torch.tensor(H)

        ones_2 = torch.ones((batch, 1, max_n_pts))
        src_ = torch.cat((src_pts.transpose(1, 2), ones_2), dim=1)
        temp = torch.bmm(H_cv2, src_)
        p = temp[:, 2]
        p1 = torch.stack((p, p, p)).transpose(0, 1)
        pred_pts = (temp / p1)[:, :2].transpose(1, 2)
        w = torch.exp(-((pred_pts[:, :, 0] - dst_pts[:, :, 0]) ** 2 + (pred_pts[:, :, 1] - dst_pts[:, :, 1]) ** 2)).detach()
        w = w / torch.sum(w)  # 后百分之十=0
        des_sorted, index = torch.sort(w, dim=1)
        for b in range(batch):
            for i in range(int(0.1 * max_n_pts)):
                w[b][index[b][i]] = 0
        w2 = torch.stack((w, w), dim=2).detach()
        loss = torch.nn.functional.mse_loss(w2 * pred_pts, w2 * dst_pts)
        if step%50==0 :
            logging.info("step: {}, openCV loss: {}".format(step, loss.item()))
