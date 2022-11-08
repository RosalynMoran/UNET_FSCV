#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Oct 3rd 2022

@author:  Rosalyn
"""

#import pytorch_lightning as pl
from torch import nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import UNet3D, UNet2D
from dataloaders_FSCV import FSCV_dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print('torch.version.cuda',torch.version.cuda)


def train():
    
    # 1. GET DATA

    dataset     =  FSCV_dataset()
    dataloader  =  DataLoader(dataset, batch_size=1,  shuffle=True)   
    
    # 2. Set up model
    model       =  UNet2D(in_channels=1, out_channels=1)     # default args
    model       =  model.to(device)
    optimizer   =  optim.Adam(model.parameters(), lr=1e-4)   # could lower learning rate to 1e-4
   
    
    losses =[]
    
    # iterate through gradient updates
    epochs = 100
    
    for epoch in range(epochs):
    
        losses.append(0)
 
    
        for n, (train_morph, train_orig) in enumerate(dataloader):
 
           # train_morph = np.squeeze(train_morph[0,:,:,0])
           # train_orig = np.squeeze(train_orig[0,:,:,0])
            
         
            morph   = train_morph.to(device)   
            orig    = train_orig.to(device)
            
           # print('size of input images',np.shape(LFimages))
           # print('size of target images',np.shape(HFtargets))
            im_pred    = model.forward(morph)
            loss       = model.loss(im_pred,orig.float())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            losses[-1] += loss.item()
            
    
         
        print('Epoch [%d / %d] ELBO: %f' %(epoch+1, epochs, losses[-1]))
        if np.remainder(epoch,10)  ==0:
            
            torch.save(model.state_dict(),'/home/rosalyn/Documents/tmp/Rosalyn_Simplified_UNET/Checkpoints/UNET_TEST_epoch'+str(epoch)+'.pth') 
                
   ## plot things
    # plt_no = 4
    # figure, axes = plt.subplots(nrows=plt_no , ncols=2)   
 
    # for i in range(plt_no):
    
    #    "test" data
    #     LFimages, HFimage = next(iter(dataloader)) 
    #     print('High Field Image', HFimage())
    #     axes[i ,0].plot(HFimage[i,0,:])        
    #     prediction      = model.forward(images)
    #     prediction  = prediction.detach().numpy()
    #     axes[i ,1].plot(prediction[i,0,:])
      
    # plt.show()
    # plt.show()
      
  


if __name__ == '__main__':
    
    train()         
         
 