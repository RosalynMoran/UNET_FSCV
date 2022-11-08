#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Oct 3rd 2022

@author:  Rosalyn
"""

#import pytorch_lightning as pl
from torch import nn
import os
import torch
import torch.optim as optim
#from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import UNet3D, UNet2D
from dataloaders_FSCV import  FSCV_TEST_dataset, FSCV_dataset


device = 'cpu'#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print('torch.version.cuda',torch.version.cuda)


def infer():
    
    # 1. GET DATA FOR INFERENCE 

    dataset     =  FSCV_TEST_dataset()
    dataloader  =  DataLoader(dataset, batch_size=1,  shuffle=True)   
    
    # 2. Load the model from checkpoint
    PATH_CKP = '/home/rosalyn/Documents/tmp/Rosalyn_Simplified_UNET/Checkpoints/UNET_TEST_epoch80.pth'
    #
    model       =  UNet2D(in_channels=1, out_channels=1) # default args
    model.load_state_dict(torch.load(PATH_CKP))
    model       =  model.to(device)
    model.eval()
    torch.no_grad() 
 
    # 3. plot things and save niftiis 
    PATH_PREDS = '/home/rosalyn/Documents/tmp/Rosalyn_Simplified_UNET/Model_Predictions'
    plt_no = 2
    figure, axes = plt.subplots(nrows=plt_no , ncols=3)   
 
   # for i in range(plt_no):
    for i, (train_morph, train_orig) in enumerate(dataloader):  
         print('i',i)
 
         if i >1:
             break
        
         morph   = train_morph.to(device)   
         orig    = train_orig.to(device)
         print('size of input sweeps',np.shape(morph))
         print('size of target sweeps',np.shape(orig))
         
         V_pred    = model.forward(morph)    

         print('size of predicted sweeps',np.shape(V_pred))
         
         target     = np.squeeze(orig.detach().numpy())
         prediction = np.squeeze(V_pred.detach().numpy())
         morph      = np.squeeze(morph.detach().numpy())
         print('size of squeezeed target sweeps',np.shape(target))
         axes[i ,0].plot( np.transpose(morph[:,:]))        
         axes[i ,1].plot( np.transpose( prediction[:,:]))
         axes[i ,2].plot( np.transpose(target[ :,:]))
         
 
    plt.show()

if __name__ == '__main__':
    
     infer()         
         
 