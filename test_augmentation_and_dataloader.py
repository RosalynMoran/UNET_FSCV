#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Nov 4th 2022

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
from torch.utils.data import DataLoader
from model import UNet3D
from dataloaders_FSCV import FSCV_dataset


device = 'cpu'#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print('torch.version.cuda',torch.version.cuda)

dataset     =  FSCV_dataset()
dataloader  =  DataLoader(dataset, batch_size=1)  
train_morph, train_orig = next(iter(dataloader))
print('size train  t',np.shape(train_morph))
print('size orig',np.shape(train_orig))
 
#def check_morph():
#    
 
#
# 
#    
#     
#         
#         
#         target     = train_orig[0,:]
#         input       = train_morph[0,:]
#         
#         axes[i ,0].imshow( target )        
 
#         axes[i ,1].imshow(input )
 
#  
#    plt.show()
#
#if __name__ == '__main__':
#    
#     check_morph()         
#         
# 