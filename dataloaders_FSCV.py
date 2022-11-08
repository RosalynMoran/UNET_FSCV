#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2, 13:58:03 2022

@author: rosalynmoran
# TIME SERIES DATA AUGMENTATION:
# https://www.arundo.com/articles/tsaug-an-open-source-python-package-for-time-series-augmentation
#>>> from tsaug import RandomTimeWarp, RandomMagnify, RandomJitter, RandomTrend
#>>> my_aug = (
#...    RandomMagnify(max_zoom=4.0, min_zoom=2.0) * 2
#...    + RandomTimeWarp() * 2
#...    + RandomJitter(strength=0.1) @ 0.5
#...    + RandomTrend(min_anchor=-0.5, max_anchor=0.5) @ 0.5
#... )
#>>> X_aug, Y_aug = my_aug.run(X, Y)
#>>> plot(X_aug, Y_aug)
#"""
import os.path as op
import os
import numpy as np
import torch
from torch.utils.data import  Dataset
from torch.utils.data import   DataLoader
import matplotlib.pyplot as plt
import SimpleITK as sitk
from glob import glob
import torchio as tio
import h5py as h5

class FSCV_dataset(Dataset):
    def __init__(self, data_dir= '/home/rosalyn/Documents/tmp/Rosalyn_Simplified_UNET/data/Train'):
        
        self.data_dir       = data_dir
        self.filelist       = os.listdir(data_dir) #matfile voltamograms and labels
        self.num_samples    = len(self.filelist )
#        self.images =[]
#        self.origs = []
#        self.load(self.data_dir,self.filelist)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        matfile = op.join(self.data_dir,self.filelist[idx],'volts.h5')
        
        with h5.File(matfile,'r') as f:
             orig = f['dataset1'][()] 
       
        # Sample 1000 and apply the same transform to all
       # print('RM size 0',np.shape(orig))
        orig = np.expand_dims(orig,0)
       # print('RM size 0',np.shape(orig))
        orig = np.expand_dims(orig,3)
       # print('RM size 0',np.shape(orig))
        cf= tio.ScalarImage(tensor=orig) 
        
        #figure, axes = plt.subplots(nrows=1 , ncols=2)   
        #plt.imshow(orig[0,:,:,0] ) 
       # plt.plot(orig[0,6,:,0] ,color='blue') 
       # plt.plot(orig[0,26,:,0] ,color='black') 
       # plt.plot(orig[0,6,:,0] ) 
       # plt.show()
        spatial_transforms = {
                tio.RandomElasticDeformation(
                num_control_points=(6,30,16),
                max_displacement=(4,240,0), #must have 0 for z dimension
                locked_borders=2,
                ): 0.5,
    
                tio.RandomAffine(scales = (0,0.1,0),
                                 degrees =(0,30,0),
                                 translation =(0,50,0),
                                 ): 0.5}
#
        transform = tio.Compose([
                tio.OneOf(spatial_transforms, p=1),
                tio.RescaleIntensity(out_min_max=((0, 1)))
                ])
    
    
        transform_orig = tio.RescaleIntensity(out_min_max=((0, 1))) 
        image_orig = tio.ScalarImage(tensor = orig)
        orig = transform_orig(image_orig) # normalise the target to 0 ---1 / min---- max
        orig = orig.data.numpy()
        
        morph_orig   =  transform(cf)
        morph        =  morph_orig.data.numpy()
        xx = np.shape(morph)
      #  plt.plot(morph[0,6,:,0] ,color='red') 
      #  plt.plot(morph[0,26,:,0] ,color='orange') 
      #  plt.show()
      
        select_voltsi = np.random.permutation(xx[1])
        morph_sample  = morph[:,select_voltsi[0:200],:,:]
        orig_sample  = orig[:,select_voltsi[0:200],:,:]
        
        return morph_sample,orig_sample
 
 
#
#      

class FSCV_TEST_dataset(Dataset):
    def __init__(self, data_dir= '/home/rosalyn/Documents/tmp/Rosalyn_Simplified_UNET/data/Test'):
        
        self.data_dir       = data_dir
        self.filelist       = os.listdir(data_dir) #matfile voltamograms and labels
        self.num_samples    = len(self.filelist )
#        self.images =[]
#        self.origs = []
#        self.load(self.data_dir,self.filelist)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        matfile = op.join(self.data_dir,self.filelist[idx],'volts.h5')
        
        with h5.File(matfile,'r') as f:
             orig = f['dataset1'][()] 
       
 
        orig = np.expand_dims(orig,0)
 
        orig = np.expand_dims(orig,3)
 
        cf= tio.ScalarImage(tensor=orig) 
 
        transform = tio.Compose([
                tio.RescaleIntensity(out_min_max=((0, 1)))
                ])
        
        # recasle intensitis - orig not used for testing - no morph applied
        transform_orig = tio.RescaleIntensity(out_min_max=((0, 1))) 
        image_orig = tio.ScalarImage(tensor = orig)
        orig = transform_orig(image_orig) # normalise the target to 0 ---1 / min---- max
        orig = orig.data.numpy()
        
 
        morph_orig   =  transform(cf)
        morph        =  morph_orig.data.numpy()
        xx = np.shape(morph)
 
        select_voltsi = np.random.permutation(xx[1])
        morph_sample  = morph[:,select_voltsi[0:200],:,:]
        orig_sample  = orig[:,select_voltsi[0:200],:,:]
        
        return morph_sample,orig_sample    