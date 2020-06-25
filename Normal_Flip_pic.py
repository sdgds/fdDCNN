#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:37:32 2019

@author: dell
"""

import torch.nn as nn
import copy
import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader     
import torchvision.transforms as transforms
import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import io as dnn_io
from dnnbrain.dnn import models
import csv
from PIL import Image
from PIL import ImageEnhance
                    


'1. Face Representation and Flip Face Representation'
###############################################################################
###############################################################################               
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms as transforms
import random
import sys
sys.path.append('/home/dell/Desktop/My_module')
import reconstruction_layer
import Guided_CAM


class NET(nn.Module):
    def __init__(self, model, selected_layer):
        super(NET, self).__init__()
        self.model = model
        self.selected_layer = selected_layer
        self.feature_map = 0
    def hook_layer(self):
        def hook_function(module, layer_in, layer_out):
            self.feature_map = layer_out
        self.model.features[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        self.model(x)
        
    
    
def Gragh(pic_number):        
    # load picture
    data_transforms = {
        'Prepare': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]),
        'Prepare_with_normalize': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225])]),
        'Prepare_flip': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor()]),
        'Prepare_flip_with_normalize': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225])])
        }
#    # Nature_stimili
#    number_of_picture = pic_number 
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_face_1000_FITW.csv', transform=data_transforms['Prepare'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0)
#    fig = plt.figure(figsize=(6,6), dpi=200) 
#    ax = fig.add_subplot(1,5,1)
#    ax.imshow(picimg[0].permute(1,2,0).data)
#    ax.axis('off')
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_face_1000_FITW.csv', transform=data_transforms['Prepare_with_normalize'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0)
#    picimg.requires_grad=True
    
    # Normal or Flip
    number_of_picture = pic_number              
    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Prepare'])
    picimg = picdataset[number_of_picture][0].unsqueeze(0) 
    fig = plt.figure(figsize=(6,6), dpi=200) 
    ax = fig.add_subplot(1,5,1)
    ax.imshow(picimg[0].permute(1,2,0).data)
    ax.axis('off')
    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Prepare_with_normalize'])
    picimg = picdataset[number_of_picture][0].unsqueeze(0)
    picimg.requires_grad=True
    
    
    # load model
    non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
    net = copy.deepcopy(non_face_net).cpu().eval()
    net_truncate = NET(net, selected_layer=11)    
    origin_alexnet = torchvision.models.alexnet(pretrained=True)
    net_alexnet = copy.deepcopy(origin_alexnet).eval()
    alexnet_truncate = NET(net_alexnet, selected_layer=11)
    target_layer = 11    
    target_class = 0
    
    ## Alexnet
    # model out and class
    face_channel_alexnet = [184, 124]
    for n,channel in enumerate(face_channel_alexnet):       
        alexnet_truncate.layeract(picimg)
        alexnet_truncate.feature_map[0,channel,:]        
        ax = fig.add_subplot(1,5,n+2)
        cax = ax.matshow(alexnet_truncate.feature_map[0,channel,:].data.numpy(), cmap='jet',vmin=0, vmax=25, alpha=1)
        ax.axis('off')
    
    ## Non-face alexnet
    # model out and class
    face_channel = [28, 49]
    for n,channel in enumerate(face_channel):      
        net_truncate.layeract(picimg)
        net_truncate.feature_map[0,channel,:] 
        ax = fig.add_subplot(1,5,n+4)
        cax = ax.matshow(net_truncate.feature_map[0,channel,:].data.numpy(), cmap='jet',vmin=0, vmax=25, alpha=1)  # Heat map
        ax.axis('off')          
    fig.savefig('/home/dell/Desktop/Visulization/Results_pictures/front_%d_heatmap.jpg' %pic_number)


#picture_number = [22, 142, 498, 529, 554]     # nature stim
picture_number = [46, 223, 402, 429, 552]     # normal or flip stim
for pic in picture_number:
    Gragh(pic)
    
    
    
f = os.listdir('/Users/mac/Desktop/Face/Results_pictures')
f.remove('.DS_Store')
f.remove('final_pic')
for pic in f:
    img = Image.open('/Users/mac/Desktop/Face/Results_pictures/'+pic)
    img = np.array(img)
    img = img[520:688,144:1084,:]
    img = Image.fromarray(img.astype('uint8'))
    img.save('/Users/mac/Desktop/Face/Results_pictures/final_pic/'+pic)
    
    
