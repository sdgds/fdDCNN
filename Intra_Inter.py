#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:29:29 2020

@author: mac
"""

import torch.nn as nn
import copy
import os
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.utils.data as Data
import torchvision
import csv
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader     
import torchvision.transforms as transforms

import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import io as dnn_io
from dnnbrain.dnn import models


transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
# face and object stimuli
picdataset_face = dnn_io.PicDataset('/home/dell/Desktop/205cat/face_80.csv', transform=transform)
picdataset_noface = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_nonpeople_204x80_256Obj.csv', transform=transform)

# load non-face net and alexnet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
net = copy.deepcopy(non_face_net)
origin_alexnet = torchvision.models.alexnet(pretrained=True).to(device)
net_alexnet = copy.deepcopy(origin_alexnet)


### extract activation and feature map
class NET(nn.Module):
    def __init__(self, model, selected_layer):
        super(NET, self).__init__()
        self.model = model
        self.selected_layer = selected_layer
        self.conv_output = 0
        self.feature_map = 0
    def hook_layer(self):
        def hook_function(module, layer_in, layer_out):
            self.feature_map = layer_out
            self.conv_output = torch.mean(layer_out, (0,2,3))
        self.model.features[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        self.model(x)

    

"""selected_layer = 11"""
### 80 Face  11th-layer  non-face net
from scipy import stats

selected_layer = 11
net_truncate = NET(net, selected_layer=selected_layer)
channel_face = []
for imgs,_ in picdataset_face:
    net_truncate.layeract(imgs.unsqueeze(0).to(device))
    channel_face.append(net_truncate.conv_output.cpu().data.numpy())
channel_face = np.array(channel_face)


### 80 Face  11th-layer  Alexnet 
net_truncate_alexnet = NET(net_alexnet, selected_layer=selected_layer)
channel_face_alexnet = []
for imgs,_ in picdataset_face:
    net_truncate_alexnet.layeract(imgs.unsqueeze(0).to(device))
    channel_face_alexnet.append(net_truncate_alexnet.conv_output.cpu().data.numpy())
channel_face_alexnet = np.array(channel_face_alexnet)


np.save('/home/dell/Desktop/DNN2Brain/Temp_results/channel_face_alexnet.npy', channel_face_alexnet)
np.save('/home/dell/Desktop/DNN2Brain/Temp_results/channel_face.npy', channel_face)








# load data(trained and nontrained obj)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
    }
     
picdataset_train = dnn_io.PicDataset('/home/dell/Desktop/data_sony\win7\win10\dell/test_effect/500th_class.csv', transform=data_transforms['val'])
picdataset_nontrain = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_nonpeople_204x80_256Obj.csv', transform=data_transforms['val'])

#load net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
net = copy.deepcopy(non_face_net)
origin_alexnet = torchvision.models.alexnet(pretrained=True).to(device)
alexnet = copy.deepcopy(origin_alexnet)


######################
'Neuron level test'
######################
class NET(torch.nn.Module):
    def __init__(self, model, selected_layer):
        super(NET, self).__init__()
        self.model = model
        self.selected_layer = selected_layer
        self.conv_output = 0
        self.feature_map = 0
    def hook_layer(self):
        def hook_function(module, layer_in, layer_out):
            self.feature_map = layer_out
            self.conv_output = torch.mean(layer_out, (0,2,3))
        self.model.features[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if index == self.selected_layer:
                break

selected_layer = 11
net_truncate = NET(net.to(device), selected_layer=selected_layer)
alexnet_truncate = NET(alexnet.to(device), selected_layer=selected_layer)


# obj(trained) activation in Conv5
n03042490_alexnet = []
for n,(img,_) in enumerate(picdataset_train):
    print(n)
    alexnet_truncate.layeract(img.unsqueeze(0).to(device))
    n03042490_alexnet.append(alexnet_truncate.conv_output.cpu().data.numpy())
n03042490_alexnet = np.array(n03042490_alexnet)
np.save('/home/dell/Desktop/DNN2Brain/Temp_results/n03042490_alexnet.npy', n03042490_alexnet)


n03042490_net = []
for n,(img,_) in enumerate(picdataset_train):
    print(n)
    net_truncate.layeract(img.unsqueeze(0).to(device))
    n03042490_net.append(net_truncate.conv_output.cpu().data.numpy())
n03042490_net = np.array(n03042490_net)
np.save('/home/dell/Desktop/DNN2Brain/Temp_results/n03042490_net.npy', n03042490_net)


# obj(nontrained) activation in Conv5
Object_nontrain = '018'
class_alexnet = []
for index, (img,_) in enumerate(picdataset_nontrain):
    print(index)
    if picdataset_nontrain.picname[index][:3]==Object_nontrain:
        alexnet_truncate.layeract(img.unsqueeze(0).to(device))
        class_alexnet.append(alexnet_truncate.conv_output.cpu().data.numpy())
class_alexnet = np.array(class_alexnet)
np.save('/home/dell/Desktop/DNN2Brain/Temp_results/class_alexnet.npy', class_alexnet)


class_net = []
for index, (img,_) in enumerate(picdataset_nontrain):
    print(index)
    if picdataset_nontrain.picname[index][:3]==Object_nontrain:
        net_truncate.layeract(img.unsqueeze(0).to(device))
        class_net.append(net_truncate.conv_output.cpu().data.numpy())
class_net = np.array(class_net)
np.save('/home/dell/Desktop/DNN2Brain/Temp_results/class_net.npy', class_net)




# load matrix
channel_face_alexnet = np.load('/Users/mac/Desktop/channel_face_alexnet.npy')
channel_face = np.load('/Users/mac/Desktop/channel_face.npy')
n03042490_alexnet = np.load('/Users/mac/Desktop/n03042490_alexnet.npy')
n03042490_net = np.load('/Users/mac/Desktop/n03042490_net.npy')
class_alexnet = np.load('/Users/mac/Desktop/class_alexnet.npy')
class_net = np.load('/Users/mac/Desktop/class_net.npy')

plt.figure(figsize=(6,6), dpi=80)
plt.imshow(np.corrcoef(np.vstack((channel_face_alexnet[:80], n03042490_alexnet[:80], class_alexnet))))
plt.colorbar()
plt.axis('off')

plt.figure(figsize=(6,6), dpi=80)
plt.imshow(np.corrcoef(np.vstack((channel_face[:80], n03042490_net[:80], class_net))))
plt.colorbar()
plt.axis('off')




def lower_triangular(R):  
    global r
    r = np.array([])     
    for i in range(1,R.shape[0]):
        for j in range(i):
            r = np.append(r, R[i,j])
    return r


R_face_alexnet = np.where(np.corrcoef(channel_face_alexnet[:80])==1, np.nan, np.corrcoef(channel_face_alexnet[:80]))
R_obj_trained_alexnet = np.where(np.corrcoef(n03042490_alexnet[:80])==1, np.nan, np.corrcoef(n03042490_alexnet[:80]))
R_obj_nontrained_alexnet = np.where(np.corrcoef(class_alexnet)==1, np.nan, np.corrcoef(class_alexnet))
R_face_net = np.where(np.corrcoef(channel_face[:80])==1, np.nan, np.corrcoef(channel_face[:80]))
R_obj_trained_net = np.where(np.corrcoef(n03042490_net[:80])==1, np.nan, np.corrcoef(n03042490_net[:80]))
R_obj_nontrained_net = np.where(np.corrcoef(class_net)==1, np.nan, np.corrcoef(class_net))

with open('/Users/mac/Desktop/Intra.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for i in range(3160):
        csv_write.writerow(['face', lower_triangular(R_face_alexnet)[i], lower_triangular(R_face_net)[i]])
    for i in range(3160):
        csv_write.writerow(['unseen', lower_triangular(R_obj_nontrained_alexnet)[i], lower_triangular(R_obj_nontrained_net)[i]])    



corr_alexnet = np.corrcoef(np.vstack((channel_face_alexnet[:80], n03042490_alexnet[:80], class_alexnet)))
corr_net = np.corrcoef(np.vstack((channel_face[:80], n03042490_net[:80], class_net)))

face_objtrained_alexnet = corr_alexnet[0:80,80:160]
face_objnontrained_alexnet = corr_alexnet[0:80,160:]
objtrained_objnontrained_alexnet = corr_alexnet[80:160,160:]
face_objtrained_net = corr_net[0:80,80:160]
face_objnontrained_net = corr_net[0:80,160:]
objtrained_objnontrained_net = corr_net[80:160,160:]

with open('/Users/mac/Desktop/Inter.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for i in range(3160):
        csv_write.writerow(['face_unseen', 
                            lower_triangular(face_objnontrained_alexnet)[i], 
                            lower_triangular(face_objnontrained_net)[i]])
    for i in range(3160):
        csv_write.writerow(['obj_unseen', 
                            lower_triangular(objtrained_objnontrained_alexnet)[i], 
                            lower_triangular(objtrained_objnontrained_net)[i]]) 





