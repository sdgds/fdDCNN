#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import copy
import scipy.stats as stats
import torch
import torchvision   
import torchvision.transforms as transforms
import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import io as dnn_io

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
    
    # Normal or Flip
    number_of_picture = pic_number              
    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Prepare_flip_with_normalize'])
    picimg = picdataset[number_of_picture][0].unsqueeze(0)
    picimg.requires_grad=True
    
    
    # load model
    non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
    net = copy.deepcopy(non_face_net).cpu().eval()
    net_truncate = NET(net, selected_layer=11)    
    origin_alexnet = torchvision.models.alexnet(pretrained=True)
    net_alexnet = copy.deepcopy(origin_alexnet).eval()
    alexnet_truncate = NET(net_alexnet, selected_layer=11)        
    
    ## Alexnet
    # model out and class
    face_channel_alexnet = [184, 124]
    A = []
    for n,channel in enumerate(face_channel_alexnet):
        alexnet_truncate.layeract(picimg)
        alexnet_truncate.feature_map[0,channel,:]
        A.append(torch.mean(alexnet_truncate.feature_map[0,channel,:]).item())
    print(A)
    CN.append(sum(A)/len(A))

    ## Non-face alexnet
    # model out and class
    face_channel = [28, 49]
    a = []
    for n,channel in enumerate(face_channel):
        net_truncate.layeract(picimg)
        net_truncate.feature_map[0,channel,:]
        a.append(torch.mean(net_truncate.feature_map[0,channel,:]).item())
    print(a)
    DN.append(sum(a)/len(a))
    

# Normal
picture_number = [589,552,173,175,402,5,394,21,609,108,429,624,640,46,223,251,327,231,674,663]      # frontor stim
CN = []
DN = []
for pic in picture_number:
    Gragh(pic)
CN_normal = CN
DN_normal = DN
    
# Flip
picture_number = [589,552,173,175,402,5,394,21,609,108,429,624,640,46,223,251,327,231,674,663]      # frontor stim
CN = []
DN = []
for pic in picture_number:
    Gragh(pic)    
CN_flip = CN
DN_flip = DN    
    
    
print(stats.ttest_rel(CN_normal,CN_flip))
print(stats.ttest_rel(DN_normal,DN_flip))

