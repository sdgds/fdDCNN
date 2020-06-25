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



#f = os.listdir('/home/dell/Desktop/Visulization/Filter_mid/Filter_mid')
#with open('/home/dell/Desktop/Visulization/Filter_mid/Filter_mid.csv', 'a+') as csvfile:
#    csvfile_write = csv.writer(csvfile)
#    csvfile_write.writerow(['/home/dell/Desktop/Visulization/Filter_mid/Filter_mid/'])
#    csvfile_write.writerow(['stimID', 'condition'])
#    for i in f:
#        csvfile_write.writerow([i, 'pic'])

                    


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
        
def total_random(array):
    random_base = list(range(array.shape[0]*array.shape[1]))
    random.shuffle(random_base)
    result = []
    for i in range(array.shape[0]):
        result_temp = []
        for j in range(array.shape[1]):
            a = random_base[j + i * array.shape[1]] % array.shape[0]
            b = int(random_base[j + i * array.shape[1]] / array.shape[0])
            result_temp.append(array[a, b])
        result.append(result_temp)
    return np.array(result)

def Filter(img, low_freq, high_freq, freq_scale_factor):
    rows, cols = img.shape
    freq_domain = np.fft.fft2(img)
    freq_domain_shifted = np.fft.fftshift(freq_domain)
    freq_pass_window = np.zeros((rows, cols))
    freq_pass_window_center_x = int(rows/2)
    freq_pass_window_center_y = int(cols/2)
    if high_freq:
        high_freq
    else:
        high_freq = 1000
    for x in range(cols):
        for y in range(rows):
            if np.sqrt((x-freq_pass_window_center_x)**2+(y-freq_pass_window_center_y)**2/freq_scale_factor**2) <= high_freq:
                if low_freq <= np.sqrt((x-freq_pass_window_center_x)**2+(y-freq_pass_window_center_y)**2/freq_scale_factor**2):
                    freq_pass_window[y,x] = 1
    windowed_freq_domain_shifted = freq_domain_shifted*freq_pass_window
    adjusted_freq_domain = np.fft.ifftshift(windowed_freq_domain_shifted)
    return np.abs(np.fft.ifft2(adjusted_freq_domain)).real


def picture_contrast(picimg, contrast):
    temp = picimg
    img = transforms.ToPILImage()(temp.permute(2,0,1))  
    enh_con = ImageEnhance.Contrast(img)  
    contrast = 2 
    img_contrasted = enh_con.enhance(contrast) 
    return np.array(img_contrasted)
    
    
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
    # Nature_stimili
    number_of_picture = pic_number 
    picdataset = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_face_1000_FITW.csv', transform=data_transforms['Prepare'])
    picimg = picdataset[number_of_picture][0].unsqueeze(0)
    fig = plt.figure(figsize=(6,6), dpi=200) 
    ax = fig.add_subplot(1,7,1)
    ax.imshow(picimg[0].permute(1,2,0).data)
    ax.axis('off')
    picdataset = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_face_1000_FITW.csv', transform=data_transforms['Prepare_with_normalize'])
    picimg = picdataset[number_of_picture][0].unsqueeze(0)
    picimg.requires_grad=True
    
#    # Normal or Flip
#    number_of_picture = pic_number              
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Prepare_flip'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0) 
#    fig = plt.figure(figsize=(6,6), dpi=200) 
#    ax = fig.add_subplot(1,7,1)
#    ax.imshow(picimg[0].permute(1,2,0).data)
#    ax.axis('off')
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Prepare_flip_with_normalize'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0)
#    picimg.requires_grad=True
    
#    # Eye Mask
#    number_of_picture = pic_number              
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Prepare'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0).data.numpy()
#    eye = picimg[0,:,100:130,30:180]
#    picimg[0,0,100:130,30:180] = total_random(eye[0,:]) 
#    picimg[0,1,100:130,30:180] = total_random(eye[1,:]) 
#    picimg[0,2,100:130,30:180] = total_random(eye[2,:])     
#    fig = plt.figure(figsize=(6,6), dpi=200) 
#    ax = fig.add_subplot(1,7,1)
#    ax.imshow(picimg[0].transpose([1,2,0]))
#    ax.axis('off')
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Prepare_with_normalize'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0).data.numpy()
#    eye = picimg[0,:,100:130,30:180]
#    picimg[0,0,100:130,30:180] = total_random(eye[0,:]) 
#    picimg[0,1,100:130,30:180] = total_random(eye[1,:]) 
#    picimg[0,2,100:130,30:180] = total_random(eye[2,:]) 
#    picimg = torch.Tensor(picimg)
#    picimg.requires_grad=True
    
#    # Mouth Mask
#    number_of_picture = pic_number              
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Prepare'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0).data.numpy()
#    mouth = picimg[0,:,185:215,30:180]
#    picimg[0,0,185:215,30:180] = total_random(mouth[0,:]) 
#    picimg[0,1,185:215,30:180] = total_random(mouth[1,:]) 
#    picimg[0,2,185:215,30:180] = total_random(mouth[2,:])     
#    fig = plt.figure(figsize=(6,6), dpi=200) 
#    ax = fig.add_subplot(1,7,1)
#    ax.imshow(picimg[0].transpose([1,2,0]))
#    ax.axis('off')
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Prepare_with_normalize'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0).data.numpy()
#    mouth = picimg[0,:,185:215,30:180]
#    picimg[0,0,185:215,30:180] = total_random(mouth[0,:]) 
#    picimg[0,1,185:215,30:180] = total_random(mouth[1,:]) 
#    picimg[0,2,185:215,30:180] = total_random(mouth[2,:]) 
#    picimg = torch.Tensor(picimg)
#    picimg.requires_grad=True
   
#    # Filter
#    low_freq = 15
#    high_freq = None
#    
#    number_of_picture = pic_number              
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Visulization/Filter_high/Filter_high.csv', transform=data_transforms['Prepare'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0)   
#    img1 = picimg[0,0,:].data.numpy()
#    img2 = picimg[0,1,:].data.numpy()
#    img3 = picimg[0,2,:].data.numpy()
#    img_back1 = Filter(img1, low_freq, high_freq, 1)
#    img_back2 = Filter(img2, low_freq, high_freq, 1)
#    img_back3 = Filter(img3, low_freq, high_freq, 1)
#    img_back = torch.Tensor([[img_back1,img_back2,img_back3]])
#    picimg = copy.deepcopy(img_back)
#    fig = plt.figure(figsize=(6,6), dpi=200) 
#    ax = fig.add_subplot(1,7,1)
#    ax.imshow(picimg[0].permute(1,2,0).data)
#    ax.axis('off')
#    number_of_picture = pic_number              
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Visulization/Filter_high/Filter_high.csv', transform=data_transforms['Prepare_with_normalize'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0)   
#    img1 = picimg[0,0,:].data.numpy()
#    img2 = picimg[0,1,:].data.numpy()
#    img3 = picimg[0,2,:].data.numpy()
#    img_back1 = Filter(img1, low_freq, high_freq, 1)
#    img_back2 = Filter(img2, low_freq, high_freq, 1)
#    img_back3 = Filter(img3, low_freq, high_freq, 1)
#    img_back = torch.Tensor([[img_back1,img_back2,img_back3]])
#    picimg = copy.deepcopy(img_back)
        
    
    # load model
    non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
    net = copy.deepcopy(non_face_net).cpu().eval()
    net_truncate = NET(net, selected_layer=11)    
    origin_alexnet = torchvision.models.alexnet(pretrained=True)
    net_alexnet = copy.deepcopy(origin_alexnet).eval()
    alexnet_truncate = NET(net_alexnet, selected_layer=11)    
    # hypoparameters
    target_layer = 11
    target_class = 0
    
    with open('/home/dell/Desktop/Visulization/test.csv', 'a+') as csvfile:
        csv_write = csv.writer(csvfile)
        csv_write.writerow(['Picture', 'channel', 'max_value', 'mean_value', 'value_number'])
        ## Alexnet
        # model out and class
        face_channel_alexnet = [184, 124, 59, 186]
        for n,channel in enumerate(face_channel_alexnet):
            alexnet_truncate.layeract(picimg)
            alexnet_truncate.feature_map[0,channel,:]
            print('Picture %d' %pic_number,
                  'Alexnet channel:%d' %channel,
                  'max_value:', torch.max(alexnet_truncate.feature_map[0,channel,:]).item(),
                  'mean_value:', torch.mean(alexnet_truncate.feature_map[0,channel,:]).item(),
                  'value_number:', torch.where(alexnet_truncate.feature_map[0,channel,:]!=0)[0].shape[0])
            csv_write.writerow([pic_number, 
                                channel,
                                torch.max(alexnet_truncate.feature_map[0,channel,:]).item(),
                                torch.mean(alexnet_truncate.feature_map[0,channel,:]).item(),
                                torch.where(alexnet_truncate.feature_map[0,channel,:]!=0)[0].shape[0]])
            
            # Guided backprop
            out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, picimg, target_layer, channel)
            
            # Grad cam
            gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
            cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
            
            # Guided Grad cam
            cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
            ax = fig.add_subplot(1,7,n+2)
            cax = ax.matshow(1*out_image.data.numpy(), cmap='jet',vmin=0, vmax=255)     # Deconvolution
            #cax = ax.matshow(gcv2.cam_channel, cmap='jet',vmin=0, vmax=255, alpha=1)  # Heat map
            ax.axis('off')
        
        ## Non-face alexnet
        # model out and class
        face_channel = [28, 49]
        for n,channel in enumerate(face_channel):
            net_truncate.layeract(picimg)
            net_truncate.feature_map[0,channel,:]
            print('Picture %d' %pic_number,
                  'Net channel:%d' %channel,
                  'max_value:', torch.max(net_truncate.feature_map[0,channel,:]).item(),
                  'mean_value:', torch.mean(net_truncate.feature_map[0,channel,:]).item(),
                  'value_number:', torch.where(net_truncate.feature_map[0,channel,:]!=0)[0].shape[0])
            csv_write.writerow([pic_number, 
                                channel,
                                torch.max(net_truncate.feature_map[0,channel,:]).item(),
                                torch.mean(net_truncate.feature_map[0,channel,:]).item(),
                                torch.where(net_truncate.feature_map[0,channel,:]!=0)[0].shape[0]])
            
            # Guided backprop
            out_image = reconstruction_layer.layer_channel_reconstruction(net.cpu(), picimg, target_layer, channel)
            
            # Grad cam
            gcv2 = Guided_CAM.GradCam(net.cpu(), target_layer=target_layer)
            cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
            
            # Guided Grad cam
            cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
            ax = fig.add_subplot(1,7,n+6)
            cax = ax.matshow(1*out_image.data.numpy(), cmap='jet',vmin=0, vmax=255)     # Deconvolution
            #cax = ax.matshow(gcv2.cam_channel, cmap='jet',vmin=0, vmax=255, alpha=1)  # Heat map
            ax.axis('off')           
        #fig.colorbar(cax)
    fig.savefig('/home/dell/Desktop/Visulization/Results_pictures/%d_deconvolution.jpg' %pic_number)
    #fig.savefig('/home/dell/Desktop/Visulization/Results_pictures/%d_heatmap.jpg' %pic_number)


#picture_number = [554,965,884,910,763,529,620,454,498,382,623,142,22,106,330,540,155,930,643,387]     # nature stim
#picture_number = [589,552,173,175,402,5,394,21,609,108,429,624,640,46,223,251,327,231,674,663]      # frontor stim
picture_number = [22, 142, 498, 529, 554]
for pic in picture_number:
    Gragh(pic)
    
    
    
    
    
    
    
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
#    fig = plt.figure(figsize=(20,4)) 
#    fig.subplots_adjust(hspace=0.1, wspace=0)
#    gs = GridSpec(2,17)
#    ax = fig.add_subplot(gs[0:2,0:4])
#    ax.imshow(picimg[0].permute(1,2,0).data)
#    ax.axis('off')
#    picdataset = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_face_1000_FITW.csv', transform=data_transforms['Prepare_with_normalize'])
#    picimg = picdataset[number_of_picture][0].unsqueeze(0)
#    picimg.requires_grad=True
    
    # Frontal or Flip
    number_of_picture = pic_number              
    picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Prepare_flip'])
    picimg = picdataset[number_of_picture][0].unsqueeze(0) 
    fig = plt.figure(figsize=(20,4)) 
    fig.subplots_adjust(hspace=0.1, wspace=0)
    gs = GridSpec(2,17)
    ax = fig.add_subplot(gs[0:2,0:4]) 
    ax.imshow(picimg[0].permute(1,2,0).data)
    ax.axis('off')
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
    # hypoparameters
    target_layer = 11
    target_class = 0
    
    ## Deconvolution
    # Alexnet
    face_channel_alexnet = [184, 124, 59, 186]
    for n,channel in enumerate(face_channel_alexnet):
        alexnet_truncate.layeract(picimg)
        alexnet_truncate.feature_map[0,channel,:]       
        # Guided backprop
        out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, picimg, target_layer, channel)       
        # Grad cam
        gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
        cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)       
        # Guided Grad cam
        cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
        ax = fig.add_subplot(gs[0,4+2*n:4+2*n+2])
        cax = ax.matshow(picture_contrast(out_image, 2), cmap='jet',vmin=0, vmax=255)     # Deconvolution
        ax.axis('off')
        
    # Non-face alexnet
    face_channel = [28, 49]
    for n,channel in enumerate(face_channel):
        net_truncate.layeract(picimg)
        net_truncate.feature_map[0,channel,:]      
        # Guided backprop
        out_image = reconstruction_layer.layer_channel_reconstruction(net.cpu(), picimg, target_layer, channel)     
        # Grad cam
        gcv2 = Guided_CAM.GradCam(net.cpu(), target_layer=target_layer)
        cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)      
        # Guided Grad cam
        cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
        ax = fig.add_subplot(gs[0,4+2*n+9:4+2*n+2+9])
        cax = ax.matshow(picture_contrast(out_image, 2), cmap='jet',vmin=0, vmax=255)     # Deconvolution
        ax.axis('off')  
        
        
    ## feature map
    # Alexnet
    face_channel_alexnet = [184, 124, 59, 186]
    for n,channel in enumerate(face_channel_alexnet):
        alexnet_truncate.layeract(picimg)
        alexnet_truncate.feature_map[0,channel,:]      
        ax = fig.add_subplot(gs[1,4+2*n:4+2*n+2])
        cax = ax.matshow(alexnet_truncate.feature_map[0,channel,:].data.numpy(), cmap='jet',vmin=0, vmax=25, alpha=1)  # Heat map
        ax.axis('off')
        
    # Non-face alexnet
    face_channel = [28, 49]
    for n,channel in enumerate(face_channel):
        net_truncate.layeract(picimg)
        ax = fig.add_subplot(gs[1,4+2*n+9:4+2*n+2+9])
        cax = ax.matshow(net_truncate.feature_map[0,channel,:].data.numpy(), cmap='jet',vmin=0, vmax=25, alpha=1)  # Heat map
        ax.axis('off')           



#picture_number = [22, 142, 498, 529, 554]     # nature stim
picture_number = [46, 223, 402, 429, 552]     # normal or flip stim
for pic in picture_number:
    Gragh(pic)





'2. Bandpass SF'
###############################################################################
###############################################################################
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
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random
import PIL.Image as Image
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

   
     
def Gragh(picdir, pic, savedir, savecsv):        
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
                                 std = [0.229, 0.224, 0.225])])
        }   
    # Filter   
    img = Image.open(picdir)
    picimg = data_transforms['Prepare'](img).unsqueeze(0)   
    fig = plt.figure(figsize=(6,6), dpi=200) 
    ax = fig.add_subplot(1,7,1)
    ax.imshow(picimg[0].permute(1,2,0).data)
    ax.axis('off')
    img = Image.open(picdir)
    picimg = data_transforms['Prepare_with_normalize'](img).unsqueeze(0)   
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
    
    with open(savecsv, 'a+') as csvfile:
        csv_write = csv.writer(csvfile)
        ## Alexnet
        # model out and class
        face_channel_alexnet = [184, 124, 59, 186]
        for n,channel in enumerate(face_channel_alexnet):
            alexnet_truncate.layeract(picimg)
            alexnet_truncate.feature_map[0,channel,:]
            print('Picture:', pic[:-4],
                  'Alexnet channel:%d' %channel,
                  'max_value:', torch.max(alexnet_truncate.feature_map[0,channel,:]).item(),
                  'mean_value:', torch.mean(alexnet_truncate.feature_map[0,channel,:]).item(),
                  'value_number:', torch.where(alexnet_truncate.feature_map[0,channel,:]!=0)[0].shape[0])
            csv_write.writerow([pic[:-4], 
                                channel,
                                torch.max(alexnet_truncate.feature_map[0,channel,:]).item(),
                                torch.mean(alexnet_truncate.feature_map[0,channel,:]).item(),
                                torch.where(alexnet_truncate.feature_map[0,channel,:]!=0)[0].shape[0]])            
            # Guided backprop
            out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, picimg, target_layer, channel)
            
            # Grad cam
            gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
            cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
            
            # Guided Grad cam
            cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
            ax = fig.add_subplot(1,7,n+2)
            cax = ax.matshow(gcv2.cam_channel, cmap='jet',vmin=0, vmax=255, alpha=1)  # Heat map
            ax.axis('off')
        
        ## Non-face alexnet
        # model out and class
        face_channel = [28, 49]
        for n,channel in enumerate(face_channel):
            net_truncate.layeract(picimg)
            net_truncate.feature_map[0,channel,:]
            print('Picture:', pic[:-4],
                  'Net channel:%d' %channel,
                  'max_value:', torch.max(net_truncate.feature_map[0,channel,:]).item(),
                  'mean_value:', torch.mean(net_truncate.feature_map[0,channel,:]).item(),
                  'value_number:', torch.where(net_truncate.feature_map[0,channel,:]!=0)[0].shape[0])
            csv_write.writerow([pic[:-4], 
                                channel,
                                torch.max(net_truncate.feature_map[0,channel,:]).item(),
                                torch.mean(net_truncate.feature_map[0,channel,:]).item(),
                                torch.where(net_truncate.feature_map[0,channel,:]!=0)[0].shape[0]])           
            # Guided backprop
            out_image = reconstruction_layer.layer_channel_reconstruction(net.cpu(), picimg, target_layer, channel)
            
            # Grad cam
            gcv2 = Guided_CAM.GradCam(net.cpu(), target_layer=target_layer)
            cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
            
            # Guided Grad cam
            cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
            ax = fig.add_subplot(1,7,n+6)
            cax = ax.matshow(gcv2.cam_channel, cmap='jet',vmin=0, vmax=255, alpha=1)  # Heat map
            ax.axis('off')           
        #fig.colorbar(cax)
    fig.savefig(savedir + pic[:-4] + '_heatmap.jpg') 
    
    
    # Filter   
    img = Image.open(picdir)
    picimg = data_transforms['Prepare'](img).unsqueeze(0)   
    fig = plt.figure(figsize=(6,6), dpi=200) 
    ax = fig.add_subplot(1,7,1)
    ax.imshow(picimg[0].permute(1,2,0).data)
    ax.axis('off')
    img = Image.open(picdir)
    picimg = data_transforms['Prepare_with_normalize'](img).unsqueeze(0)   
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
    face_channel_alexnet = [184, 124, 59, 186]
    for n,channel in enumerate(face_channel_alexnet):
        alexnet_truncate.layeract(picimg)
        alexnet_truncate.feature_map[0,channel,:]
        
        # Guided backprop
        out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, picimg, target_layer, channel)
        
        # Grad cam
        gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
        cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
        
        # Guided Grad cam
        cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
        ax = fig.add_subplot(1,7,n+2)
        cax = ax.matshow(1*out_image.data.numpy(), cmap='jet',vmin=0, vmax=255)     # Deconvolution
        ax.axis('off')
    
    ## Non-face alexnet
    # model out and class
    face_channel = [28, 49]
    for n,channel in enumerate(face_channel):
        net_truncate.layeract(picimg)
        net_truncate.feature_map[0,channel,:]
  
        # Guided backprop
        out_image = reconstruction_layer.layer_channel_reconstruction(net.cpu(), picimg, target_layer, channel)
        
        # Grad cam
        gcv2 = Guided_CAM.GradCam(net.cpu(), target_layer=target_layer)
        cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
        
        # Guided Grad cam
        cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
        ax = fig.add_subplot(1,7,n+6)
        cax = ax.matshow(1*out_image.data.numpy(), cmap='jet',vmin=0, vmax=255)     # Deconvolution
        ax.axis('off')           
        #fig.colorbar(cax)
    fig.savefig(savedir + pic[:-4] + '_deconvolution.jpg')
    

picture_number = [589,552,173,175,402,5,394,21,609,108,429,624,640,46,223,251,327,231,674,663]
with open('/home/dell/Desktop/Visulization/Face_filter/Filter_rescale/Highpass/Highpass.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    csv_write.writerow(['Picture', 'channel', 'max_value', 'mean_value', 'value_number'])
    csvfile.close()
f = os.listdir('/home/dell/Desktop/Visulization/high_res2')
for pic in f:
    picdir = '/home/dell/Desktop/Visulization/high_res2/' + pic
    savedir = '/home/dell/Desktop/Visulization/Face_filter/Filter_rescale/Highpass/'
    savecsv = '/home/dell/Desktop/Visulization/Face_filter/Filter_rescale/Highpass/Highpass.csv'
    Gragh(picdir, pic, savedir, savecsv)
    
    
    
    
    
    
    
    
    
data_transforms = {
    'unNormalize': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]),
    'Normalize': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
    }
number_of_picture = 300               
picdataset = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/standard_face.csv', transform=data_transforms['Normalize'])
picimg = picdataset[number_of_picture][0].unsqueeze(0)
picimg.requires_grad=True
plt.figure()
plt.imshow(picimg[0].permute(1,2,0).data.numpy())
plt.axis('off')


# load model
non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
net = copy.deepcopy(non_face_net).cpu()
origin_alexnet = torchvision.models.alexnet(pretrained=True)
net_alexnet = copy.deepcopy(origin_alexnet)


# fft    
def Filter(img, low_freq, high_freq, freq_scale_factor):
    rows, cols = img.shape
    freq_domain = np.fft.fft2(img)
    freq_domain_shifted = np.fft.fftshift(freq_domain)
    freq_pass_window = np.zeros((rows, cols))
    freq_pass_window_center_x = int(rows/2)
    freq_pass_window_center_y = int(cols/2)
    if high_freq:
        high_freq
    else:
        high_freq = 10000
    for x in range(cols):
        for y in range(rows):
            if np.sqrt((x-freq_pass_window_center_x)**2+(y-freq_pass_window_center_y)**2/freq_scale_factor**2) <= high_freq:
                if low_freq <= np.sqrt((x-freq_pass_window_center_x)**2+(y-freq_pass_window_center_y)**2/freq_scale_factor**2):
                    freq_pass_window[y,x] = 1
    windowed_freq_domain_shifted = freq_domain_shifted*freq_pass_window
    adjusted_freq_domain = np.fft.ifftshift(windowed_freq_domain_shifted)
    return np.fft.ifft2(adjusted_freq_domain).real


def Filter(img, low_freq, high_freq, freq_scale_factor):
    rows, cols = img.shape
    freq_domain = np.fft.fft2(img)
    freq_domain_shifted = np.fft.fftshift(freq_domain)
    freq_pass_window = np.ones((rows, cols))
    freq_pass_window_center_x = int(rows/2)
    freq_pass_window_center_y = int(cols/2)  
    gauss_filter = []
    for x in range(cols):
        for y in range(rows):
            gauss_filter.append(1-np.exp(-4*np.log(2)*(x-freq_pass_window_center_x)**2/(2*low_freq)**2) * np.exp(-4*np.log(2)*(y-freq_pass_window_center_y)**2/(2*low_freq)**2/freq_scale_factor**2))
    gauss_filter = np.array(gauss_filter).reshape(224, 224)
    freq_pass_window = freq_pass_window * gauss_filter
    windowed_freq_domain_shifted = freq_domain_shifted * freq_pass_window
    adjusted_freq_domain = np.fft.ifftshift(windowed_freq_domain_shifted)
    return np.abs(np.fft.ifft2(adjusted_freq_domain)).real

img1 = picimg[0,0,:].data.numpy()
img2 = picimg[0,1,:].data.numpy()
img3 = picimg[0,2,:].data.numpy()

low_freq = 15
high_freq = None
img_back1 = Filter(img1, low_freq, high_freq, 1)
img_back2 = Filter(img2, low_freq, high_freq, 1)
img_back3 = Filter(img3, low_freq, high_freq, 1)
img_back = torch.Tensor([[img_back1,img_back2,img_back3]])
plt.figure()
plt.imshow(img_back[0].permute(1,2,0).data.numpy())
plt.axis('off')

    

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
alexnet_truncate = NET(net_alexnet, selected_layer=selected_layer)
net_truncate = NET(net, selected_layer=selected_layer)

alexnet_truncate.layeract(picimg)
Conv5_alexnet = alexnet_truncate.feature_map
net_truncate.layeract(picimg)
Conv5_net = net_truncate.feature_map
fig = plt.figure(figsize=(15,15))
ax = fig.subplots(1,6)
ax[0].imshow(Conv5_alexnet[0, 124, :].data)
ax[1].imshow(Conv5_alexnet[0, 184, :].data)
ax[2].imshow(Conv5_alexnet[0, 59, :].data)
ax[3].imshow(Conv5_alexnet[0, 186, :].data)
ax[4].imshow(Conv5_net[0, 28, :].data)
ax[5].imshow(Conv5_net[0, 49, :].data)

alexnet_truncate.layeract(img_back)
Conv5_alexnet = alexnet_truncate.feature_map
net_truncate.layeract(img_back)
Conv5_net = net_truncate.feature_map
fig = plt.figure(figsize=(15,15))
ax = fig.subplots(1,6)
ax[0].imshow(Conv5_alexnet[0, 124, :].data)
ax[1].imshow(Conv5_alexnet[0, 184, :].data)
ax[2].imshow(Conv5_alexnet[0, 59, :].data)
ax[3].imshow(Conv5_alexnet[0, 186, :].data)
ax[4].imshow(Conv5_net[0, 28, :].data)
ax[5].imshow(Conv5_net[0, 49, :].data)
        


import sys
sys.path.append('/home/dell/Desktop/My_module')
import reconstruction_layer
import Guided_CAM

p_alexnet = plt.figure(figsize=(12,12) ,dpi=80) 
for n,channel in enumerate([124, 184, 59, 186]):
    # Guided backprop
    out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, picimg, target_layer, channel)
    #plt.imshow(out_image)
    
    # Grad cam
    gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
    cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
    #plt.imshow(gcv2.cam_channel)
    
    # Guided Grad cam
    cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
    ax_alexnet = p_alexnet.add_subplot(1,4,n+1)
    plt.imshow(Guided_CAM.convert_to_grayscale(cam_gb, threshold=99)[0,:])
    plt.axis('off')
    
    
p_net = plt.figure(figsize=(12,12) ,dpi=80) 
for n,channel in enumerate([28, 49]):
    # Guided backprop
    out_image = reconstruction_layer.layer_channel_reconstruction(net.cpu(), picimg, target_layer, channel)
    #plt.imshow(out_image)
    
    # Grad cam
    gcv2 = Guided_CAM.GradCam(net.cpu(), target_layer=target_layer)
    cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
    #plt.imshow(gcv2.cam_channel)
    
    # Guided Grad cam
    cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
    ax_net = p_net.add_subplot(1,4,n+1)
    plt.imshow(Guided_CAM.convert_to_grayscale(cam_gb, threshold=99)[0,:])
    plt.axis('off')
    
        
p_alexnet = plt.figure(figsize=(12,12) ,dpi=80) 
for n,channel in enumerate([124, 184, 59, 186]):
    # Guided backprop
    out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, img_back, target_layer, channel)
    #plt.imshow(out_image)
    
    # Grad cam
    gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
    cam, weights = gcv2.generate_cam(img_back, target_class=target_class, channel=channel)
    #plt.imshow(gcv2.cam_channel)
    
    # Guided Grad cam
    cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
    ax_alexnet = p_alexnet.add_subplot(1,4,n+1)
    plt.imshow(Guided_CAM.convert_to_grayscale(cam_gb, threshold=99)[0,:])
    plt.axis('off')
    
    
p_net = plt.figure(figsize=(12,12) ,dpi=80) 
for n,channel in enumerate([28, 49]):
    # Guided backprop
    out_image = reconstruction_layer.layer_channel_reconstruction(net.cpu(), img_back, target_layer, channel)
    #plt.imshow(out_image)
    
    # Grad cam
    gcv2 = Guided_CAM.GradCam(net.cpu(), target_layer=target_layer)
    cam, weights = gcv2.generate_cam(img_back, target_class=target_class, channel=channel)
    #plt.imshow(gcv2.cam_channel)
    
    # Guided Grad cam
    cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
    ax_net = p_net.add_subplot(1,4,n+1)
    plt.imshow(Guided_CAM.convert_to_grayscale(cam_gb, threshold=99)[0,:])
    plt.axis('off')
        
        
        

'3. Composite effect'
###############################################################################
###############################################################################
import PIL.Image as Image

data_transforms = {
    'unNormalize': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]),
    'Normalize': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
    }
    
for i in ['09','17','10','18','11','19','12','20','13','21','14','22','15','23','16','24']:
    number_of_picture = i
    picdir = '/home/dell/Desktop/DNN2Brain/FaceA/' + number_of_picture +'.jpg'             
    picimg = data_transforms['unNormalize'](Image.open(picdir).convert('RGB')).unsqueeze(0)
    picimg.requires_grad=True
    plt.figure()
    plt.imshow(picimg[0].permute(1,2,0).data.numpy())
    plt.axis('off')
    
    # load model
    non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
    net = copy.deepcopy(non_face_net).cpu()
    origin_alexnet = torchvision.models.alexnet(pretrained=True)
    net_alexnet = copy.deepcopy(origin_alexnet)
    
    p_alexnet = plt.figure(figsize=(12,12) ,dpi=80) 
    for n,channel in enumerate([124, 184, 59, 186]):
        # Guided backprop
        out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, picimg, target_layer, channel)
        #plt.imshow(out_image)
        
        # Grad cam
        gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
        cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
        #plt.imshow(gcv2.cam_channel)
        
        # Guided Grad cam
        cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
        ax_alexnet = p_alexnet.add_subplot(1,4,n+1)
        plt.imshow(Guided_CAM.convert_to_grayscale(cam_gb, threshold=99)[0,:])
        plt.axis('off')
        
        
    p_net = plt.figure(figsize=(12,12) ,dpi=80) 
    for n,channel in enumerate([28, 49]):
        # Guided backprop
        out_image = reconstruction_layer.layer_channel_reconstruction(net.cpu(), picimg, target_layer, channel)
        #plt.imshow(out_image)
        
        # Grad cam
        gcv2 = Guided_CAM.GradCam(net.cpu(), target_layer=target_layer)
        cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
        #plt.imshow(gcv2.cam_channel)
        
        # Guided Grad cam
        cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
        ax_net = p_net.add_subplot(1,4,n+1)
        plt.imshow(Guided_CAM.convert_to_grayscale(cam_gb, threshold=99)[0,:])
        plt.axis('off')
    

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
alexnet_truncate = NET(net_alexnet, selected_layer=selected_layer)
net_truncate = NET(net, selected_layer=selected_layer)

alexnet_truncate.layeract(picimg)
Conv5_alexnet = alexnet_truncate.feature_map
net_truncate.layeract(picimg)
Conv5_net = net_truncate.feature_map

fig = plt.figure(figsize=(15,15))
ax = fig.subplots(1,6)
ax[0].imshow(Conv5_alexnet[0, 124, :].data)
ax[1].imshow(Conv5_alexnet[0, 184, :].data)
ax[2].imshow(Conv5_alexnet[0, 59, :].data)
ax[3].imshow(Conv5_alexnet[0, 186, :].data)
ax[4].imshow(Conv5_net[0, 28, :].data)
ax[5].imshow(Conv5_net[0, 49, :].data)

alexnet_truncate.layeract(picimg)
Conv5_alexnet = alexnet_truncate.conv_output
net_truncate.layeract(picimg)
Conv5_net = net_truncate.conv_output





'4. Effective receptive field'
###############################################################################
###############################################################################
import torch.nn as nn
from tqdm import tqdm
import copy
import os
import PIL.Image as Image
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt  
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader     
import torchvision.transforms as transforms

#net = torch.load('/nfs/s2/userhome/zhangyiyuan/workingdir/beifen/models/Nonface_alexnet/model_epoch89', map_location=torch.device('cpu'))
#net.eval()
#alexnet = torchvision.models.alexnet()
#alexnet.load_state_dict(torch.load('/nfs/s2/dnnbrain_data/models/alexnet_param.pth'))
#alexnet.eval()

net = torch.load('/Users/mac/Desktop/Face/model_epoch89', map_location=torch.device('cpu'))
alexnet = torchvision.models.alexnet(pretrained=True)

# [59, 124, 184, 186]
unit_row, unit_col = (6,6)
p_alexnet = plt.figure(figsize=(12,12) ,dpi=80) 
for n,channel in enumerate([59, 124, 184, 186]):
    grad_alexnet_avg = []
    for trail in tqdm(range(10000)):
        x_alexnet = torch.Tensor(np.random.uniform(0,255,(1, 3, 224, 224)))
        x_alexnet.requires_grad=True
        Conv5_out_alexnet = alexnet.features[:11](x_alexnet)
        target_unit_alexnet = Conv5_out_alexnet[0,channel,unit_row,unit_col]
        target_unit_alexnet.backward()
        grad_alexnet = x_alexnet.grad[0].permute(1,2,0).data.numpy()
        grad_alexnet_avg.append(grad_alexnet)
    grad_alexnet_avg = np.array(grad_alexnet_avg)
    grad_alexnet_avg = np.mean(grad_alexnet_avg, axis=0)
    #save_dir = '/nfs/s2/userhome/zhangyiyuan/Desktop/ERF/grad_alexnet_avg_' + str(channel)
    save_dir = '/Users/mac/Desktop/Face/grad_alexnet_avg_' + str(channel)
    np.save(save_dir, grad_alexnet_avg)
    print(np.where(grad_alexnet_avg>0.0)[0].shape[0], np.sum(np.where(grad_alexnet_avg>0.0,grad_alexnet_avg,0)))
    ax_alexnet = p_alexnet.add_subplot(1,4,n+1)
    ax_alexnet.imshow(np.where(grad_alexnet_avg>0.0, grad_alexnet_avg, 0)*100)
    ax_alexnet.axis('off')

# [28, 49]
p_net = plt.figure(figsize=(12,12) ,dpi=80) 
for n,channel in enumerate([28, 49]):
    grad_net_avg = []
    for trail in tqdm(range(10000)):
        x_net = torch.Tensor(np.random.uniform(0,255,(1, 3, 224, 224)))
        x_net.requires_grad=True
        Conv5_out_net = net.features[:11](x_net)
        target_unit_net = Conv5_out_net[0,channel,unit_row,unit_col]
        target_unit_net.backward()
        grad_net = x_net.grad[0].permute(1,2,0).data.numpy()
        grad_net_avg.append(grad_net)
    grad_net_avg = np.array(grad_net_avg)
    grad_net_avg = np.mean(grad_net_avg, axis=0)
    #save_dir = '/nfs/s2/userhome/zhangyiyuan/Desktop/ERF/grad_net_avg_' + str(channel)
    save_dir = '/Users/mac/Desktop/Face/grad_net_avg_' + str(channel)
    np.save(save_dir, grad_net_avg)
    print(np.where(grad_net_avg>0.0)[0].shape[0], np.sum(np.where(grad_net_avg>0.0,grad_net_avg,0)))
    ax_net = p_net.add_subplot(1,4,n+1)
    ax_net.imshow(np.where(grad_net_avg>0.0, grad_net_avg, 0)*100)
    ax_net.axis('off')


A = np.load('/Users/mac/Desktop/Face/grad_alexnet_avg_124.npy')
B = np.load('/Users/mac/Desktop/Face/grad_alexnet_avg_184.npy')
C = np.load('/Users/mac/Desktop/Face/grad_alexnet_avg_186.npy')
D = np.load('/Users/mac/Desktop/Face/grad_alexnet_avg_59.npy')
a = np.load('/Users/mac/Desktop/Face/grad_net_avg_28.npy')
b = np.load('/Users/mac/Desktop/Face/grad_net_avg_49.npy')

def plt_ERF(grad_img):
    img_2D = np.where(np.max(grad_img,axis=2)>np.abs(np.min(grad_img,axis=2)), 
                        np.abs(np.max(grad_img,axis=2)), 
                        np.abs(np.min(grad_img,axis=2)))
    #img_2D = grad_img[:,:,2]
    plt.figure()
    ax = plt.axes(projection='3d')
    xx = np.arange(0,224,1)
    yy = np.arange(0,224,1)
    X, Y = np.meshgrid(xx, yy)
    Z = img_2D
    ax.plot_surface(X,Y,Z,cmap='jet',alpha=0.7)
    #ax.contour(X,Y,Z,zdir='z', offset=img_2D.min())
    ax.contour(X,Y,Z,zdir='x', offset=0)
    ax.contour(X,Y,Z,zdir='y', offset=224)
    plt.show()
    
    plt.figure()
    plt.imshow(np.where(img_2D>0.0005, img_2D, 0), cmap='jet')
    plt.colorbar()
    
    
plt_ERF(A)
plt_ERF(B)
plt_ERF(C)
plt_ERF(D)
plt_ERF(a)
plt_ERF(b)




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
                             std = [0.229, 0.224, 0.225])])}  
    
with open('/Users/mac/Data/caltech256/people_nonpeople_label.csv','r',encoding='utf-8') as f:
    dic = dict()
    for line in f.readlines():
        line=line.strip('\n') 
        b=line.split(',')
        dic[b[0]] = b[1]

net = torch.load('/Users/mac/Desktop/Face/model_epoch89', map_location=torch.device('cpu'))
alexnet = torchvision.models.alexnet(pretrained=True)
f = os.listdir('/Users/mac/Data/caltech256/all_stim')
f = f[1:29780]

# [59, 124, 184, 186]
unit_row, unit_col = (6,6)
p_alexnet = plt.figure(figsize=(12,12) ,dpi=80) 
for n,channel in enumerate([59, 124, 184, 186]):
    grad_alexnet_avg = []
    for pic in tqdm(f):
        if dic[pic] == 'with people':
            img = Image.open('/Users/mac/Data/caltech256/all_stim/'+pic).convert('RGB')
            picimg = data_transforms['Prepare_with_normalize'](img).unsqueeze(0) 
            picimg.requires_grad=True
            Conv5_out_alexnet = alexnet.features[:11](picimg)
            target_unit_alexnet = Conv5_out_alexnet[0,channel,unit_row,unit_col]
            target_unit_alexnet.backward()
            grad_alexnet = picimg.grad[0].permute(1,2,0).data.numpy()
            grad_alexnet_avg.append(grad_alexnet)
    grad_alexnet_avg = np.array(grad_alexnet_avg)
    grad_alexnet_avg = np.mean(grad_alexnet_avg, axis=0)
    save_dir = '/Users/mac/Desktop/Face/face_grad_alexnet_avg_' + str(channel)
    np.save(save_dir, grad_alexnet_avg)
    print(np.where(grad_alexnet_avg>0.0)[0].shape[0], np.sum(np.where(grad_alexnet_avg>0.0,grad_alexnet_avg,0)))
    ax_alexnet = p_alexnet.add_subplot(1,4,n+1)
    ax_alexnet.imshow(np.where(grad_alexnet_avg>0.0, grad_alexnet_avg, 0)*100)
    ax_alexnet.axis('off')

# [28, 49]
p_net = plt.figure(figsize=(12,12) ,dpi=80) 
for n,channel in enumerate([28, 49]):
    grad_net_avg = []
    for pic in tqdm(f):
        if dic[pic] == 'with people':
            img = Image.open('/Users/mac/Data/caltech256/all_stim/'+pic).convert('RGB')
            picimg = data_transforms['Prepare_with_normalize'](img).unsqueeze(0) 
            picimg.requires_grad=True
            Conv5_out_net = net.features[:11](picimg)
            target_unit_net = Conv5_out_net[0,channel,unit_row,unit_col]
            target_unit_net.backward()
            grad_net = picimg.grad[0].permute(1,2,0).data.numpy()
            grad_net_avg.append(grad_net)
    grad_net_avg = np.array(grad_net_avg)
    grad_net_avg = np.mean(grad_net_avg, axis=0)
    save_dir = '/Users/mac/Desktop/Face/face_grad_net_avg_' + str(channel)
    np.save(save_dir, grad_net_avg)
    print(np.where(grad_net_avg>0.0)[0].shape[0], np.sum(np.where(grad_net_avg>0.0,grad_net_avg,0)))
    ax_net = p_net.add_subplot(1,4,n+1)
    ax_net.imshow(np.where(grad_net_avg>0.0, grad_net_avg, 0)*100)
    ax_net.axis('off')


A = np.load('/Users/mac/Desktop/Face/face_grad_alexnet_avg_184.npy')
B = np.load('/Users/mac/Desktop/Face/face_grad_alexnet_avg_124.npy')
C = np.load('/Users/mac/Desktop/Face/face_grad_alexnet_avg_59.npy')
D = np.load('/Users/mac/Desktop/Face/face_grad_alexnet_avg_186.npy')
a = np.load('/Users/mac/Desktop/Face/face_grad_net_avg_28.npy')
b = np.load('/Users/mac/Desktop/Face/face_grad_net_avg_49.npy')

def plt_ERF(grad_img, title):
    img_2D = np.where(np.max(grad_img,axis=2)>np.abs(np.min(grad_img,axis=2)), 
                      np.max(grad_img,axis=2), 
                      np.min(grad_img,axis=2))
    #img_2D = grad_img[:,:,1]
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    xx = np.arange(0,224,1)
    yy = np.arange(0,224,1)
    X, Y = np.meshgrid(xx, yy)
    #Z = img_2D
    Z = (img_2D+0.010557145)/(0.011000458+0.010557145)
    Z = Z*255
    norm = colors.Normalize(0, 255)
    surf = ax.plot_surface(X,Y,Z,cmap='jet',alpha=0.7, norm=norm)
    ax.contour(X,Y,Z,zdir='x', offset=0)
    ax.contour(X,Y,Z,zdir='y', offset=224)
    fig.colorbar(surf)
    
    plt.figure()
    plt.imshow(np.where(Z>150, Z, 0), cmap='jet', norm=norm)
    plt.colorbar()
    
    print(np.where(img_2D>0)[0].shape[0])
    print(img_2D.std())
    
    
plt_ERF(A, 'fDCNN channel 184')
plt_ERF(B, 'fDCNN channel 124')
plt_ERF(a, 'fdDCNN channel 28')
plt_ERF(b, 'fdDCNN channel 49')


for k in [A,B,C,D,a,b]:
    grad_img = k
    img_2D = np.where(np.max(grad_img,axis=2)>np.abs(np.min(grad_img,axis=2)), 
                        np.abs(np.max(grad_img,axis=2)), 
                        np.abs(np.min(grad_img,axis=2)))
    Z = (img_2D+0.010557145)/(0.011000458+0.010557145)
    Z = Z*255
    z = []
    for i in range(100,200):
        z.append(np.where(Z>i, 1, 0).sum())
    plt.plot(z)    
    print(np.percentile(Z,95))
    



'5. Feature Receptive field'
###############################################################################
###############################################################################
f = os.listdir('/home/dell/Desktop/205cat/All_stimuli')
f.sort()

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
                             std = [0.229, 0.224, 0.225])])}   
    

def Feature_RF(model, channel):
    Pic_to_activation = dict()
    Mean_activation_conv5 = []
    for n,pic in enumerate(f):
        print('Now is number of picture is:', n)
        picdir = '/home/dell/Desktop/205cat/All_stimuli/' + pic
        img = Image.open(picdir)
        picimg = data_transforms['Prepare_with_normalize'](img).unsqueeze(0)   
        picimg.requires_grad=True
        
        model.eval()
        model_truncate = NET(model, selected_layer=11)           
        model_truncate.layeract(picimg)
        model_truncate.feature_map[0,channel,:]
        Pic_to_activation[pic] = torch.mean(model_truncate.feature_map[0,channel,:]).item()
        Mean_activation_conv5.append(torch.mean(model_truncate.feature_map).item())
    
    Avg_avtivation_conv5 = sum(Mean_activation_conv5)/len(Mean_activation_conv5)
    
    Feature_stim = np.zeros((224,224,3))
    for pic,value in Pic_to_activation.items():
        if value > Avg_avtivation_conv5:
            picdir = '/home/dell/Desktop/205cat/All_stimuli/' + pic
            img = Image.open(picdir)
            picimg = data_transforms['Prepare_with_normalize'](img)   
            picimg = picimg.permute(1,2,0).data.numpy()
            Feature_stim += picimg * value
        
    return Feature_stim


non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
net = copy.deepcopy(non_face_net).cpu().eval()
origin_alexnet = torchvision.models.alexnet(pretrained=True)
net_alexnet = copy.deepcopy(origin_alexnet).eval()

Feature_stim = Feature_RF(net_alexnet, 184)


'6. Empirical Receptive field'
###############################################################################
###############################################################################
from matplotlib.colors import Normalize
import sys
sys.path.append('/Users/mac/dnnbrain')
from dnnbrain.dnn.algo import EmpiricalReceptiveField, UpsamplingActivationMapping
from dnnbrain.dnn.core import Stimulus
from dnnbrain.dnn.models import AlexNet

dnn = AlexNet()
dnn.model = torchvision.models.alexnet(pretrained=True)
dnn.model.eval()
layer = 'conv5'
for chn in [185,125]:
    up_map = UpsamplingActivationMapping(dnn, layer, chn)
    up_map.set_params(interp_meth='bicubic', interp_threshold=0.9)
    emp_rf = EmpiricalReceptiveField(up_map)
    
    path = '/Users/mac/Desktop/Face/empRF_stim_FITW_1000.stim.csv'
    stim = Stimulus()
    stim.load(path)
    rf_pic = emp_rf.compute(stim)
    plt.figure()
    norm = Normalize(0,15)
    plt.imshow(rf_pic, cmap='gist_stern', norm=norm)
    plt.colorbar()
    plt.axis('off')


dnn = AlexNet()
dnn.model = torch.load('/Users/mac/Models/DN_Alexnet/model_epoch89', map_location=torch.device('cpu'))
layer = 'conv5'
for chn in [29,50]:
    up_map = UpsamplingActivationMapping(dnn, layer, chn)
    up_map.set_params(interp_meth='bicubic', interp_threshold=0.9)
    emp_rf = EmpiricalReceptiveField(up_map)
    
    path = '/Users/mac/Desktop/Face/empRF_stim_FITW_1000.stim.csv'
    stim = Stimulus()
    stim.load(path)
    rf_pic = emp_rf.compute(stim)
    plt.figure()
    norm = Normalize(0,15)
    plt.imshow(rf_pic, cmap='gist_stern', norm=norm)
    plt.colorbar()




'6. Yiselie'
###############################################################################
###############################################################################
import torch.nn as nn
import copy
import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt  
import torchvision.transforms as transforms
import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import io as dnn_io
from dnnbrain.dnn import models
import csv
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL.Image as Image
import sys
sys.path.append('/home/dell/Desktop/My_module')
import reconstruction_layer
import Guided_CAM


# Analyze based on mean of face channel
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

   
     
def Gragh(picdir, pic, savedir, savecsv):        
    # load picture
    data_transforms = {
        'Prepare': transforms.Compose([
            transforms.ToTensor()]),
        'Prepare_with_normalize': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225])])
        }   
    # Filter   
    img = Image.open(picdir)
    picimg = data_transforms['Prepare'](img).unsqueeze(0)   
    fig = plt.figure(figsize=(6,6), dpi=200) 
    ax = fig.add_subplot(1,7,1)
    ax.imshow(picimg[0].permute(1,2,0).data)
    ax.axis('off')
    img = Image.open(picdir)
    picimg = data_transforms['Prepare_with_normalize'](img).unsqueeze(0)   
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
    
    with open(savecsv, 'a+') as csvfile:
        csv_write = csv.writer(csvfile)
        ## Alexnet
        # model out and class
        face_channel_alexnet = [184, 124, 59, 186]
        for n,channel in enumerate(face_channel_alexnet):
            alexnet_truncate.layeract(picimg)
            alexnet_truncate.feature_map[0,channel,:]
            print('Picture:', pic[:-4],
                  'Alexnet channel:%d' %channel,
                  'max_value:', torch.max(alexnet_truncate.feature_map[0,channel,:]).item(),
                  'mean_value:', torch.mean(alexnet_truncate.feature_map[0,channel,:]).item(),
                  'value_number:', torch.where(alexnet_truncate.feature_map[0,channel,:]!=0)[0].shape[0])
            csv_write.writerow([pic[:-4], 
                                channel,
                                torch.max(alexnet_truncate.feature_map[0,channel,:]).item(),
                                torch.mean(alexnet_truncate.feature_map[0,channel,:]).item(),
                                torch.where(alexnet_truncate.feature_map[0,channel,:]!=0)[0].shape[0]])            
            # Guided backprop
            out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, picimg, target_layer, channel)
            
            # Grad cam
            gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
            cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
            
            # Guided Grad cam
            cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
            ax = fig.add_subplot(1,7,n+2)
            cax = ax.matshow(gcv2.cam_channel, cmap='jet',vmin=0, vmax=255, alpha=1)  # Heat map
            ax.axis('off')
        
        ## Non-face alexnet
        # model out and class
        face_channel = [28, 49]
        for n,channel in enumerate(face_channel):
            net_truncate.layeract(picimg)
            net_truncate.feature_map[0,channel,:]
            print('Picture:', pic[:-4],
                  'Net channel:%d' %channel,
                  'max_value:', torch.max(net_truncate.feature_map[0,channel,:]).item(),
                  'mean_value:', torch.mean(net_truncate.feature_map[0,channel,:]).item(),
                  'value_number:', torch.where(net_truncate.feature_map[0,channel,:]!=0)[0].shape[0])
            csv_write.writerow([pic[:-4], 
                                channel,
                                torch.max(net_truncate.feature_map[0,channel,:]).item(),
                                torch.mean(net_truncate.feature_map[0,channel,:]).item(),
                                torch.where(net_truncate.feature_map[0,channel,:]!=0)[0].shape[0]])           
            # Guided backprop
            out_image = reconstruction_layer.layer_channel_reconstruction(net.cpu(), picimg, target_layer, channel)
            
            # Grad cam
            gcv2 = Guided_CAM.GradCam(net.cpu(), target_layer=target_layer)
            cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
            
            # Guided Grad cam
            cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
            ax = fig.add_subplot(1,7,n+6)
            cax = ax.matshow(gcv2.cam_channel, cmap='jet',vmin=0, vmax=255, alpha=1)  # Heat map
            ax.axis('off')           
        #fig.colorbar(cax)
    fig.savefig(savedir + pic[:-4] + '_heatmap.jpg') 
    
    
    # Filter   
    img = Image.open(picdir)
    picimg = data_transforms['Prepare'](img).unsqueeze(0)   
    fig = plt.figure(figsize=(6,6), dpi=200) 
    ax = fig.add_subplot(1,7,1)
    ax.imshow(picimg[0].permute(1,2,0).data)
    ax.axis('off')
    img = Image.open(picdir)
    picimg = data_transforms['Prepare_with_normalize'](img).unsqueeze(0)   
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
    face_channel_alexnet = [184, 124, 59, 186]
    for n,channel in enumerate(face_channel_alexnet):
        alexnet_truncate.layeract(picimg)
        alexnet_truncate.feature_map[0,channel,:]
        
        # Guided backprop
        out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, picimg, target_layer, channel)
        
        # Grad cam
        gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
        cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
        
        # Guided Grad cam
        cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
        ax = fig.add_subplot(1,7,n+2)
        cax = ax.matshow(1*out_image.data.numpy(), cmap='jet',vmin=0, vmax=255)     # Deconvolution
        ax.axis('off')
    
    ## Non-face alexnet
    # model out and class
    face_channel = [28, 49]
    for n,channel in enumerate(face_channel):
        net_truncate.layeract(picimg)
        net_truncate.feature_map[0,channel,:]
  
        # Guided backprop
        out_image = reconstruction_layer.layer_channel_reconstruction(net.cpu(), picimg, target_layer, channel)
        
        # Grad cam
        gcv2 = Guided_CAM.GradCam(net.cpu(), target_layer=target_layer)
        cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
        
        # Guided Grad cam
        cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
        ax = fig.add_subplot(1,7,n+6)
        cax = ax.matshow(1*out_image.data.numpy(), cmap='jet',vmin=0, vmax=255)     # Deconvolution
        ax.axis('off')           
        #fig.colorbar(cax)
    fig.savefig(savedir + pic[:-4] + '_deconvolution.jpg')
    

picdict = '/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/low-ps/'
savecsv = '/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/Results/low-ps/low-ps.csv'
with open(savecsv, 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    csv_write.writerow(['Picture', 'channel', 'max_value', 'mean_value', 'value_number'])
    csvfile.close()
f = os.listdir(picdict)
for pic in f:
    picdir = picdict + pic
    savedir = '/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/Results/low-ps/'
    Gragh(picdir, pic, savedir, savecsv)





### Analyze based on the pattern of face channel
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

def Euch_d(x, y):
    return np.sqrt(np.sum((x - y)**2))
     

def Gragh(picdir):        
    # load picture
    data_transforms = {
        'Prepare': transforms.Compose([
            transforms.ToTensor()]),
        'Prepare_with_normalize': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225])])
        }   
    # Filter   
    img = Image.open(picdir)
    picimg = data_transforms['Prepare_with_normalize'](img).unsqueeze(0)   
    picimg.requires_grad=True
    
    # load model
    non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
    net = copy.deepcopy(non_face_net).cpu().eval()
    net_truncate = NET(net, selected_layer=11)    
    origin_alexnet = torchvision.models.alexnet(pretrained=True)
    net_alexnet = copy.deepcopy(origin_alexnet).eval()
    alexnet_truncate = NET(net_alexnet, selected_layer=11)    
    
    ## Alexnet
    Activation_alexnet = []
    face_channel_alexnet = [184, 124, 59, 186]
    for n,channel in enumerate(face_channel_alexnet):
        alexnet_truncate.layeract(picimg)
        activation_alexnet = alexnet_truncate.feature_map[0,channel,:].data.numpy().reshape(-1)     
        Activation_alexnet.append(activation_alexnet)
    Activation_alexnet = np.array(Activation_alexnet)
    
    ## Non-face alexnet
    Activation_net = []
    face_channel = [28, 49]
    for n,channel in enumerate(face_channel):
        net_truncate.layeract(picimg)
        activation_net = net_truncate.feature_map[0,channel,:].data.numpy().reshape(-1)
        Activation_net.append(activation_net)
    Activation_net = np.array(Activation_net)
    
    return np.vstack((Activation_alexnet, Activation_net))
    



## 4Shan
# refs vs base
Dict = {'ak_ref': 'h_00_kutcher',
        'ba_ref': 'h_00_affleck',
        'dd_ref': 'h_00_duchovny',
        'gc_ref': 'h_00_clooney',
        'jb_ref': 'h_00_bieber',
        'js_ref': 'h_00_seinfeld',
        'md_ref': 'h_00_daimon',
        'mz_ref': 'h_00_zukerberg',
        'rn_ref': 'h_00_deNiro',
        'th_ref': 'h_00_hanks'}
Refs_Base = []
for k,v in Dict.items():
    Activation_x = Gragh('/home/dell/Desktop/Visulization/4Shan/refs/' + k + '.jpg')
    Activation_y = Gragh('/home/dell/Desktop/Visulization/4Shan/base/' + v + '.jpg')
    Refs_base = []
    for channel in range(6):
        Refs_base.append(Euch_d(Activation_x[channel], Activation_y[channel]))
    Refs_Base.append(Refs_base)
Refs_Base = np.array(Refs_Base)
    

# refs
refs = os.listdir('/home/dell/Desktop/Visulization/4Shan/refs/')
Refs_all_people = []
for i in range(10):
    if i!=9:
        picdir = '/home/dell/Desktop/Visulization/4Shan/refs/' + refs[i]
        Activation_x = Gragh(picdir)
        picdir = '/home/dell/Desktop/Visulization/4Shan/refs/' + refs[i+1]
        Activation_y = Gragh(picdir)
        Refs_x = []
        for channel in range(6):
            Refs_x.append(Euch_d(Activation_x[channel], Activation_y[channel]))
        Refs_all_people.append(Refs_x)
    if i==9:
        picdir = '/home/dell/Desktop/Visulization/4Shan/refs/' + refs[i]
        Activation_x = Gragh(picdir)
        picdir = '/home/dell/Desktop/Visulization/4Shan/refs/' + refs[0]
        Activation_y = Gragh(picdir)
        Refs_x = []
        for channel in range(6):
            Refs_x.append(Euch_d(Activation_x[channel], Activation_y[channel]))
        Refs_all_people.append(Refs_x)
Refs_all_people = np.array(Refs_all_people)


# refs vs hps
Dict = {'ak_ref': 'h_05_kutcher',
        'ba_ref': 'h_05_affleck',
        'dd_ref': 'h_05_duchovny',
        'gc_ref': 'h_05_clooney',
        'jb_ref': 'h_05_bieber',
        'js_ref': 'h_05_seinfeld',
        'md_ref': 'h_05_daimon',
        'mz_ref': 'h_05_zukerberg',
        'rn_ref': 'h_05_deNiro',
        'th_ref': 'h_05_hanks'}
Refs_Highps = []
for k,v in Dict.items():
    Activation_x = Gragh('/home/dell/Desktop/Visulization/4Shan/refs/' + k + '.jpg')
    f_high = os.listdir('/home/dell/Desktop/Visulization/4Shan/high_ps/')
    for i in f_high:
        if i[:len(v)]==v:
            v_real = i
        else:
            pass
    Activation_y = Gragh('/home/dell/Desktop/Visulization/4Shan/high_ps/' + v_real)
    refs_highps = []
    for channel in range(6):
        refs_highps.append(Euch_d(Activation_x[channel], Activation_y[channel]))
    Refs_Highps.append(refs_highps)
Refs_Highps = np.array(Refs_Highps)


# refs vs lps
Dict = {'ak_ref': 'l_05_kutcher',
        'ba_ref': 'l_05_affleck',
        'dd_ref': 'l_05_duchovny',
        'gc_ref': 'l_05_clooney',
        'jb_ref': 'l_05_bieber',
        'js_ref': 'l_05_seinfeld',
        'md_ref': 'l_05_daimon',
        'mz_ref': 'l_05_zukerberg',
        'rn_ref': 'l_05_deNiro',
        'th_ref': 'l_05_hanks'}
Refs_Lowps = []
for k,v in Dict.items():
    Activation_x = Gragh('/home/dell/Desktop/Visulization/4Shan/refs/' + k + '.jpg')
    f_low = os.listdir('/home/dell/Desktop/Visulization/4Shan/low_ps/')
    for i in f_low:
        if i[:len(v)]==v:
            v_real = i
        else:
            pass
    Activation_y = Gragh('/home/dell/Desktop/Visulization/4Shan/low_ps/' + v_real)
    refs_lowps = []
    for channel in range(6):
        refs_lowps.append(Euch_d(Activation_x[channel], Activation_y[channel]))
    Refs_Lowps.append(refs_lowps)
Refs_Lowps = np.array(Refs_Lowps)


# csv
savecsv = '/home/dell/Desktop/Visulization/Familiar_distance.csv'
with open(savecsv, 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    csv_write.writerow(['ID', 
                        'CN_same', 'CN_diff', 'CN_hps', 'CN_lps', 
                        'DN_same', 'DN_diff', 'DN_hps', 'DN_lps'])
    familiar = np.hstack((Refs_Base[:,0].reshape(-1,1), Refs_all_people[:,0].reshape(-1,1), Refs_Highps[:,0].reshape(-1,1), Refs_Lowps[:,0].reshape(-1,1),
                          Refs_Base[:,1].reshape(-1,1), Refs_all_people[:,1].reshape(-1,1), Refs_Highps[:,1].reshape(-1,1), Refs_Lowps[:,1].reshape(-1,1),
                          Refs_Base[:,4].reshape(-1,1), Refs_all_people[:,4].reshape(-1,1), Refs_Highps[:,4].reshape(-1,1), Refs_Lowps[:,4].reshape(-1,1),
                          Refs_Base[:,5].reshape(-1,1), Refs_all_people[:,5].reshape(-1,1), Refs_Highps[:,5].reshape(-1,1), Refs_Lowps[:,5].reshape(-1,1)))
    for i in range(10):
        csv_write.writerow([i+1, 
                            Refs_Base[i,0:2].mean(), 
                            Refs_all_people[i,0:2].mean(),
                            Refs_Highps[i,0:2].mean(),
                            Refs_Lowps[i,0:2].mean(),
                            Refs_Base[i,4:6].mean(),
                            Refs_all_people[i,4:6].mean(),
                            Refs_Highps[i,4:6].mean(),
                            Refs_Lowps[i,4:6].mean()])



## CaucasianUnfamiliar4Shan
# refs vs base
Refs_Base = []
for i in [46,47,48,50,51,55,60,61,62,64,80,81,82,83,85]:
    for refs_pic in os.listdir('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/'):
        if int(refs_pic[2:4])==i:
            k = refs_pic
    Activation_x = Gragh('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/' + k)
    for base_pic in os.listdir('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/base/'):
        if int(base_pic[5:7])==i:
            v = base_pic
    Activation_y = Gragh('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/base/' + v)
    Refs_base = []
    for channel in range(6):
        Refs_base.append(Euch_d(Activation_x[channel], Activation_y[channel]))
    Refs_Base.append(Refs_base)
Refs_Base = np.array(Refs_Base)
    

# refs
refs = os.listdir('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/')
Refs_all_people = []
for i in range(15):
    if i!=14:
        picdir = '/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/' + refs[i]
        Activation_x = Gragh(picdir)
        picdir = '/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/' + refs[i+1]
        Activation_y = Gragh(picdir)
        Refs_x = []
        for channel in range(6):
            Refs_x.append(Euch_d(Activation_x[channel], Activation_y[channel]))
        Refs_all_people.append(Refs_x)
    if i==14:
        picdir = '/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/' + refs[i]
        Activation_x = Gragh(picdir)
        picdir = '/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/' + refs[0]
        Activation_y = Gragh(picdir)
        Refs_x = []
        for channel in range(6):
            Refs_x.append(Euch_d(Activation_x[channel], Activation_y[channel]))
        Refs_all_people.append(Refs_x)
Refs_all_people = np.array(Refs_all_people)


# refs vs hps
Refs_Highps = []
for i in [46,47,48,50,51,55,60,61,62,64,80,81,82,83,85]:
    for refs_pic in os.listdir('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/'):
        if int(refs_pic[2:4])==i:
            k = refs_pic
    Activation_x = Gragh('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/' + k)
    for highps_pic in os.listdir('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/high_ps/'):
        if int(highps_pic[5:7])==i:
            v = highps_pic
    Activation_y = Gragh('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/high_ps/' + v)
    refs_highps = []
    for channel in range(6):
        refs_highps.append(Euch_d(Activation_x[channel], Activation_y[channel]))
    Refs_Highps.append(refs_highps)
Refs_Highps = np.array(Refs_Highps)


# refs vs lps
Refs_Lowps = []
for i in [46,47,48,50,51,55,60,61,62,64,80,81,82,83,85]:
    for refs_pic in os.listdir('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/'):
        if int(refs_pic[2:4])==i:
            k = refs_pic
    Activation_x = Gragh('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/refs/' + k)
    for lowps_pic in os.listdir('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/low_ps/'):
        if int(lowps_pic[5:7])==i:
            v = lowps_pic
    Activation_y = Gragh('/home/dell/Desktop/Visulization/CaucasianUnfamiliar4Shan/low_ps/' + v)
    refs_lowps = []
    for channel in range(6):
        refs_lowps.append(Euch_d(Activation_x[channel], Activation_y[channel]))
    Refs_Lowps.append(refs_lowps)
Refs_Lowps = np.array(Refs_Lowps)


# csv
savecsv = '/home/dell/Desktop/Visulization/Unfamiliar_distance.csv'
with open(savecsv, 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    csv_write.writerow(['ID', 
                        'CN_same', 'CN_diff', 'CN_hps', 'CN_lps', 
                        'DN_same', 'DN_diff', 'DN_hps', 'DN_lps'])
    familiar = np.hstack((Refs_Base[:,0].reshape(-1,1), Refs_all_people[:,0].reshape(-1,1), Refs_Highps[:,0].reshape(-1,1), Refs_Lowps[:,0].reshape(-1,1),
                          Refs_Base[:,1].reshape(-1,1), Refs_all_people[:,1].reshape(-1,1), Refs_Highps[:,1].reshape(-1,1), Refs_Lowps[:,1].reshape(-1,1),
                          Refs_Base[:,4].reshape(-1,1), Refs_all_people[:,4].reshape(-1,1), Refs_Highps[:,4].reshape(-1,1), Refs_Lowps[:,4].reshape(-1,1),
                          Refs_Base[:,5].reshape(-1,1), Refs_all_people[:,5].reshape(-1,1), Refs_Highps[:,5].reshape(-1,1), Refs_Lowps[:,5].reshape(-1,1)))
    for i in range(15):
        csv_write.writerow([i+1, 
                            Refs_Base[i,0:2].mean(), 
                            Refs_all_people[i,0:2].mean(),
                            Refs_Highps[i,0:2].mean(),
                            Refs_Lowps[i,0:2].mean(),
                            Refs_Base[i,4:6].mean(),
                            Refs_all_people[i,4:6].mean(),
                            Refs_Highps[i,4:6].mean(),
                            Refs_Lowps[i,4:6].mean()])


