'''
###############################################################################
###############################################################################
################ Select and transform NoFace Picture in ImgNet ################
###############################################################################
'''


'1. Select the optimal params'
###############################################################################
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
sys.path.append('/home/dell/Desktop/FaceInsight/faceinsight')

from PIL import Image, TarIO
from detection.detector import detect_faces

#假设文件名为 n01440764.tar 并置于E盘下
import tarfile

#读取文件名，并放到list中
name_list = []
with tarfile.open("/home/dell/Desktop/traindata/ILSVRC2012_img_train.tar", "r") as file:
    for i in file.getmembers():
        name_list.append(i.name)

indir = '/home/dell/Desktop/Imgnet_traindata_2012/'+name_list[500]    
jpg_name = []
with tarfile.open(indir, "r") as file:
    for i in file.getmembers():
        jpg_name.append(i.name)
        
#Real category
real_category = np.zeros((98),dtype=bool)
real_category[0] = True
real_category[11] = True
real_category[15] = True
real_category[31] = True
real_category[35] = True
real_category[40] = True
real_category[44] = True
real_category[46] = True
real_category[52] = True
real_category[60] = True
real_category[78] = True
real_category[88] = True
real_category[96] = True
        

evaluation = {}     
for i in [5.0,10.0,15.0,20.0]:
    for j in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        for k in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            FACE = []
            for pic in range(100):
                fp = TarIO.TarIO(indir, jpg_name[pic])
                im = Image.open(fp)
                try:
                    _, landmarks = detect_faces(im, min_face_size=i, thresholds=[0.6, 0.7, j], nms_thresholds=[0.7, 0.7, k])
                    FACE.append(len(landmarks) != 0)
                except:
                    continue

            face = np.array(FACE)
            
            hit = np.where(face[np.where(real_category==True)[0]]==True)[0].size/np.where(real_category==True)[0].size
            correct_reject = np.where(face[np.where(real_category==False)[0]]==False)[0].size/np.where(real_category==False)[0].size
            miss = np.where(face[np.where(real_category==True)[0]]==False)[0].size/np.where(real_category==True)[0].size
            false_alarm = np.where(face[np.where(real_category==False)[0]]==True)[0].size/np.where(real_category==False)[0].size
        
            key = (i, j, k)
            value = [hit, correct_reject, miss, false_alarm]
            evaluation[key] = value
            
EVA = []
for index,key in enumerate(evaluation):
    eva = evaluation[key][0] - evaluation[key][2]
    EVA.append(eva)
EVA = np.array(EVA)
np.where(EVA==np.max(EVA))

EVA_new = []
for index,key in enumerate(evaluation):
    eva = evaluation[key][1] - evaluation[key][3]
    EVA_new.append(eva)
EVA_new = np.array(EVA_new)

e = 0.7*EVA+0.3*EVA_new
np.where(e==np.max(e))
for index,key in enumerate(evaluation):
    print(index, key)
    
    
    
'2. plt'
###############################################################################
###############################################################################
FACE = []
for pic in range(100):
    fp = TarIO.TarIO(indir, jpg_name[pic])
    im = Image.open(fp)
    try:
        #bounding_boxes is [n,5], n is the number of face, every face have 5 number, 4 of 5number is bounding
        #landmarks is [n,10], 10 is Five senses, top 5 in 10 is Five_senses's x, other 5 is Five_senses's y
        bounding_boxes, landmarks = detect_faces(im, min_face_size=20.0, thresholds=[0.6, 0.7, 0.4])
        FACE.append(len(bounding_boxes) != 0)
    except:
        continue

face = np.array(FACE)
face_pos = np.where(face==True)
face_num = face_pos[0].size

noface_pos = np.where(face==False)
noface_num = noface_pos[0].size

jpg = jpg_name[:100]
jpg.pop(16)
jpg.pop(27)

for i in range(noface_num):
    fp = TarIO.TarIO(indir, jpg[noface_pos[0][i]])
    im = Image.open(fp)
    img = np.array(im)
    plt.figure()
    plt.imshow(img)
    plt.show()


for i in range(100,200):
    fp = TarIO.TarIO(indir, jpg_name[i])
    im = Image.open(fp)
    img = np.array(im)
    plt.figure()
    plt.imshow(img)
    plt.show()
    
    

'3. Use selected prarms to choose face and noface picture'
###############################################################################
###############################################################################
import sys
import os
sys.path.append(os.getcwd())
sys.path.append('/home/dell/Desktop/FaceInsight/faceinsight')

from PIL import Image, TarIO
from detection.detector import detect_faces
import tarfile

#读取文件名，并放到list中
name_list = []
with tarfile.open("/home/dell/Desktop/traindata/ILSVRC2012_img_train.tar", "r") as file:
    for i in file.getmembers():
        name_list.append(i.name)

os.makedirs('/home/dell/Desktop/Generated_Noface/nf0001')
noface_dir = '/home/dell/Desktop/Generated_Noface/nf0001'
number = 0
for index_pack, pack in enumerate(name_list):#range(len(name_list)):
    indir = '/home/dell/Desktop/Imgnet_traindata_2012/'+pack   
    
    jpg_name = []
    with tarfile.open(indir, "r") as file:
        for i in file.getmembers():
            jpg_name.append(i.name)
    
    for index_pic, pic in enumerate(jpg_name):
        fp = TarIO.TarIO(indir, pic)
        im = Image.open(fp)
        try:
        #bounding_boxes is [n,5], n is the number of face, every face have 5 number, 4 of 5number is bounding
        #landmarks is [n,10], 10 is Five senses, top 5 in 10 is Five_senses's x, other 5 is Five_senses's y
            bounding_boxes, landmarks = detect_faces(im, min_face_size=20.0, thresholds=[0.6, 0.7, 0.4])
            print('index_pack:', index_pack, 'index_pic:', index_pic, indir, pic)
            
            if len(landmarks) != 0:    #have face
                face_in_dir = '/home/dell/Desktop/Generated_face/'+pic
                im.save(face_in_dir)
                    
            elif len(landmarks) == 0:    #no face
                number = number + 1
                if number%5000 != 0:
                    noface_in_dir = noface_dir+'/'+pic
                    im.save(noface_in_dir)
                else:
                    num = "%04d" % (int(noface_dir[-4:])+1)
                    noface_dir = '/home/dell/Desktop/Generated_Noface/'+'nf'+num
                    os.makedirs(noface_dir)
                    noface_in_dir = noface_dir+'/'+pic
                    im.save(noface_in_dir)                       
        except:
            continue
        
        
        
'4. Princple of phase scrambling'    
###############################################################################
###############################################################################
#import math  
#fp = TarIO.TarIO(indir, jpg_name[0])
#im = Image.open(fp)    
#bounding_boxes, landmarks = detect_faces(im, min_face_size=20.0, thresholds=[0.6, 0.7, 0.4])
#show_bboxes(im, bounding_boxes, landmarks)
#img = np.array(im)
#pro = img[int(bounding_boxes[0][1]):int(bounding_boxes[0][3]), int(bounding_boxes[0][0]):int(bounding_boxes[0][2]), 2]
#F = np.fft.fft2(pro)
#F_mag = np.abs(np.fft.fftshift(F))
#F_phase = np.angle(np.fft.fftshift(F))
#Fnew_phase = 2.0*math.pi*np.random.rand(F_phase.shape[0], F_phase.shape[1])
#Fnew = F_mag*np.exp(1j*(Fnew_phase))
#fnew = np.fft.ifft2(np.fft.ifftshift(Fnew))
#img[int(bounding_boxes[0][1]):int(bounding_boxes[0][3]), int(bounding_boxes[0][0]):int(bounding_boxes[0][2]), 2] = np.real(fnew)
#plt.figure(figsize=(10,10))
#plt.imshow(img[:,:,2])



'5. Show the bounds'
###############################################################################
###############################################################################
from PIL import ImageDraw
from detection.visualization_utils import show_bboxes

# one way
draw = ImageDraw.Draw(im)
for b in bounding_boxes:
    draw.rectangle([(b[0], b[1]), (b[2], b[3])])
im

# anther way
show_bboxes(im, bounding_boxes, landmarks)



'6. Use package to phase scrambling'
###############################################################################
###############################################################################
import os
from PIL import Image
import sys
sys.path.append('/home/dell/Desktop/pyMask-master')
import makeMask

directory_name = '/home/dell/Desktop/Generated_face/'
mask_dir = '/home/dell/Desktop/Mask_face/mask0001'
filename = os.listdir(directory_name)

number = 0
for index_pack, pack in enumerate(filename):
    indir = '/home/dell/Desktop/Generated_face/'+pack        
    im = Image.open(indir)   
    img = np.array(im)
    
    bounding_boxes, landmarks = detect_faces(im, min_face_size=20.0, thresholds=[0.6, 0.7, 0.4])
    
    for box in bounding_boxes:
        try:
            box = abs(box)
            pro = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            pro_0 = pro[:,:,0]
            pro_1 = pro[:,:,1]
            pro_2 = pro[:,:,2]
            
            scrambled_0 = makeMask.makeMaskOneChannelNormalized(pro_0)
            img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), 0] = scrambled_0           
            scrambled_1 = makeMask.makeMaskOneChannelNormalized(pro_1)
            img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), 1] = scrambled_1
            scrambled_2 = makeMask.makeMaskOneChannelNormalized(pro_2)
            img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), 2] = scrambled_2          
        except:
            continue
    im_save = Image.fromarray(img.astype('uint8')).convert('RGB')
    
    number = number+1
    if number%5000 != 0:
        savedir = mask_dir + '/' + pack
        im_save.save(savedir)
    else:
        num = "%04d" % (int(mask_dir[-4:])+1)
        mask_dir = '/home/dell/Desktop/Mask_face/'+'mask'+num
        os.makedirs(mask_dir)
        mask_in_dir = mask_dir + '/' + pack
        im_save.save(mask_in_dir) 
        
    print('index_pack:', index_pack, indir)
    
    
    
'7. Retrain the model'
###############################################################################
###############################################################################
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader     
import torchvision.transforms as transforms

import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import io as dnn_io
from dnnbrain.dnn import models
from dnnbrain.dnn import analyzer

   
'Real_noface'
###############################################################################
import scipy.io as sio 
import csv

def mat2list(mat):
    L = []
    for i in range(mat['stimnames'].shape[0]):
        l = mat['stimnames'][i][0][0][0]
        L.append(l[:9]+'/'+l)
    return L

def txt2dic(txtdir):
    tmp_dict = {}
    with open(txtdir, 'r') as f:
        txt = f.read()
        l1 = []
        l2 = []
        for index,t in enumerate(txt.split()):
            if index%2 == 0:
                l1.append(t)
            else:
                l2.append(t)
        tmp_dict = dict(zip(l1, l2))
    return tmp_dict

txtdir_train = '/home/dell/Desktop/Imgnet/train.txt'
tmp_dict_train = txt2dic(txtdir_train)
convert_dict_train = {v:k[:9] for k,v in tmp_dict_train.items()}

txtdir_val = '/home/dell/Desktop/Imgnet/val.txt'
tmp_dict_val = txt2dic(txtdir_val)
convert_dict_val = {v:k for k,v in tmp_dict_val.items()}


def write_csv(path, sheet_name, tmp_dict_train, tmp_dict_val):
    d = pd.read_excel('/home/dell/Desktop/data_sony\win7\win10\dell/arrangement.xlsx', sheet_name=sheet_name)
    N = {'华硕':'sony', 'win7':'win7', 'win10':'win10','DELL':'dell'}
    D = {'华硕':'data_sony/', 'win7':'data_win7/', 'win10':'data_win10/','DELL':'data_dell/'}
    f = open(path,'wb')
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        if sheet_name == 1:
            csv_head = ['/home/dell/Desktop/Imgnet/traindata/']
            csv_write.writerow(csv_head)
        if sheet_name == 2:
            csv_head = ['/home/dell/Desktop/Imgnet/valdata/']
            csv_write.writerow(csv_head)
        csv_head = ["stimID","condition"]
        csv_write.writerow(csv_head)
        for i in range(d.shape[0]):
            print(i)
            name = d.iloc[i][0]
            if d.iloc[i][1] == d.iloc[i][2]:
                indexdir = '/home/dell/Desktop/data_sony\win7\win10\dell/'+D[d.iloc[i][1]]+name+'_'+N[d.iloc[i][1]]+'.mat'
                mat = sio.loadmat(indexdir)
                L = mat2list(mat)
                label = []
                for j in L:
                    if sheet_name == 1:
                        label.append(tmp_dict_train[j])
                    if sheet_name == 2:
                        label.append(tmp_dict_val[j[10:]])
            else:
                indexdir1 = '/home/dell/Desktop/data_sony\win7\win10\dell/'+D[d.iloc[i][1]]+name+'_'+N[d.iloc[i][1]]+'.mat'
                mat1 = sio.loadmat(indexdir1)
                L1 = mat2list(mat1)
                indexdir2 = '/home/dell/Desktop/data_sony\win7\win10\dell/'+D[d.iloc[i][2]]+name+'_'+N[d.iloc[i][2]]+'.mat'
                mat2 = sio.loadmat(indexdir2)
                L2 = mat2list(mat2)
                L = list(set(L1).intersection(set(L2)))
                label = []
                for j in L:    
                    if sheet_name == 1:
                        label.append(tmp_dict_train[j])
                    if sheet_name == 2:
                        label.append(tmp_dict_val[j[10:]])
            
            for t in range(len(L)):
                if sheet_name == 1:
                    data_row = [L[t], label[t]]
                if sheet_name == 2:
                    data_row = [L[t][10:], label[t]]
                csv_write.writerow(data_row)
        
write_csv('/home/dell/Desktop/noface_train.csv', 1, tmp_dict_train, tmp_dict_val)
write_csv('/home/dell/Desktop/noface_val.csv', 2, tmp_dict_train, tmp_dict_val)


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ])])
picdataset_train = dnn_io.PicDataset('/home/dell/Desktop/data_sony\\win7\\win10\\dell/real_noface_train.csv', transform=train_transform)
picdataloader_train = DataLoader(picdataset_train, batch_size=128, shuffle=True, num_workers=40)

val_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ])])
picdataset_val = dnn_io.PicDataset('/home/dell/Desktop/data_sony\\win7\\win10\\dell/real_noface_val.csv', transform=val_transform)
picdataloader_val = DataLoader(picdataset_val, batch_size=128, shuffle=False, num_workers=40)


train_target = picdataset_train.condition
sta_train = set(train_target)
class_640 = []
for i in sta_train:
    if np.where(train_target==i)[0].shape[0] > 640:
        class_640.append(i)        



# generate above 640 pictures class csv
f1 = pd.read_csv('/home/dell/Desktop/real_noface_train.csv')
d1 = f1.set_index('stimID').to_dict()['condition']
f2 = pd.read_csv('/home/dell/Desktop/real_noface_val.csv')
d2 = f2.set_index('stimID').to_dict()['condition']

num_class = []
for i in class_640:
    num_class.append(np.where(np.array(list(d1.values()))==i)[0].shape[0])


def write_csv_640(path, sheet_name, d1, d2, tmp_dict_train, tmp_dict_val, class_n):
    d = pd.read_excel('/home/dell/Desktop/data_sony\win7\win10\dell/arrangement.xlsx', sheet_name=sheet_name)
    N = {'华硕':'sony', 'win7':'win7', 'win10':'win10','DELL':'dell'}
    D = {'华硕':'data_sony/', 'win7':'data_win7/', 'win10':'data_win10/','DELL':'data_dell/'}
    f = open(path,'wb')
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        if sheet_name == 1:
            csv_head = ['/home/dell/Desktop/Imgnet/traindata/']
            csv_write.writerow(csv_head)
        if sheet_name == 2:
            csv_head = ['/home/dell/Desktop/Imgnet/valdata/']
            csv_write.writerow(csv_head)
        csv_head = ["stimID","condition"]
        csv_write.writerow(csv_head)
        for i in range(d.shape[0]):
            print(i)
            name = d.iloc[i][0]
            if d.iloc[i][1] == d.iloc[i][2]:
                indexdir = '/home/dell/Desktop/data_sony\win7\win10\dell/'+D[d.iloc[i][1]]+name+'_'+N[d.iloc[i][1]]+'.mat'
                mat = sio.loadmat(indexdir)
                L = mat2list(mat)
                List = []
                label = []
                for j in L:
                    if sheet_name == 1:
                        if d1[j] in class_n:
                            List.append(j)
                            label.append(d1[j])
                    if sheet_name == 2:
                        if d2[j[10:]] in class_n:
                            List.append(j[10:])
                            label.append(d2[j[10:]])
            else:
                indexdir1 = '/home/dell/Desktop/data_sony\win7\win10\dell/'+D[d.iloc[i][1]]+name+'_'+N[d.iloc[i][1]]+'.mat'
                mat1 = sio.loadmat(indexdir1)
                L1 = mat2list(mat1)
                indexdir2 = '/home/dell/Desktop/data_sony\win7\win10\dell/'+D[d.iloc[i][2]]+name+'_'+N[d.iloc[i][2]]+'.mat'
                mat2 = sio.loadmat(indexdir2)
                L2 = mat2list(mat2)
                L = list(set(L1).intersection(set(L2)))
                List = []
                label = []
                for j in L:    
                    if sheet_name == 1:
                        if d1[j] in class_n:
                            List.append(j)
                            label.append(d1[j])
                    if sheet_name == 2:
                        if d2[j[10:]] in class_n:
                            List.append(j[10:])
                            label.append(d2[j[10:]])
            
            for t in range(len(List)):
                if sheet_name == 1:
                    data_row = [List[t], label[t]]
                if sheet_name == 2:
                    data_row = [List[t], label[t]]
                csv_write.writerow(data_row)
        
write_csv_640('/home/dell/Desktop/data_sony\\win7\\win10\\dell/real_noface_train.csv', 1, d1, d2, tmp_dict_train, tmp_dict_val, class_640)
write_csv_640('/home/dell/Desktop/data_sony\\win7\\win10\\dell/real_noface_val.csv', 2, d1, d2, tmp_dict_train, tmp_dict_val, class_640)


# load the stim and data
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
    
picdataset_train = dnn_io.PicDataset('/home/dell/Desktop/data_sony\\win7\\win10\\dell/real_noface_train.csv', transform=data_transforms['train'])
picdataloader_train = DataLoader(picdataset_train, batch_size=128, shuffle=True, num_workers=40)

picdataset_train_val = dnn_io.PicDataset('/home/dell/Desktop/data_sony\\win7\\win10\\dell/real_noface_train.csv', transform=data_transforms['val'])
dataloaders_train_test = DataLoader(picdataset_train_val, batch_size=128, shuffle=False, num_workers=40)

picdataset_test_val = dnn_io.PicDataset('/home/dell/Desktop/data_sony\\win7\\win10\\dell/real_noface_val.csv', transform=data_transforms['val'])
dataloaders_val_test = DataLoader(picdataset_test_val, batch_size=128, shuffle=False, num_workers=40)


# train model
alexnet = torchvision.models.alexnet(pretrained=False)
alexnet.classifier[6] = torch.nn.Linear(4096, 736, bias=True)
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=250**(-1/3))
path_loss_acc = '/home/dell/Desktop/Nonface_alexnet/metric_nonface_net.csv'
save_model_path = '/home/dell/Desktop/Nonface_alexnet/'
model, metric_dict = models.dnn_train_model(picdataloader_train, 
                                            dataloaders_train_test, 
                                            dataloaders_val_test,
                                            alexnet, 
                                            criterion, 
                                            optimizer, 
                                            num_epoches=100, 
                                            scheduler=scheduler,
                                            train_method='tradition',
                                            save_model_path=save_model_path,
                                            metric_path=path_loss_acc)


def plot_metric(metric_path):
    metric = pd.read_csv(metric_path)
    metric.iloc[0]
    
    # plot LOSS curve
    plt.figure()
    plt.title('LOSS')
    plt.plot(metric.T.iloc[1], label="LOSS")
    plt.legend()
    
    # plot 4 ACC curves
    plt.figure()
    plt.title('ACC of Train and Test')
    plt.plot(metric.T.iloc[2], label="ACC_train_top1")
    plt.plot(metric.T.iloc[3], label="ACC_train_top5")
    plt.plot(metric.T.iloc[4], label="ACC_val_top1")
    plt.plot(metric.T.iloc[5], label="ACC_val_top5")
    plt.legend()
plot_metric(path_loss_acc)

metric_path = path_loss_acc
metric = pd.read_csv(metric_path)
plt.figure(figsize=(7,6), dpi=80)
plt.xlim((0,1.5))
plt.xticks([0.4,1.1], ['Top1', 'Top2'])
plt.ylabel('Accuracy')
plt.bar([0.4,1.1], [metric.T.iloc[4][90], metric.T.iloc[5][90]], width=0.3, facecolor='red')
plt.text(0.35, 0.57+0.01, 0.57)
plt.text(1.05, 0.80+0.01, 0.80)


# test the last model in all classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch.load('/home/dell/Desktop/data_sony\\win7\\win10\\dell/model_epoch89')
alexnet = torchvision.models.alexnet(pretrained=True).to(device)

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
picdataset_alexnet_val = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/val.csv', transform=data_transforms['val'])
dataloaders_alexnet_val = DataLoader(picdataset_alexnet_val, batch_size=128, shuffle=False, num_workers=40)

picdataset_net_val = dnn_io.PicDataset('/home/dell/Desktop/data_sony\\win7\\win10\\dell/real_noface_val.csv', transform=data_transforms['val'])
dataloaders_net_val = DataLoader(picdataset_net_val, batch_size=128, shuffle=False, num_workers=40)

models.dnn_test_model(dataloaders_net_val, net)      # test for the last Nonface_net

mapping = dict()      # map 736class to 1000class
for index, (inputs,targets) in enumerate(picdataset_net_val):
    mapping[targets] = picdataset_net_val.condition[index]
inv_mapping = dict([(v,k) for k,v in mapping.items()])

condition = np.array([])
for i in picdataloader_train.dataset.condition:
    condition = np.append(condition, inv_mapping[i])
picnumber = np.array([])
for i in range(736):
    picnumber = np.append(picnumber, np.where(condition==i)[0].shape[0])
            
def test(dataloader, network, n_class):
    network.eval()
    ACC_top1 = np.zeros((1,n_class))
    ACC_top5 = np.zeros((1,n_class))
    N = np.zeros((1,n_class))
    time0 = time.time()
    for inputs, targets in dataloader:
        if network==alexnet:
            targets = torch.Tensor([mapping[j.item()] for j in targets]).int()
        else:
            pass
        for i in targets:
            N[0][i.cpu()] += 1
        inputs = inputs.to(device)
        targets = targets.to(device)
        out = network(inputs) 
        _, outputs_top1 = torch.max(out, 1)
        outputs_top5 = torch.topk(out, 5)[1]
        for i in range(n_class):
            location = torch.where(targets==i)[0]
            acc_top1 = torch.sum((outputs_top1[location].int()==targets[location].int()))
            acc_top1 = acc_top1.item()
            if location.shape == torch.Size([0]):
                acc_top5 = 0
            else:
                acc_top5 = 0
                for raw in range(outputs_top5[location].shape[0]):
                    acc_top5 += np.sum(targets[location].int()[raw] in outputs_top5[location].int()[raw])
            ACC_top1[0, i] = ACC_top1[0, i] + acc_top1
            ACC_top5[0, i] = ACC_top5[0, i] + acc_top5
    time1 = time.time()
    print(time1-time0, 's')
    return ACC_top1, ACC_top5, N

#ACC_alexnet, N_alexnet = test(dataloaders_alexnet_val, alexnet, 1000)
ACC_alexnet_top1, ACC_alexnet_top5, N_alexnet = test(dataloaders_net_val, alexnet, 1000)    
ACC_alexnet_top1 = ACC_alexnet_top1/N_alexnet
ACC_alexnet_top5 = ACC_alexnet_top5/N_alexnet
plt.figure()
plt.bar(np.arange(1000), ACC_alexnet_top1.reshape(1000))
plt.bar(np.arange(1000), ACC_alexnet_top5.reshape(1000))

ACC_net_top1, ACC_net_top5, N_net = test(dataloaders_net_val, net, 736)
ACC_net_top1 = ACC_net_top1/N_net
ACC_net_top5 = ACC_net_top5/N_net
ACC2 = np.zeros(1000)
for i in range(736):
    ACC2[mapping[i]] = ACC_net_top1[0][i]
plt.figure()
plt.bar(np.arange(1000), ACC2)

plt.figure()
index_net = np.where(picnumber>0)[0]
index_alexnet = np.array([mapping[i] for i in index_net])
plt.plot(ACC_alexnet_top5[0][index_alexnet])
plt.plot(ACC_net_top5[0][index_net])




'13. Baseline_model'
###############################################################################
###############################################################################
import os
import pandas as pd
import csv
import tarfile
import random
from PIL import TarIO, Image

#l = os.listdir('/home/dell/Desktop/Imgnet/traindata/')
#l = random.sample(l, 736)      # 793 classes over 640 pictures in each one (average number of pictures in a class is 1281)

def txt2dic(txtdir):
    tmp_dict = {}
    with open(txtdir, 'r') as f:
        txt = f.read()
        l1 = []
        l2 = []
        for index,t in enumerate(txt.split()):
            if index%2 == 0:
                l1.append(t)
            else:
                l2.append(t)
        tmp_dict = dict(zip(l1, l2))
    return tmp_dict

txtdir_train = '/home/dell/Desktop/Imgnet/train.txt'
tmp_dict_train = txt2dic(txtdir_train)
convert_dict_train = {v:k[:9] for k,v in tmp_dict_train.items()}
set_dict_train = {v:k[:9] for k,v in convert_dict_train.items()}
txtdir_val = '/home/dell/Desktop/Imgnet/val.txt'
tmp_dict_val = txt2dic(txtdir_val)
convert_dict_val = {v:k for k,v in tmp_dict_val.items()}


f = pd.read_csv('/home/dell/Desktop/data_sony\win7\win10\dell/real_noface_train.csv')
class_736 = []
for classes in range(1,len(f)):
    class_736.append(f.iloc[classes].name[0:9])
class_736 = np.array(class_736)
l = list(set(class_736))

picnumber = []
for i in l:
    picnumber.append(np.where(class_736==i)[0].shape[0])


path = '/home/dell/Desktop/baseline_model/involve_face_train.csv'
with open(path,'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    csv_head = ['/home/dell/Desktop/Imgnet/traindata/']
    csv_write.writerow(csv_head)
    csv_head = ["stimID", "condition"]
    csv_write.writerow(csv_head)   
    for i in range(736):
        folder = '/home/dell/Desktop/Imgnet/traindata/' + l[i]
        imgfolder = os.listdir(folder)
        imgs = random.sample(imgfolder, int(picnumber[i])) 
        imgname = []
        imglabel = []
        for j in imgs:
            name = j[:9]+'/'+j
            imgname.append(name)
            imglabel.append(tmp_dict_train[name])
        for t in range(len(imgname)):
            data_row = [imgname[t], imglabel[t]]
            csv_write.writerow(data_row)
            
path = '/home/dell/Desktop/baseline_model/involve_face_val.csv'
with open(path,'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    csv_head = ['/home/dell/Desktop/Imgnet/valdata/']
    csv_write.writerow(csv_head)
    csv_head = ["stimID", "condition"]
    csv_write.writerow(csv_head)   
    classnumber = []
    for i in l:
        classnumber.append(set_dict_train[i])
    for k,v in tmp_dict_val.items():
        if v in classnumber:
            data_row = [k, v]
            csv_write.writerow(data_row)
            
            
# train baseline model
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
    
picdataset_train = dnn_io.PicDataset('/home/dell/Desktop/baseline_model/involve_face_train.csv', transform=data_transforms['train'])
picdataloader_train = DataLoader(picdataset_train, batch_size=128, shuffle=True, num_workers=40)

picdataset_train_val = dnn_io.PicDataset('/home/dell/Desktop/baseline_model/involve_face_train.csv', transform=data_transforms['val'])
dataloaders_train_test = DataLoader(picdataset_train_val, batch_size=128, shuffle=False, num_workers=40)

picdataset_test_val = dnn_io.PicDataset('/home/dell/Desktop/baseline_model/involve_face_val.csv', transform=data_transforms['val'])
dataloaders_val_test = DataLoader(picdataset_test_val, batch_size=128, shuffle=False, num_workers=40)


alexnet = torchvision.models.alexnet(pretrained=False).to(device)
alexnet.classifier[6] = torch.nn.Linear(4096, len(l), bias=True)
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
path_loss_acc = '/home/dell/Desktop/baseline_model/Metric.csv'
save_model_path = '/home/dell/Desktop/baseline_model/'
model = models.dnn_train_model(picdataloader_train, 
                           dataloaders_train_test, 
                           dataloaders_val_test,
                           alexnet, 
                           criterion, 
                           optimizer, 
                           100, 
                           save_model_path=save_model_path,
                           metric_path=path_loss_acc)


def plot_metric(metric_path):
    metric = pd.read_csv(metric_path)
    metric.iloc[0]    
    # plot LOSS curve
    plt.figure()
    plt.title('LOSS')
    plt.plot(metric.T.iloc[0], label="LOSS")
    plt.legend()    
    # plot 4 ACC curves
    plt.figure()
    plt.title('ACC of Train and Test')
    plt.plot(metric.T.iloc[1], label="ACC_train_top1")
    plt.plot(metric.T.iloc[2], label="ACC_val_top1")
    plt.legend()
plot_metric(path_loss_acc)
            
            
            
            
'8. Find and analysis Face channel'
###############################################################################
###############################################################################
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

#f = os.listdir('/home/dell/Desktop/205cat/FITW_1000')
#f = random.sample(f, 80)
#with open('/home/dell/Desktop/205cat/face_80.csv', 'a+') as csvfile:
#    csv_write = csv.writer(csvfile)
#    csv_write.writerow(['/home/dell/Desktop/205cat/FITW_1000/'])
#    csv_write.writerow(['stimID', 'condition'])
#    for pic in f:
#        csv_write.writerow([pic, 'face'])

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
# face and object stimuli
#picdataset_face = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_face_1000_FITW.csv', transform=transform)
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
### 1000 Face  11th-layer  non-face net
from scipy import stats

selected_layer = 11
net_truncate = NET(net, selected_layer=selected_layer)
channel_face = []
for imgs,_ in picdataset_face:
    net_truncate.layeract(imgs.unsqueeze(0).to(device))
    channel_face.append(net_truncate.conv_output.cpu().data.numpy())
channel_face = np.array(channel_face)


### 1000 Face  11th-layer  Alexnet 
net_truncate_alexnet = NET(net_alexnet, selected_layer=selected_layer)
channel_face_alexnet = []
for imgs,_ in picdataset_face:
    net_truncate_alexnet.layeract(imgs.unsqueeze(0).to(device))
    channel_face_alexnet.append(net_truncate_alexnet.conv_output.cpu().data.numpy())
channel_face_alexnet = np.array(channel_face_alexnet)


### Find Face channel
# non-face alexnet 11th in objext pictures (channel_class_noface shape is (204,256))
net_truncate = NET(net, selected_layer=selected_layer)

label = []
for index,(imgs,_) in enumerate(picdataset_noface):
    label.append(picdataset_noface.picname[index][:3])
label = np.array(label)

channel_class_noface = np.array([]) 
channel_noface = np.array([])
Conv5_all_noface = []
for index,(imgs,_) in enumerate(picdataset_noface):
    net_truncate.layeract(imgs.unsqueeze(0).to(device))
    if index in np.where(label==picdataset_noface.picname[index][:3])[0]:
        channel_noface = np.append(channel_noface, net_truncate.conv_output.cpu().data.numpy())
        channel_noface = channel_noface.reshape(-1, net_truncate.conv_output.shape[0])
        if index==np.where(label==picdataset_noface.picname[index][:3])[0][-1]:
            Conv5_all_noface.append(channel_noface)
            channel_class_noface = np.append(channel_class_noface, np.mean(channel_noface, 0))
            channel_noface = np.array([])
channel_class_noface = channel_class_noface.reshape(-1, net_truncate.conv_output.shape[0])
Conv5_all_noface = np.array(Conv5_all_noface)


# Alexnet 11th layer in object pictures (channel_class_noface_alexnet shape is (204,256))
net_truncate_alexnet = NET(net_alexnet, selected_layer=selected_layer)

channel_class_noface_alexnet = np.array([]) 
channel_noface_alexnet = np.array([])
alexConv5_all_noface = []
for index,(imgs,_) in enumerate(picdataset_noface):
    net_truncate_alexnet.layeract(imgs.unsqueeze(0).to(device))
    if index in np.where(label==picdataset_noface.picname[index][:3])[0]:
        channel_noface_alexnet = np.append(channel_noface_alexnet, net_truncate_alexnet.conv_output.cpu().data.numpy())
        channel_noface_alexnet = channel_noface_alexnet.reshape(-1, net_truncate_alexnet.conv_output.shape[0])
        if index==np.where(label==picdataset_noface.picname[index][:3])[0][-1]:
            channel_class_noface_alexnet = np.append(channel_class_noface_alexnet, np.mean(channel_noface_alexnet, 0))
            alexConv5_all_noface.append(channel_noface_alexnet)
            channel_noface_alexnet = np.array([])
channel_class_noface_alexnet = channel_class_noface_alexnet.reshape(-1, net_truncate_alexnet.conv_output.shape[0])
alexConv5_all_noface = np.array(alexConv5_all_noface)



## find face channel in nonface net (face_channel)
def minmax(data):
    return (data-data.min())/(data.max()-data.min())


conv5_nonface = np.mean(channel_face, axis=0)
obj_activation = channel_class_noface.T
face_obj_net = stats.zscore(np.column_stack((conv5_nonface, obj_activation)), axis=1)      # shape is (256, 205)
np.savetxt("/home/dell/Desktop/DN_tuning.csv", face_obj_net, delimiter=',')

          
# use Mann-U test to select face channel             
face_channel = []
for channel in range(256):
    if np.max(np.mean(Conv5_all_noface[1:,:,channel],axis=1))<channel_face[:,channel].mean():
        obj_index = np.where(np.mean(Conv5_all_noface[1:,:,channel],axis=1)==np.max(np.mean(Conv5_all_noface[1:,:,channel],axis=1)))[0].item() + 1
        u_statistic, p_value = stats.mannwhitneyu(channel_face[:,channel], Conv5_all_noface[obj_index,:,channel])
        if p_value<0.05/3:
            print('Channel:', channel, ' ',
                  'p_value:', p_value, ' ',
                  'Ratio:', channel_face[:,channel].mean()/np.max(np.mean(Conv5_all_noface[1:,:,channel],axis=1)))
            face_channel.append(channel) 
            
with open('/home/dell/Desktop/DNN2Brain/Temp_results/DN_face_channel.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    Activation = np.vstack((np.mean(channel_face, axis=0), np.mean(Conv5_all_noface, axis=1)))
    csv_write.writerow(Activation[:,28].tolist())
    csv_write.writerow(Activation[:,49].tolist())

act_face_obj = np.mean(np.vstack((channel_face[np.newaxis, :], Conv5_all_noface)), axis=1)
sparse_facechannel1_net = np.mean(minmax(act_face_obj[:,28]))**2/np.mean(minmax(act_face_obj[:,28])**2)
sparse_facechannel2_net = np.mean(minmax(act_face_obj[:,49]))**2/np.mean(minmax(act_face_obj[:,49])**2)
sparse_conv5_face_net = np.mean(minmax(act_face_obj[0,:]))**2/np.mean(minmax(act_face_obj[0,:])**2)



## find face channel in Alexnet (face_channel)  (184,124,59,186)
conv5_face_alexnet = np.mean(channel_face_alexnet, axis=0)
obj_activation_alexnet = channel_class_noface_alexnet.T
face_obj_alexnet = stats.zscore(np.column_stack((conv5_face_alexnet, obj_activation_alexnet)), axis=1) 
np.savetxt("/home/dell/Desktop/CN_tuning.csv", face_obj_alexnet, delimiter=',')

            
# use Mann-U test to select face channel
face_channel_alexnet = []
for channel in range(256):
    if np.max(np.mean(alexConv5_all_noface[1:,:,channel],axis=1))<channel_face_alexnet[:,channel].mean():
        obj_index = np.where(np.mean(alexConv5_all_noface[1:,:,channel],axis=1)==np.max(np.mean(alexConv5_all_noface[1:,:,channel],axis=1)))[0].item() + 1
        u_statistic, p_value = stats.mannwhitneyu(channel_face_alexnet[:,channel], alexConv5_all_noface[obj_index,:,channel])
        if p_value<0.05/5:
            print('Channel:', channel, ' ',
                  'p_value:', p_value, ' ',
                  'Ratio:', channel_face_alexnet[:,channel].mean()/np.max(np.mean(alexConv5_all_noface[1:,:,channel],axis=1)))
            face_channel_alexnet.append(channel) 

with open('/home/dell/Desktop/DNN2Brain/Temp_results/CN_face_channel.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    Activation = np.vstack((np.mean(channel_face_alexnet, axis=0), np.mean(alexConv5_all_noface, axis=1)))
    csv_write.writerow(Activation[:,184].tolist())
    csv_write.writerow(Activation[:,124].tolist())
    csv_write.writerow(Activation[:,59].tolist())
    csv_write.writerow(Activation[:,186].tolist())
          
Act_face_obj = np.mean(np.vstack((channel_face_alexnet[np.newaxis, :], alexConv5_all_noface)), axis=1)
sparse_facechannel1_alexnet = np.mean(minmax(Act_face_obj[:,184]))**2/np.mean(minmax(Act_face_obj[:,184])**2)
sparse_facechannel2_alexnet = np.mean(minmax(Act_face_obj[:,124]))**2/np.mean(minmax(Act_face_obj[:,124])**2)
sparse_facechannel3_alexnet = np.mean(minmax(Act_face_obj[:,59]))**2/np.mean(minmax(Act_face_obj[:,59])**2)
sparse_facechannel4_alexnet = np.mean(minmax(Act_face_obj[:,186]))**2/np.mean(minmax(Act_face_obj[:,186])**2)
sparse_conv5_face_alexnet = np.mean(minmax(Act_face_obj[0,:]))**2/np.mean(minmax(Act_face_obj[0,:])**2)

plt.figure()
plt.bar(np.arange(6), [sparse_facechannel1_net,
                       sparse_facechannel2_net,
                       sparse_facechannel1_alexnet,
                       sparse_facechannel2_alexnet,
                       sparse_facechannel3_alexnet,
                       sparse_facechannel4_alexnet])
plt.figure()
plt.bar(np.arange(2), [sparse_conv5_face_net, sparse_conv5_face_alexnet])

    
    
    
'9. What representation in Face channel'
###############################################################################
###############################################################################
# make the dict of 736label mapping to 1000label
f = pd.read_csv('/home/dell/Desktop/data_sony\win7\win10\dell/noface_train.csv')
stim_map_condition = f.set_index('stimID').to_dict()['condition']
convert_stim_map_condition = dict(zip(stim_map_condition.values(), stim_map_condition.keys()))
label736_map_label1000 = {}
for n in range(len(picdataset_train.picname)):
    pair = picdataset_train.get_picname(n)
    picname = pair[0][:9]+'/'+pair[0]
    label736_map_label1000[picdataset_train[n][1]] = tmp_dict_train[picname]
    
    
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import sys
sys.path.append('/home/dell/Desktop/My_module')
import reconstruction_layer
import Guided_CAM

# load picture [107, 99, 778, 556, 945]
for row,pic in enumerate([107, 99, 778, 556, 945]):
    p = plt.figure(figsize=(14,14) ,dpi=80) 
    number_of_picture = pic
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std = [ 0.229, 0.224, 0.225 ])])    
    #transform = transforms.Compose([
    #    transforms.Resize(224),
    #    transforms.CenterCrop((224,224)),
    #    transforms.ToTensor()])                          
    picdataset = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_face_1000_FITW.csv', transform=transform)
    picimg = picdataset[number_of_picture][0].unsqueeze(0)
    picimg.requires_grad=True
    ax = p.add_subplot(row+1,7,1)
    ax.imshow(picimg[0].permute(1,2,0).data.numpy())
    ax.axis('off')
        
    
    # load model
    non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
    net = copy.deepcopy(non_face_net).cpu()
    origin_alexnet = torchvision.models.alexnet(pretrained=True)
    net_alexnet = copy.deepcopy(origin_alexnet)
    
    # hypoparameters
    target_layer = 11
    target_class = 0
    
    ## Alexnet
    # model out and class
    p_alexnet = plt.figure(figsize=(12,12) ,dpi=80) 
    for n,channel in enumerate(face_channel_alexnet):
        # Guided backprop
        out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, picimg, target_layer, channel)
        #plt.imshow(out_image)
        
        # Grad cam
        gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
        cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
        #plt.imshow(gcv2.cam_channel)
        
        # Guided Grad cam
        cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
        ax = p.add_subplot(row+1,7,n+2)
        #ax.imshow(Guided_CAM.convert_to_grayscale(cam_gb, threshold=99)[0,:])
        ax.imshow(1.5*out_image.data.numpy(), alpha=1)
        ax.imshow(gcv2.cam_channel, alpha=0.5)
        ax.axis('off')    
    
    ## Non-face alexnet
    # model out and class
    p_net = plt.figure(figsize=(12,12) ,dpi=80) 
    for n,channel in enumerate(face_channel):
        # Guided backprop
        out_image = reconstruction_layer.layer_channel_reconstruction(net.cpu(), picimg, target_layer, channel)
        #plt.imshow(out_image)
        
        # Grad cam
        gcv2 = Guided_CAM.GradCam(net.cpu(), target_layer=target_layer)
        cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
        #plt.imshow(gcv2.cam_channel)
        
        # Guided Grad cam
        cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
        ax = p.add_subplot(row+1,7,n+6)
        #ax.imshow(Guided_CAM.convert_to_grayscale(cam_gb, threshold=99)[0,:])
        ax.imshow(1.5*out_image.data.numpy(), alpha=1)
        ax.imshow(gcv2.cam_channel, alpha=0.5)        
        ax.axis('off')
    


## weights of representation for face channel
# Alexnet
target_layer = 10
target_class = 0
Location_alexnet = np.array([])
for n in range(len(picdataset)):
    picimg = picdataset[n][0].unsqueeze(0)
    gcv2 = Guided_CAM.GradCam(net_alexnet, target_layer=target_layer)
    cam, weights = gcv2.generate_cam(picimg, target_class=target_class)
    for channel in face_channel_alexnet:
        location = np.max(np.where(np.sort(weights)==weights[channel])[0]).item()
        Location_alexnet = np.append(Location_alexnet, location)
Location_alexnet = Location_alexnet.reshape(-1, len(face_channel_alexnet))

    
# Non-face alexnet
target_layer = 10
target_class = 0
Location_nonface = np.array([])
for n in range(len(picdataset)):
    picimg = picdataset[n][0].unsqueeze(0)
    gcv2 = Guided_CAM.GradCam(net, target_layer=target_layer)
    cam, weights = gcv2.generate_cam(picimg, target_class=target_class)
    for channel in face_channel:
        location = np.max(np.where(np.sort(weights)==weights[channel])[0]).item()
        Location_nonface = np.append(Location_nonface, location)
Location_nonface = Location_nonface.reshape(-1, len(face_channel))


plt.figure()
Location_alexnet_avg = np.mean(Location_alexnet, axis=0)
std_err = np.std(Location_alexnet, axis=0)
plt.bar(['59','66','93','124','184','186'], Location_alexnet_avg, yerr=std_err, label='Alexnet')
Location_nonface_avg = np.mean(Location_nonface, axis=0)
std_err = np.std(Location_nonface, axis=0)
plt.bar(['46','48','139','157'], Location_nonface_avg, yerr=std_err, label='Non-face alexnet')
plt.legend()
plt.xlabel('channel number')
plt.ylabel('The importance of weight in this channel')
plt.title('Compare face channels in Alexnet and Non-face alexnet')




'10. test the effct and model'
###############################################################################
###############################################################################
import pandas as pd
from scipy import stats
import copy
import csv

# obj(trained) simuli
Object_class = '500'
f = pd.read_csv('/home/dell/Desktop/data_sony\win7\win10\dell/noface_train.csv')
f_dict = f.set_index('stimID').to_dict()['condition']
with open('/home/dell/Desktop/data_sony\win7\win10\dell/test_effect/500th_class.csv', 'a+') as csvfile:
    writer = csv.writer(csvfile)
    picfolder = '/home/dell/Desktop/Imgnet/traindata/'+convert_dict_train[Object_class]+'/'
    writer.writerow([picfolder])
    writer.writerow(['stimID','condition'])
    tmp_pos = os.listdir(picfolder)
    for pic in tmp_pos:
        if pic[:9]+'/'+pic in f_dict:
            writer.writerow([pic, pic[:9]])
            
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
plt.figure()
plt.title('Alexnet')
plt.imshow(np.corrcoef(n03042490_alexnet))
plt.colorbar()

n03042490_net = []
for n,(img,_) in enumerate(picdataset_train):
    print(n)
    net_truncate.layeract(img.unsqueeze(0).to(device))
    n03042490_net.append(net_truncate.conv_output.cpu().data.numpy())
n03042490_net = np.array(n03042490_net)
plt.figure()
plt.title('Non-face alexnet')
plt.imshow(np.corrcoef(n03042490_net))
plt.colorbar()


# obj(nontrained) activation in Conv5
Object_nontrain = '018'
class_alexnet = []
for index, (img,_) in enumerate(picdataset_nontrain):
    print(index)
    if picdataset_nontrain.picname[index][:3]==Object_nontrain:
        alexnet_truncate.layeract(img.unsqueeze(0).to(device))
        class_alexnet.append(alexnet_truncate.conv_output.cpu().data.numpy())
class_alexnet = np.array(class_alexnet)
plt.figure()
plt.title('Alexnet')
plt.imshow(np.corrcoef(class_alexnet))
plt.colorbar()

class_net = []
for index, (img,_) in enumerate(picdataset_nontrain):
    print(index)
    if picdataset_nontrain.picname[index][:3]==Object_nontrain:
        net_truncate.layeract(img.unsqueeze(0).to(device))
        class_net.append(net_truncate.conv_output.cpu().data.numpy())
class_net = np.array(class_net)
plt.figure()
plt.title('Non-face alexnet')
plt.imshow(np.corrcoef(class_net))
plt.colorbar()


### 3 class RSA
plt.figure(figsize=(6,6), dpi=80)
plt.imshow(np.corrcoef(np.vstack((channel_face_alexnet[:80], n03042490_alexnet[:80], class_alexnet))))
plt.colorbar()
plt.axis('off')

plt.figure(figsize=(6,6), dpi=80)
plt.imshow(np.corrcoef(np.vstack((channel_face[:80], n03042490_net[:80], class_net))))
plt.colorbar()
plt.axis('off')



### Intra analysis
# Distrbution
R_face_alexnet = np.where(np.corrcoef(channel_face_alexnet[:80])==1, np.nan, np.corrcoef(channel_face_alexnet[:80]))
R_obj_trained_alexnet = np.where(np.corrcoef(n03042490_alexnet[:80])==1, np.nan, np.corrcoef(n03042490_alexnet[:80]))
R_obj_nontrained_alexnet = np.where(np.corrcoef(class_alexnet)==1, np.nan, np.corrcoef(class_alexnet))
R_face_net = np.where(np.corrcoef(channel_face[:80])==1, np.nan, np.corrcoef(channel_face[:80]))
R_obj_trained_net = np.where(np.corrcoef(n03042490_net[:80])==1, np.nan, np.corrcoef(n03042490_net[:80]))
R_obj_nontrained_net = np.where(np.corrcoef(class_net)==1, np.nan, np.corrcoef(class_net))

bins = 80
alpha = 0.4
plt.figure()
plt.title('Alexnet distribution')
plt.xlabel('R (intra pictures)')
plt.ylabel('Number')
n, bins, patches = plt.hist(R_face_alexnet.reshape(-1,1), bins=bins, label='Face', alpha=alpha, color='b')
plt.plot(bins[:-1], n, 'b--', linewidth=1)
n, bins, patches = plt.hist(R_obj_trained_alexnet.reshape(-1,1), bins=bins, label='Obj(trained)', alpha=alpha ,color='g')
plt.plot(bins[:-1], n, 'g--', linewidth=1)
n, bins, patches = plt.hist(R_obj_nontrained_alexnet.reshape(-1,1), bins=bins, label='Obj(nontrained)', alpha=alpha, color='r')
plt.plot(bins[:-1], n, 'r--', linewidth=1)
plt.legend()
plt.figure()
plt.title('Non-face alexnet distribution')
plt.xlabel('R (intra pictures)')
plt.ylabel('Number')
n, bins, patches = plt.hist(R_face_net.reshape(-1,1), bins=bins, label='Face', alpha=alpha, color='b')
plt.plot(bins[:-1], n, 'b--', linewidth=1)
n, bins, patches = plt.hist(R_obj_trained_net.reshape(-1,1), bins=bins, label='Obj(trained)', alpha=alpha, color='g')
plt.plot(bins[:-1], n, 'g--', linewidth=1)
n, bins, patches = plt.hist(R_obj_nontrained_net.reshape(-1,1), bins=bins, label='Obj(nontrained)', alpha=alpha, color='r')
plt.plot(bins[:-1], n, 'r--', linewidth=1)
plt.legend()

# Bar
def lower_triangular(R):  
    global r
    r = np.array([])     
    for i in range(1,R.shape[0]):
        for j in range(i):
            r = np.append(r, R[i,j])
    return r
            
plt.figure()
plt.title('Intra compare')
plt.xlabel('R (intra pictures)')
plt.ylabel('Mean of R')
plt.ylim((0,0.7))
bar_width=0.3
index = np.arange(2) 
plt.bar(index, [np.mean(lower_triangular(R_face_alexnet)), 
                #np.mean(lower_triangular(R_obj_trained_alexnet)), 
                np.mean(lower_triangular(R_obj_nontrained_alexnet))],
        yerr=[np.std(lower_triangular(R_face_alexnet))/np.sqrt(lower_triangular(R_face_alexnet).shape[0]),
              #np.std(lower_triangular(R_obj_trained_alexnet))/np.sqrt(lower_triangular(R_obj_trained_alexnet).shape[0]),
              np.std(lower_triangular(R_obj_nontrained_alexnet))/np.sqrt(lower_triangular(R_obj_nontrained_alexnet).shape[0])],
        width=0.3, label='Alexnet')
plt.bar(index+bar_width, [np.mean(lower_triangular(R_face_net)), 
                          #np.mean(lower_triangular(R_obj_trained_net)), 
                          np.mean(lower_triangular(R_obj_nontrained_net))],
        yerr=[np.std(lower_triangular(R_face_net))/np.sqrt(lower_triangular(R_face_net).shape[0]),
              #np.std(lower_triangular(R_obj_trained_net))/np.sqrt(lower_triangular(R_obj_trained_net).shape[0]),
              np.std(lower_triangular(R_obj_nontrained_net))/np.sqrt(lower_triangular(R_obj_nontrained_net).shape[0])],
        width=0.3, label='Non-face alexnet')
plt.legend() 
plt.xticks([])

with open('/home/dell/Desktop/DNN2Brain/Temp_results/Intra.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for i in range(3160):
        csv_write.writerow(['face', lower_triangular(R_face_alexnet)[i], lower_triangular(R_face_net)[i]])
    for i in range(3160):
        csv_write.writerow(['unseen', lower_triangular(R_obj_nontrained_alexnet)[i], lower_triangular(R_obj_nontrained_net)[i]])    



# t test
from scipy.stats import ttest_ind
import statsmodels.api as sm

def Fisher_z(r):
    return 0.5*np.log((1+r)/(1-r))

def R_ttest(A,B):  
    #global a,b
    a = np.array([])     
    for i in range(1,A.shape[0]):
        for j in range(i):
            a = np.append(a, Fisher_z(A[i,j]))
    b = np.array([])     
    for i in range(1,B.shape[0]):
        for j in range(i):
            b = np.append(b, Fisher_z(B[i,j]))
    #return a,b
    return ttest_ind(a,b,equal_var = True)
    

np.nanmean(Fisher_z(R_face_alexnet)), np.nanmean(Fisher_z(R_obj_trained_alexnet)), np.nanmean(Fisher_z(R_obj_nontrained_alexnet))
np.nanstd(Fisher_z(R_face_alexnet)),np.nanstd(Fisher_z(R_obj_trained_alexnet)),np.nanstd(Fisher_z(R_obj_nontrained_alexnet))
np.nanmean(Fisher_z(R_face_net)), np.nanmean(Fisher_z(R_obj_trained_net)), np.nanmean(Fisher_z(R_obj_nontrained_net))
np.nanstd(Fisher_z(R_face_net)),np.nanstd(Fisher_z(R_obj_trained_net)),np.nanstd(Fisher_z(R_obj_nontrained_net))

print(R_ttest(R_face_alexnet, R_face_net))
print(R_ttest(R_obj_trained_alexnet, R_obj_trained_net))
print(R_ttest(R_obj_nontrained_alexnet, R_obj_nontrained_net))

R_ttest(R_face_alexnet, R_obj_trained_alexnet)
R_ttest(R_face_alexnet, R_obj_nontrained_alexnet)
R_ttest(R_obj_trained_alexnet, R_obj_nontrained_alexnet)

R_ttest(R_face_net, R_obj_trained_net)
print(R_ttest(R_face_net, R_obj_nontrained_net))
R_ttest(R_obj_trained_net, R_obj_nontrained_net)


with open('/home/dell/Desktop/temp.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for i in range(Fisher_z(lower_triangular(R_face_alexnet)).shape[0]):
        csv_write.writerow(['Alexnet', 'Face', Fisher_z(lower_triangular(R_face_alexnet))[i]])
        csv_write.writerow(['Alexnet', 'SeenObj', Fisher_z(lower_triangular(R_obj_trained_alexnet))[i]])
        csv_write.writerow(['Alexnet', 'NonSeenObj', Fisher_z(lower_triangular(R_obj_nontrained_alexnet))[i]])
    for i in range(Fisher_z(lower_triangular(R_face_net)).shape[0]):
        csv_write.writerow(['Nonface_alexnet', 'Face', Fisher_z(lower_triangular(R_face_net))[i]])
        csv_write.writerow(['Nonface_alexnet', 'SeenObj', Fisher_z(lower_triangular(R_obj_trained_net))[i]])   
        csv_write.writerow(['Nonface_alexnet', 'NonSeenObj', Fisher_z(lower_triangular(R_obj_nontrained_net))[i]])
    

R_ttest(R_face_alexnet-R_face_net, R_obj_trained_alexnet-R_obj_trained_net)



### Inter analysis
# Distrbution
corr_alexnet = np.corrcoef(np.vstack((channel_face_alexnet[:80], n03042490_alexnet[:80], class_alexnet)))
corr_net = np.corrcoef(np.vstack((channel_face[:80], n03042490_net[:80], class_net)))

face_objtrained_alexnet = corr_alexnet[0:80,80:160]
face_objnontrained_alexnet = corr_alexnet[0:80,160:]
objtrained_objnontrained_alexnet = corr_alexnet[80:160,160:]
face_objtrained_net = corr_net[0:80,80:160]
face_objnontrained_net = corr_net[0:80,160:]
objtrained_objnontrained_net = corr_net[80:160,160:]

bins = 80
alpha = 0.4
plt.figure()
plt.title('Alexnet distribution')
plt.xlabel('R (inter pictures)')
plt.ylabel('Number')
n, bins, patches = plt.hist(face_objtrained_alexnet.reshape(-1,1), bins=bins, label='face_objtrained', alpha=alpha, color='b')
plt.plot(bins[:-1], n, 'b--', linewidth=1)
n, bins, patches = plt.hist(face_objnontrained_alexnet.reshape(-1,1), bins=bins, label='face_objnontrained', alpha=alpha ,color='g')
plt.plot(bins[:-1], n, 'g--', linewidth=1)
n, bins, patches = plt.hist(objtrained_objnontrained_alexnet.reshape(-1,1), bins=bins, label='objtrained_objnontrained', alpha=alpha, color='r')
plt.plot(bins[:-1], n, 'r--', linewidth=1)
plt.legend()
plt.figure()
plt.title('Non-face alexnet distribution')
plt.xlabel('R (inter pictures)')
plt.ylabel('Number')
n, bins, patches = plt.hist(face_objtrained_net.reshape(-1,1), bins=bins, label='face_objtrained_net', alpha=alpha, color='b')
plt.plot(bins[:-1], n, 'b--', linewidth=1)
n, bins, patches = plt.hist(face_objnontrained_net.reshape(-1,1), bins=bins, label='face_objnontrained_net', alpha=alpha, color='g')
plt.plot(bins[:-1], n, 'g--', linewidth=1)
n, bins, patches = plt.hist(objtrained_objnontrained_net.reshape(-1,1), bins=bins, label='objtrained_objnontrained_net', alpha=alpha, color='r')
plt.plot(bins[:-1], n, 'r--', linewidth=1)
plt.legend()

# Bar
plt.figure()
plt.title('Inter compare')
plt.xlabel('R (intra pictures)')
plt.ylabel('Mean of R')
plt.ylim((-0.15,0.3))
bar_width=0.3
index = np.arange(2) 
plt.bar(index, [#np.mean(lower_triangular(face_objtrained_alexnet)), 
                np.mean(lower_triangular(face_objnontrained_alexnet)), 
                np.mean(lower_triangular(objtrained_objnontrained_alexnet))],
        yerr=[#np.std(lower_triangular(face_objtrained_alexnet))/np.sqrt(lower_triangular(face_objtrained_alexnet).shape[0]),
              np.std(lower_triangular(face_objnontrained_alexnet))/np.sqrt(lower_triangular(face_objnontrained_alexnet).shape[0]),
              np.std(lower_triangular(objtrained_objnontrained_alexnet))/np.sqrt(lower_triangular(objtrained_objnontrained_alexnet).shape[0])],
        width=0.3, label='Alexnet')
plt.bar(index+bar_width, [#np.mean(lower_triangular(face_objtrained_net)), 
                          np.mean(lower_triangular(face_objnontrained_net)), 
                          np.mean(lower_triangular(objtrained_objnontrained_net))],
        yerr=[#np.std(lower_triangular(face_objtrained_net))/np.sqrt(lower_triangular(face_objtrained_net).shape[0]),
              np.std(lower_triangular(face_objnontrained_net))/np.sqrt(lower_triangular(face_objnontrained_net).shape[0]),
              np.std(lower_triangular(objtrained_objnontrained_net))/np.sqrt(lower_triangular(objtrained_objnontrained_net).shape[0])],
        width=0.3, label='Non-face alexnet')
plt.legend() 
plt.xticks([])


with open('/home/dell/Desktop/DNN2Brain/Temp_results/Inter.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for i in range(3160):
        csv_write.writerow(['face_unseen', 
                            lower_triangular(face_objnontrained_alexnet)[i], 
                            lower_triangular(face_objnontrained_net)[i]])
    for i in range(3160):
        csv_write.writerow(['obj_unseen', 
                            lower_triangular(objtrained_objnontrained_alexnet)[i], 
                            lower_triangular(objtrained_objnontrained_net)[i]])    


print(R_ttest(face_objtrained_alexnet, face_objtrained_net))
print(R_ttest(face_objnontrained_alexnet, face_objnontrained_net))
print(R_ttest(objtrained_objnontrained_alexnet, objtrained_objnontrained_net))
print(R_ttest(face_objtrained_alexnet-face_objtrained_net, face_objnontrained_alexnet-face_objnontrained_net))
print(R_ttest(face_objnontrained_alexnet-face_objnontrained_net, objtrained_objnontrained_alexnet-objtrained_objnontrained_net))


### Intra-Inter
plt.figure()
plt.title('Intra-Inter')
plt.ylabel('Mean of R')
plt.ylim((0,0.7))
bar_width=0.3
index = np.arange(6) 
plt.bar(index, [np.mean(lower_triangular(R_face_alexnet)-lower_triangular(face_objtrained_alexnet)), 
                np.mean(lower_triangular(R_face_alexnet)-lower_triangular(face_objnontrained_alexnet)), 
                np.mean(lower_triangular(R_obj_trained_alexnet)-lower_triangular(face_objtrained_alexnet)),
                np.mean(lower_triangular(R_obj_trained_alexnet)-lower_triangular(objtrained_objnontrained_alexnet)),
                np.mean(lower_triangular(R_obj_nontrained_alexnet)-lower_triangular(face_objnontrained_alexnet)),
                np.mean(lower_triangular(R_obj_nontrained_alexnet)-lower_triangular(objtrained_objnontrained_alexnet))],
        yerr=[np.std(lower_triangular(R_face_alexnet)-lower_triangular(face_objtrained_alexnet))/np.sqrt(lower_triangular(face_objtrained_alexnet).shape[0]),
              np.std(lower_triangular(R_face_alexnet)-lower_triangular(face_objnontrained_alexnet))/np.sqrt(lower_triangular(face_objnontrained_alexnet).shape[0]),
              np.std(lower_triangular(R_obj_trained_alexnet)-lower_triangular(face_objtrained_alexnet))/np.sqrt(lower_triangular(objtrained_objnontrained_alexnet).shape[0]),
              np.std(lower_triangular(R_obj_trained_alexnet)-lower_triangular(objtrained_objnontrained_alexnet))/np.sqrt(lower_triangular(face_objtrained_alexnet).shape[0]),
              np.std(lower_triangular(R_obj_nontrained_alexnet)-lower_triangular(face_objnontrained_alexnet))/np.sqrt(lower_triangular(face_objtrained_alexnet).shape[0]),
              np.std(lower_triangular(R_obj_nontrained_alexnet)-lower_triangular(objtrained_objnontrained_alexnet))/np.sqrt(lower_triangular(face_objtrained_alexnet).shape[0])],
        width=0.3, label='Alexnet')
plt.bar(index+bar_width, [np.mean(lower_triangular(R_face_net)-lower_triangular(face_objtrained_net)), 
                np.mean(lower_triangular(R_face_net)-lower_triangular(face_objnontrained_net)), 
                np.mean(lower_triangular(R_obj_trained_net)-lower_triangular(face_objtrained_net)),
                np.mean(lower_triangular(R_obj_trained_net)-lower_triangular(objtrained_objnontrained_net)),
                np.mean(lower_triangular(R_obj_nontrained_net)-lower_triangular(face_objnontrained_net)),
                np.mean(lower_triangular(R_obj_nontrained_net)-lower_triangular(objtrained_objnontrained_net))],
        yerr=[np.std(lower_triangular(R_face_net)-lower_triangular(face_objtrained_net))/np.sqrt(lower_triangular(face_objtrained_alexnet).shape[0]),
              np.std(lower_triangular(R_face_net)-lower_triangular(face_objnontrained_net))/np.sqrt(lower_triangular(face_objnontrained_alexnet).shape[0]),
              np.std(lower_triangular(R_obj_trained_net)-lower_triangular(face_objtrained_net))/np.sqrt(lower_triangular(objtrained_objnontrained_alexnet).shape[0]),
              np.std(lower_triangular(R_obj_trained_net)-lower_triangular(objtrained_objnontrained_net))/np.sqrt(lower_triangular(face_objtrained_alexnet).shape[0]),
              np.std(lower_triangular(R_obj_nontrained_net)-lower_triangular(face_objnontrained_net))/np.sqrt(lower_triangular(face_objtrained_alexnet).shape[0]),
              np.std(lower_triangular(R_obj_nontrained_net)-lower_triangular(objtrained_objnontrained_net))/np.sqrt(lower_triangular(face_objtrained_alexnet).shape[0])],
        width=0.3, label='Non-face alexnet')
plt.legend() 
plt.xticks([])

print(stats.ttest_1samp(lower_triangular(R_face_alexnet-face_objtrained_alexnet), 0))
print(stats.ttest_1samp(lower_triangular(R_face_alexnet-face_objnontrained_alexnet),0))
print(stats.ttest_1samp(lower_triangular(R_obj_trained_alexnet-face_objtrained_alexnet), 0))
print(stats.ttest_1samp(lower_triangular(R_obj_trained_alexnet-objtrained_objnontrained_alexnet), 0))
print(stats.ttest_1samp(lower_triangular(R_obj_nontrained_alexnet-face_objnontrained_alexnet), 0))
print(stats.ttest_1samp(lower_triangular(R_obj_nontrained_alexnet-objtrained_objnontrained_alexnet), 0))




######################
'Behavior level test'
######################
'''
Face: picdataset   [len(picdataset)=1000]
Obj(trained): picdataset_train   [len(picdataset_train)=889]
Obj(nontrained): picdataset_nontrain   [len(picdataset_nontrain)=16320]
alexnet
net
'''
## Alexnet
face_alexnet = np.array([])
face_alexnet_top1 = np.array([])
count = 0
for img,target in picdataset:
    outputs = alexnet(img.unsqueeze(0).to(device))
    face_alexnet = np.append(face_alexnet, outputs.cpu().data.numpy())
    count += 1
    _, pred = torch.max(outputs, 1)
    face_alexnet_top1 = np.append(face_alexnet_top1, pred.cpu().item())
    if count==80:
        break
face_alexnet = face_alexnet.reshape(80,-1)
    
objtrained_alexnet = np.array([])
objtrained_alexnet_top1 = np.array([])
count = 0
for img,target in picdataset_train:
    outputs = alexnet(img.unsqueeze(0).to(device))
    objtrained_alexnet = np.append(objtrained_alexnet, outputs.cpu().data.numpy())
    count += 1
    _, pred = torch.max(outputs, 1)
    objtrained_alexnet_top1 = np.append(objtrained_alexnet_top1, pred.cpu().item())
    if count==80:
        break
objtrained_alexnet = objtrained_alexnet.reshape(80,-1)

Object_nontrain = '018'
objnontrained_alexnet = np.array([])
objnontrained_alexnet_top1 = np.array([])
for index, (img,_) in enumerate(picdataset_nontrain):
    if picdataset_nontrain.picname[index][:3]==Object_nontrain:
        outputs = alexnet(img.unsqueeze(0).to(device))
        objnontrained_alexnet = np.append(objnontrained_alexnet, outputs.cpu().data.numpy())
        _, pred = torch.max(outputs, 1)
        objnontrained_alexnet_top1 = np.append(objnontrained_alexnet_top1, pred.cpu().item())
objnontrained_alexnet = objnontrained_alexnet.reshape(80,-1)

# See the class
plt.figure()
plt.title('Alexnet:face_alexnet_top1')
plt.hist(face_alexnet_top1, bins=50)
plt.figure()
plt.title('Alexnet:objtrained_alexnet_top1')
plt.hist(objtrained_alexnet_top1, bins=50)
plt.figure()
plt.title('Alexnet:objnontrained_alexnet_top1')
plt.hist(objnontrained_alexnet_top1, bins=50)

# See the RSA of behavior
plt.figure()
plt.title("Alexnet: face, obj(trained), obj(nontrained)")
plt.imshow(np.corrcoef(np.vstack((face_alexnet[:80], objtrained_alexnet[:80], objnontrained_alexnet))))
plt.colorbar()
plt.axis('off')


## net
face_net = np.array([])
face_net_top1 = np.array([])
count = 0
for img,target in picdataset:
    outputs = net(img.unsqueeze(0).to(device))
    face_net = np.append(face_net, outputs.cpu().data.numpy())
    count += 1
    _, pred = torch.max(outputs, 1)
    face_net_top1 = np.append(face_net_top1, pred.cpu().item())
    if count==80:
        break
face_net = face_net.reshape(80,-1)
    
objtrained_net = np.array([])
objtrained_net_top1 = np.array([])
count = 0
for img,target in picdataset_train:
    outputs = net(img.unsqueeze(0).to(device))
    objtrained_net = np.append(objtrained_net, outputs.cpu().data.numpy())
    count += 1
    _, pred = torch.max(outputs, 1)
    objtrained_net_top1 = np.append(objtrained_net_top1, pred.cpu().item())
    if count==80:
        break
objtrained_net = objtrained_net.reshape(80,-1)

Object_nontrain = '018'
objnontrained_net = np.array([])
objnontrained_net_top1 = np.array([])
for index, (img,_) in enumerate(picdataset_nontrain):
    if picdataset_nontrain.picname[index][:3]==Object_nontrain:
        outputs = net(img.unsqueeze(0).to(device))
        objnontrained_net = np.append(objnontrained_net, outputs.cpu().data.numpy())
        _, pred = torch.max(outputs, 1)
        objnontrained_net_top1 = np.append(objnontrained_net_top1, pred.cpu().item())    
objnontrained_net = objnontrained_net.reshape(80,-1)
   
# See the class     
plt.figure()
plt.title('Nonface alexnet:face_net_top1')
plt.hist(face_net_top1, bins=50)
plt.figure()
plt.title('Nonface alexnet:objtrained_net_top1')
plt.hist(objtrained_net_top1, bins=50)
plt.figure()
plt.title('Nonface alexnet:objnontrained_net_top1')
plt.hist(objnontrained_net_top1, bins=50)

# See the RSA of behavior
plt.figure()
plt.title("Non-face alexnet: face, obj(trained), obj(nontrained)")
plt.imshow(np.corrcoef(np.vstack((face_net[:80], objtrained_net[:80], objnontrained_net))))
plt.colorbar()
plt.axis('off')


### Intra analysis
# Distrbution
R_face_alexnet = np.where(np.corrcoef(face_alexnet[:80])==1, np.nan, np.corrcoef(face_alexnet[:80]))
R_obj_trained_alexnet = np.where(np.corrcoef(objtrained_alexnet[:80])==1, np.nan, np.corrcoef(objtrained_alexnet[:80]))
R_obj_nontrained_alexnet = np.where(np.corrcoef(objnontrained_alexnet)==1, np.nan, np.corrcoef(objnontrained_alexnet))
R_face_net = np.where(np.corrcoef(face_net[:80])==1, np.nan, np.corrcoef(face_net[:80]))
R_obj_trained_net = np.where(np.corrcoef(objtrained_net[:80])==1, np.nan, np.corrcoef(objtrained_net[:80]))
R_obj_nontrained_net = np.where(np.corrcoef(objnontrained_net)==1, np.nan, np.corrcoef(objnontrained_net))

bins = 80
alpha = 0.4
plt.figure()
plt.title('Alexnet distribution')
plt.xlabel('R (intra pictures)')
plt.ylabel('Number')
n1, bins1, patches = plt.hist(R_face_alexnet.reshape(-1,1), bins=bins, label='Face', alpha=alpha, color='b')
plt.plot(bins1[:-1], n1, 'b--', linewidth=1, label='Face')
n2, bins2, patches = plt.hist(R_obj_trained_alexnet.reshape(-1,1), bins=bins, label='Obj(trained)', alpha=alpha ,color='g')
plt.plot(bins2[:-1], n2, 'g--', linewidth=1, label='Obj(trained)')
n3, bins3, patches = plt.hist(R_obj_nontrained_alexnet.reshape(-1,1), bins=bins, label='Obj(nontrained)', alpha=alpha, color='r')
plt.plot(bins3[:-1], n3, 'r--', linewidth=1, label='Obj(nontrained)')
plt.legend()
plt.figure()
plt.title('Non-face alexnet distribution')
plt.xlabel('R (intra pictures)')
plt.ylabel('Number')
n1, bins1, patches = plt.hist(R_face_net.reshape(-1,1), bins=bins, label='Face', alpha=alpha, color='b')
plt.plot(bins1[:-1], n1, 'b--', linewidth=1, label='Face')
n2, bins2, patches = plt.hist(R_obj_trained_net.reshape(-1,1), bins=bins, label='Obj(trained)', alpha=alpha, color='g')
plt.plot(bins2[:-1], n2, 'g--', linewidth=1, label='Obj(trained)')
n3, bins3, patches = plt.hist(R_obj_nontrained_net.reshape(-1,1), bins=bins, label='Obj(nontrained)', alpha=alpha, color='r')
plt.plot(bins3[:-1], n3, 'r--', linewidth=1, label='Obj(nontrained)')
plt.legend()

# Bar
plt.figure()
plt.xlabel('R (intra pictures)')
plt.ylabel('Mean of R')
plt.ylim((0,0.7))
bar_width=0.3
index = np.arange(3) 
plt.bar(index, [np.mean(lower_triangular(R_face_alexnet)), 
                np.mean(lower_triangular(R_obj_trained_alexnet)), 
                np.mean(lower_triangular(R_obj_nontrained_alexnet))],
        yerr=[np.std(lower_triangular(R_face_alexnet))/R_face_alexnet.shape[0],
              np.std(lower_triangular(R_obj_trained_alexnet))/R_face_alexnet.shape[0],
              np.std(lower_triangular(R_obj_nontrained_alexnet))/R_face_alexnet.shape[0]],
        width=0.3, label='Alexnet')
plt.bar(index+bar_width, [np.mean(lower_triangular(R_face_net)), 
                          np.mean(lower_triangular(R_obj_trained_net)), 
                          np.mean(lower_triangular(R_obj_nontrained_net))],
        yerr=[np.std(lower_triangular(R_face_net))/R_face_alexnet.shape[0],
              np.std(lower_triangular(R_obj_trained_net))/R_face_alexnet.shape[0],
              np.std(lower_triangular(R_obj_nontrained_net))/R_face_alexnet.shape[0]],
        width=0.3, label='Non-face alexnet')
plt.legend() 
plt.xticks([])

plt.figure()
plt.ylabel('Mean of R')
plt.bar(np.arange(2),[np.mean(lower_triangular(R_face_net)),np.mean(lower_triangular(R_obj_nontrained_net))],color='orange')
plt.xticks([])


# t test
print(R_ttest(R_face_alexnet, R_face_net))
print(R_ttest(R_obj_trained_alexnet, R_obj_trained_net))
print(R_ttest(R_obj_nontrained_alexnet, R_obj_nontrained_net))



######################
'Discrimination MVPA'
######################
import torch.nn as nn
import copy
import os
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader     
import torchvision.transforms as transforms
import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import io as dnn_io
from dnnbrain.dnn import models
import shutil
import random
import time
import csv


f = os.listdir('/home/dell/Desktop/Imgnet/casiawebface_224x224')
folder = []
for i in f:
    picdir = '/home/dell/Desktop/Imgnet/casiawebface_224x224/'+i
    if len(os.listdir(picdir))>300:
        folder.append(i)
    else:
        shutil.rmtree(picdir, True)


f = os.listdir('/home/dell/Desktop/Imgnet/casiawebface_224x224')
with open('/home/dell/Desktop/Imgnet/casiawebface_224x224/train.csv', 'a+') as train:
    with open('/home/dell/Desktop/Imgnet/casiawebface_224x224/val.csv', 'a+') as val:
        train_write = csv.writer(train)
        val_write = csv.writer(val)
        train_write.writerow(['/home/dell/Desktop/Imgnet/casiawebface_224x224/'])
        train_write.writerow(['stimID', 'condition'])
        val_write.writerow(['/home/dell/Desktop/Imgnet/casiawebface_224x224/'])
        val_write.writerow(['stimID', 'condition'])
        for i in f:
            picdir = '/home/dell/Desktop/Imgnet/casiawebface_224x224/'+i
            temp = os.listdir(picdir)
            select_pic = random.sample(temp, 300)
            traindata = random.sample(select_pic, 250)
            for j in select_pic:
                if j in traindata:
                    name = i+'/'+j
                    train_write.writerow([name, i])
                else:
                    name = i+'/'+j
                    val_write.writerow([name, i])
                    
   
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
non_face_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
net = copy.deepcopy(non_face_net)
origin_alexnet = torchvision.models.alexnet(pretrained=True).to(device)
alexnet = copy.deepcopy(origin_alexnet)

                 
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
picdataset_train = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/casiawebface_224x224/train.csv', transform=data_transforms['train'])
dataloader_train = DataLoader(picdataset_train, batch_size=128, shuffle=True, num_workers=10)

picdataset_train_test = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/casiawebface_224x224/train.csv', transform=data_transforms['val'])
dataloader_train_test = DataLoader(picdataset_train_test, batch_size=128, shuffle=True, num_workers=10)

picdataset_val_test = dnn_io.PicDataset('/home/dell/Desktop/Imgnet/casiawebface_224x224/val.csv', transform=data_transforms['val'])
dataloader_val_test = DataLoader(picdataset_val_test, batch_size=128, shuffle=False, num_workers=10)


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
            self.conv_output = torch.mean(layer_out, (2,3))
        self.handle = self.model.features[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if index == self.selected_layer:
                break
        self.handle.remove()

selected_layer = 11
net_truncate = NET(net, selected_layer=selected_layer)
alexnet_truncate = NET(alexnet, selected_layer=selected_layer)


# Alexnet (discriminator:ANN after relu)
def test(dataloader, discriminator, network_truncate):     
    ACC = np.zeros((1,133))
    time0 = time.time()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        network_truncate.layeract(inputs)
        truncate = network_truncate.feature_map.view(-1,256*13*13)
        out = discriminator.forward(truncate) 
        _, outputs = torch.max(out, 1)
        for i in range(133):
            location = torch.where(targets==i)[0]
            acc = torch.sum(outputs[location]==targets[location])
            acc = acc.item()
            ACC[0, i] = ACC[0, i] + acc
    time1 = time.time()
    print(time1-time0, 's')
    return ACC

discriminator = torch.nn.Sequential(torch.nn.Linear(256*13*13, 4096, bias=True),
                                    torch.nn.Sigmoid(),
                                    torch.nn.Linear(4096, 133, bias=True)).to(device) 
optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
loss_function = torch.nn.CrossEntropyLoss()  
num_epoches = 100
metric_path = '/home/dell/Desktop/205cat/discrimination/Alexnet/Metric.csv'
f = open(metric_path, 'a+')
csv_write = csv.writer(f)
csv_write.writerow(['EPOCH', 'ACC_train_test', 'ACC_val_test'])
f.close()
for epoch in range(num_epoches):  
    print('Epoch step {}/{}'.format(epoch+1, num_epoches))
    for inputs, targets in dataloader_train:
        inputs = inputs.to(device)
        targets = targets.to(device)
        alexnet_truncate.layeract(inputs)
        truncate = alexnet_truncate.feature_map.view(-1,256*13*13)
        out = discriminator.forward(truncate)  
        loss = loss_function(out, targets) 
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()
    ACC_train_test = test(dataloader_train_test, discriminator, alexnet_truncate)
    ACC_train_test = np.mean(ACC_train_test/250)
    ACC_val_test = test(dataloader_val_test, discriminator, alexnet_truncate)
    ACC_val_test = np.mean(ACC_val_test/50)
    with open(metric_path,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([epoch, ACC_train_test, ACC_val_test])
    modelname = '/home/dell/Desktop/205cat/discrimination/Alexnet/' + 'Alexnet_epoch' + str(epoch)
    torch.save(discriminator, modelname)


# Nonface_net (discriminator:ANN after relu)
def test(dataloader, discriminator, network_truncate):     
    ACC = np.zeros((1,133))
    time0 = time.time()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        network_truncate.layeract(inputs)
        truncate = network_truncate.feature_map.view(-1,256*13*13)
        out = discriminator.forward(truncate) 
        _, outputs = torch.max(out, 1)
        for i in range(133):
            location = torch.where(targets==i)[0]
            acc = torch.sum(outputs[location]==targets[location])
            acc = acc.item()
            ACC[0, i] = ACC[0, i] + acc
    time1 = time.time()
    print(time1-time0, 's')
    return ACC

discriminator = torch.nn.Sequential(torch.nn.Linear(256*13*13, 4096, bias=True),
                                    torch.nn.Sigmoid(),
                                    torch.nn.Linear(4096, 133, bias=True)).to(device) 
optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
loss_function = torch.nn.CrossEntropyLoss()  
num_epoches = 100
metric_path = '/home/dell/Desktop/205cat/discrimination/net/Metric.csv'
f = open(metric_path, 'a+')
csv_write = csv.writer(f)
csv_write.writerow(['EPOCH', 'ACC_train_test', 'ACC_val_test'])
f.close()
for epoch in range(num_epoches):  
    print('Epoch step {}/{}'.format(epoch+1, num_epoches))
    for inputs, targets in dataloader_train:
        inputs = inputs.to(device)
        targets = targets.to(device)
        net_truncate.layeract(inputs)
        truncate = net_truncate.feature_map.view(-1,256*13*13)
        out = discriminator.forward(truncate)  
        loss = loss_function(out, targets) 
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()
    ACC_train_test = test(dataloader_train_test, discriminator, net_truncate)
    ACC_train_test = np.mean(ACC_train_test/250)
    ACC_val_test = test(dataloader_val_test, discriminator, net_truncate)
    ACC_val_test = np.mean(ACC_val_test/50)
    with open(metric_path,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([epoch, ACC_train_test, ACC_val_test])
    modelname = '/home/dell/Desktop/205cat/discrimination/net/' + 'net_epoch' + str(epoch)
    torch.save(discriminator, modelname)
    
    
# plot test acc
f_alexnet = pd.read_csv('/home/dell/Desktop/205cat/discrimination/Alexnet/Metric.csv')    
f_net = pd.read_csv('/home/dell/Desktop/205cat/discrimination/net/Metric.csv')    

acc_alexnet = []
for i in range(len(f_alexnet)):
    acc_alexnet.append(float(f_alexnet.loc[i][2]))
acc_net = []
for i in range(len(f_net)):
    acc_net.append(float(f_net.loc[i][2]))
    
plt.figure()
plt.plot(acc_alexnet, label='Alexnet')
plt.plot(acc_net, label='Nonface_net')
plt.plot(np.array(acc_net)-np.array(acc_alexnet), label='difference')
plt.legend()


# the last step error bar
discriminator_alexnet = torch.load('/home/dell/Desktop/205cat/discrimination/Alexnet/Alexnet_epoch89')
discriminator_net = torch.load('/home/dell/Desktop/205cat/discrimination/net/net_epoch89')
acc_alexnet = test(dataloader_val_test, discriminator_alexnet, alexnet_truncate)/50
acc_net = test(dataloader_val_test, discriminator_net, net_truncate)/50
plt.figure()
plt.title('Compare discrimination')
#plt.ylim((0.6,0.7))
plt.bar(['Alexnet','Nonface_alexnet'], 
        [np.mean(acc_alexnet),np.mean(acc_net)], 
        yerr=[np.std(acc_alexnet)/np.sqrt(acc_alexnet.shape[1]),np.std(acc_net)/np.sqrt(acc_net.shape[1])])

from scipy.stats import ttest_1samp, ttest_rel
print('ACC_alexnet:', ttest_1samp(acc_alexnet.reshape(133), 1/133))
print('ACC_net:', ttest_1samp(acc_net.reshape(133), 1/133))
print('ACC_alexnet vs ACC_net:', ttest_rel(acc_alexnet.reshape(133), acc_net.reshape(133)))




######################
'Classification Conv5'
######################
import torch.nn as nn
import copy
import os
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader     
import torchvision.transforms as transforms
import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import io as dnn_io
from dnnbrain.dnn import models
import shutil
import random
import time
import csv


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
picdataset_train = dnn_io.PicDataset('/home/dell/Desktop/205cat/allstim_train.csv', transform=data_transforms['train'])
dataloader_train = DataLoader(picdataset_train, batch_size=128, shuffle=True, num_workers=20)

picdataset_train_test = dnn_io.PicDataset('/home/dell/Desktop/205cat/allstim_train.csv', transform=data_transforms['val'])
dataloader_train_test = DataLoader(picdataset_train_test, batch_size=128, shuffle=False, num_workers=20)

picdataset_val_test = dnn_io.PicDataset('/home/dell/Desktop/205cat/allstim_val.csv', transform=data_transforms['val'])
dataloader_val_test = DataLoader(picdataset_val_test, batch_size=128, shuffle=False, num_workers=20)


# load non-face net and alexnet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')
alexnet = torchvision.models.alexnet(pretrained=True).to(device)


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
            self.conv_output = torch.mean(layer_out, (2,3))
        self.model.features[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if index == self.selected_layer:
                break

selected_layer = 11
net_truncate = NET(net, selected_layer=selected_layer)
alexnet_truncate = NET(alexnet, selected_layer=selected_layer)


# Alexnet (classification after relu)
def test(dataloader, classifier, network_truncate):     
    ACC = np.zeros((1,205))
    time0 = time.time()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        network_truncate.layeract(inputs)
        truncate = network_truncate.feature_map.view(-1,256*13*13)
        out = classifier.forward(truncate) 
        _, outputs = torch.max(out, 1)
        for i in range(205):
            location = torch.where(targets==i)[0]
            acc = torch.sum(outputs[location]==targets[location])
            acc = acc.item()
            ACC[0, i] = ACC[0, i] + acc
    time1 = time.time()
    print(time1-time0, 's')
    return ACC

classifier = torch.nn.Sequential(torch.nn.Linear(256*13*13, 4096, bias=True),
                                 torch.nn.Sigmoid(),
                                 torch.nn.Linear(4096, 205, bias=True)).to(device) 
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)  
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
loss_function = torch.nn.CrossEntropyLoss()  
num_epoches = 100
metric_path = '/home/dell/Desktop/205cat/classification/Alexnet/Metric.csv'
f = open(metric_path, 'a+')
csv_write = csv.writer(f)
csv_write.writerow(['EPOCH', 'ACC_train_test', 'ACC_val_test'])
f.close()
for epoch in range(num_epoches):  
    print('Epoch step {}/{}'.format(epoch+1, num_epoches))
    for inputs, targets in dataloader_train:
        inputs = inputs.to(device)
        targets = targets.to(device)
        alexnet_truncate.layeract(inputs)
        truncate = alexnet_truncate.feature_map.view(-1,256*13*13)
        out = classifier.forward(truncate)  
        loss = loss_function(out, targets) 
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()
    ACC_train_test = test(dataloader_train_test, classifier, alexnet_truncate)
    ACC_train_test = np.mean(ACC_train_test/70)
    ACC_val_test = test(dataloader_val_test, classifier, alexnet_truncate)
    ACC_val_test = np.mean(ACC_val_test/10)
    with open(metric_path,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([epoch, ACC_train_test, ACC_val_test])
    modelname = '/home/dell/Desktop/205cat/classification/Alexnet/' + 'Alexnet_epoch' + str(epoch)
    torch.save(classifier, modelname)


# Nonface_net (classification after relu)
def test(dataloader, classifier, network_truncate):     
    ACC = np.zeros((1,205))
    time0 = time.time()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        network_truncate.layeract(inputs)
        truncate = network_truncate.feature_map.view(-1,256*13*13)
        out = classifier.forward(truncate) 
        _, outputs = torch.max(out, 1)
        for i in range(205):
            location = torch.where(targets==i)[0]
            acc = torch.sum(outputs[location]==targets[location])
            acc = acc.item()
            ACC[0, i] = ACC[0, i] + acc
    time1 = time.time()
    print(time1-time0, 's')
    return ACC

classifier = torch.nn.Sequential(torch.nn.Linear(256*13*13, 4096, bias=True),
                                    torch.nn.Sigmoid(),
                                    torch.nn.Linear(4096, 205, bias=True)).to(device) 
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)   
loss_function = torch.nn.CrossEntropyLoss()  
num_epoches = 100
metric_path = '/home/dell/Desktop/205cat/classification/net/Metric.csv'
f = open(metric_path, 'a+')
csv_write = csv.writer(f)
csv_write.writerow(['EPOCH', 'ACC_train_test', 'ACC_val_test'])
f.close()
for epoch in range(num_epoches):  
    print('Epoch step {}/{}'.format(epoch+1, num_epoches))
    for inputs, targets in dataloader_train:
        inputs = inputs.to(device)
        targets = targets.to(device)
        net_truncate.layeract(inputs)
        truncate = net_truncate.feature_map.view(-1,256*13*13)
        out = classifier.forward(truncate)  
        loss = loss_function(out, targets) 
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()
    ACC_train_test = test(dataloader_train_test, classifier, net_truncate)
    ACC_train_test = np.mean(ACC_train_test/70)
    ACC_val_test = test(dataloader_val_test, classifier, net_truncate)
    ACC_val_test = np.mean(ACC_val_test/10)
    with open(metric_path,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([epoch, ACC_train_test, ACC_val_test])
    modelname = '/home/dell/Desktop/205cat/classification/net/' + 'net_epoch' + str(epoch)
    torch.save(classifier, modelname)
    
    
# plot test acc
f_alexnet = pd.read_csv('/home/dell/Desktop/205cat/classification/Alexnet/Metric.csv')    
f_net = pd.read_csv('/home/dell/Desktop/205cat/classification/net/Metric.csv')    

acc_alexnet = []
for i in range(len(f_alexnet)):
    acc_alexnet.append(f_alexnet.loc[i][2])
acc_net = []
for i in range(len(f_net)):
    acc_net.append(f_net.loc[i][2])
    
plt.figure()
plt.plot(acc_alexnet, label='Alexnet')
plt.plot(acc_net, label='Nonface_net')
plt.plot(np.array(acc_alexnet)-np.array(acc_net), label='difference')
plt.legend()


# the last step error bar
f_bag = os.listdir('/home/dell/Desktop/Imgnet/traindata/n02769748/')
f_bag = random.sample(f_bag, 930)
with open('/home/dell/Desktop/205cat/bag_930.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    csv_write.writerow(['/home/dell/Desktop/Imgnet/traindata/n02769748/'])
    csv_write.writerow(['stimID', 'condition'])
    for pic in f_bag:
        csv_write.writerow([pic, 'n02769748'])
    
    
picdataset_face_val = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_face_930_FITW.csv', transform=data_transforms['val'])
dataloader_face_val = DataLoader(picdataset_face_val, batch_size=128, shuffle=False, num_workers=20)
picdataset_bag_val = dnn_io.PicDataset('/home/dell/Desktop/205cat/bag_930.csv', transform=data_transforms['val'])
dataloader_bag_val = DataLoader(picdataset_bag_val, batch_size=128, shuffle=False, num_workers=20)

def oneclass_test(dataloader, classifier, network_truncate, classnumber):
    ACC = 0
    time0 = time.time()
    network_truncate.eval()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        network_truncate.layeract(inputs)
        truncate = network_truncate.feature_map.view(-1,256*13*13)
        out = classifier.forward(truncate) 
        _, outputs = torch.max(out, 1)
        ACC += torch.where(outputs==classnumber)[0].shape[0]
    time1 = time.time()
    print(time1-time0, 's')
    return ACC

classifier_alexnet = torch.load('/home/dell/Desktop/205cat/classification/Alexnet/Alexnet_epoch89')
classifier_net = torch.load('/home/dell/Desktop/205cat/classification/net/net_epoch89')

acc_alexnet = test(dataloader_val_test, classifier_alexnet, alexnet_truncate)/10
acc_net = test(dataloader_val_test, classifier_net, net_truncate)/10
face_acc_alexnet = oneclass_test(dataloader_face_val, classifier_alexnet, alexnet_truncate, 204)/930
face_acc_net = oneclass_test(dataloader_face_val, classifier_net, net_truncate, 204)/930
bag_acc_alexnet = oneclass_test(dataloader_bag_val, classifier_alexnet, alexnet_truncate, 2)/930
bag_acc_net = oneclass_test(dataloader_bag_val, classifier_net, net_truncate, 2)/930
plt.figure()
plt.title('Compare classification')
plt.bar(['Alexnet','Nonface_net'], 
        [np.mean(acc_alexnet),np.mean(acc_net)], 
        yerr=[np.std(acc_alexnet)/np.sqrt(acc_alexnet.shape[1]),np.std(acc_net)/np.sqrt(acc_net.shape[1])])
plt.bar(['Face_Alexnet','Face_Nonface_net'], 
        [face_acc_alexnet,face_acc_net])
plt.bar(['Bag_Alexnet','Bag_Nonface_net'], 
        [bag_acc_alexnet,bag_acc_net])


# test
from scipy.stats import ttest_ind, ttest_rel

classifier_alexnet = torch.load('/home/dell/Desktop/205cat/classification/Alexnet/Alexnet_epoch89')
a = test(dataloader_val_test, classifier_alexnet, alexnet_truncate)/10
classifier_net = torch.load('/home/dell/Desktop/205cat/classification/net/net_epoch89')
b = test(dataloader_val_test, classifier_net, net_truncate)/10
print(ttest_ind(a.reshape(205), b.reshape(205)))
print(ttest_rel(a.reshape(205), b.reshape(205)))




'11. Transfer learning to face'
###############################################################################
###############################################################################
import torch.nn as nn
import copy
import os
import csv
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt  
from torch.utils.data import DataLoader     
import torchvision.transforms as transforms
import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import io as dnn_io
from dnnbrain.dnn import models


### 70+10 205class pictures
###########################
f = os.listdir('/home/dell/Desktop/205cat/nonpeople_204x80_256Obj')
f.sort()
with open('/home/dell/Desktop/205cat/allstim_train.csv', 'a+') as train:
    with open('/home/dell/Desktop/205cat/allstim_val.csv', 'a+') as val:
        train_write = csv.writer(train)
        val_write = csv.writer(val)
        count = 0
        for i in f:
            count += 1
            row = [i, i[:3]]
            if count<=70:
                train_write.writerow(row)
            elif 70<count<=80:
                val_write.writerow(row)
            else:
                count = 1
                train_write.writerow(row)


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
picdataset_train = dnn_io.PicDataset('/home/dell/Desktop/205cat/allstim_train.csv', transform=data_transforms['train'])
dataloader_train = DataLoader(picdataset_train, batch_size=128, shuffle=True, num_workers=20)

picdataset_train_test = dnn_io.PicDataset('/home/dell/Desktop/205cat/allstim_train.csv', transform=data_transforms['val'])
dataloader_train_test = DataLoader(picdataset_train_test, batch_size=128, shuffle=False, num_workers=20)

picdataset_val_test = dnn_io.PicDataset('/home/dell/Desktop/205cat/allstim_val.csv', transform=data_transforms['val'])
dataloader_val_test = DataLoader(picdataset_val_test, batch_size=128, shuffle=False, num_workers=20)


# load non-face net and alexnet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
non_face_net = torch.load('/home/dell/Desktop/data_sony\\win7\\win10\\dell/model_epoch89')
net = copy.deepcopy(non_face_net)
origin_alexnet = torchvision.models.alexnet(pretrained=True).to(device)
alexnet = copy.deepcopy(origin_alexnet)


# Transfer Learning face (all params)
alexnet.classifier[6] = torch.nn.Linear(4096, 205, bias=True)
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
save_model_path = '/home/dell/Desktop/205cat/Transfer_Face/Alexnet_allparams/'
path_loss_acc = '/home/dell/Desktop/205cat/Transfer_Face/Alexnet_allparams/Metric.csv'
model, metric_dict = models.dnn_train_model(dataloaders_train=dataloader_train, 
                               dataloaders_train_test=dataloader_train_test, 
                               dataloaders_val_test=dataloader_val_test,
                               model=alexnet, 
                               criterion=criterion, 
                               optimizer=optimizer, 
                               num_epoches=20, 
                               train_method='tradition',
                               save_model_path=save_model_path,
                               metric_path=path_loss_acc)


net.classifier[6] = torch.nn.Linear(4096, 205, bias=True)
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
save_model_path = '/home/dell/Desktop/205cat/Transfer_Face/Nonface-alexnet_allparams/'
path_loss_acc = '/home/dell/Desktop/205cat/Transfer_Face/Nonface-alexnet_allparams/Metric.csv'
model, metric_dict = models.dnn_train_model(dataloaders_train=dataloader_train, 
                               dataloaders_train_test=dataloader_train_test, 
                               dataloaders_val_test=dataloader_val_test,
                               model=net, 
                               criterion=criterion, 
                               optimizer=optimizer, 
                               num_epoches=20, 
                               train_method='tradition',
                               save_model_path=save_model_path,
                               metric_path=path_loss_acc)




### Imgnet based 1001class pictures
###################################
import shutil
import os
import random
import csv

f = os.listdir('/home/dell/Desktop/205cat/FITW_1000')
l1 = random.sample(f, 850)
l2 = list(set(f)-set(l1))   
for pic in l1:
    picdir = '/home/dell/Desktop/205cat/FITW_1000/'+pic
    target = '/home/dell/Desktop/Imgnet/traindata/n99999999/'+pic
    shutil.copy(picdir, target)
for pic in l2:
    picdir = '/home/dell/Desktop/205cat/FITW_1000/'+pic
    target = '/home/dell/Desktop/Imgnet/valdata/'+pic
    shutil.copy(picdir, target)


f1 = os.listdir('/home/dell/Desktop/Imgnet/traindata/n99999999')
with open('/home/dell/Desktop/data_sony\win7\win10\dell/Fine-tuning_1001_train.csv','a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for pic in f1:
        picdir = 'n99999999/'+pic
        condition = '1001'
        csv_write.writerow([picdir, condition])
with open('/home/dell/Desktop/data_sony\win7\win10\dell/Fine-tuning_1001_val.csv','a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for pic in l2:
        picdir = pic
        condition = '1001'
        csv_write.writerow([picdir, condition])


# Transfer Learning Non-face alexnet (all params)
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
picdataset_train = dnn_io.PicDataset('/home/dell/Desktop/data_sony\win7\win10\dell/Fine-tuning_1001_train.csv',transform=data_transforms['train'])
dataloader_train = DataLoader(picdataset_train, batch_size=128, shuffle=True, num_workers=20)

picdataset_train_test = dnn_io.PicDataset('/home/dell/Desktop/data_sony\win7\win10\dell/Fine-tuning_1001_train.csv',transform=data_transforms['val'])
dataloader_train_test = DataLoader(picdataset_train_test, batch_size=128, shuffle=False, num_workers=20)

picdataset_val_test = dnn_io.PicDataset('/home/dell/Desktop/data_sony\win7\win10\dell/Fine-tuning_1001_val.csv',transform=data_transforms['val'])
dataloader_val_test = DataLoader(picdataset_val_test, batch_size=128, shuffle=False, num_workers=20)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
non_face_net = torch.load('/home/dell/Desktop/data_sony\\win7\\win10\\dell/model_epoch89')
net = copy.deepcopy(non_face_net)


net.classifier[6] = torch.nn.Linear(4096, 737, bias=True)
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
save_model_path = '/home/dell/Desktop/data_sony\win7\win10\dell/addface_fune-tuning/net/'
path_loss_acc = '/home/dell/Desktop/data_sony\win7\win10\dell/addface_fune-tuning/net/Metric.csv'
model = models.dnn_train_model(dataloaders_train=dataloader_train, 
                               dataloaders_train_test=dataloader_train_test, 
                               dataloaders_val_test=dataloader_val_test,
                               model=net, 
                               criterion=criterion, 
                               optimizer=optimizer, 
                               num_epoches=100, 
                               train_method='tradition',
                               save_model_path=save_model_path,
                               metric_path=path_loss_acc)



### Analysis
model = torch.load('/home/dell/Desktop/205cat/Transfer_Face/Nonface-alexnet_allparams/model_epoch10')

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
sys.path.append('/home/dell/Desktop/My_module')
import reconstruction_layer
import Guided_CAM

# load picture [107, 99, 778, 887, 556, 138, 945]
number_of_picture = 945
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ])])                            
picdataset = dnn_io.PicDataset('/home/dell/Desktop/205cat/stimuli_table_face_1000_FITW.csv', transform=transform)
picimg = picdataset[number_of_picture][0].unsqueeze(0)
picimg.requires_grad=True
plt.figure()
plt.imshow(picimg[0].permute(1,2,0).data.numpy())
plt.axis('off')

# hypoparameters
target_layer = 10
target_class = 0

## Non-face alexnet
# model out and class
for channel in face_channel:
    # Guided backprop
    out_image = reconstruction_layer.layer_channel_reconstruction(net_alexnet, picimg, target_layer, channel)
    #plt.imshow(out_image)
    
    # Grad cam
    gcv2 = Guided_CAM.GradCam(net, target_layer=target_layer)
    cam, weights = gcv2.generate_cam(picimg, target_class=target_class, channel=channel)
    #plt.imshow(gcv2.cam_channel)
    
    # Guided Grad cam
    cam_gb = Guided_CAM.guided_grad_cam(gcv2.cam_channel/255, out_image.permute(2,0,1).data.numpy())
    plt.figure()
    #plt.title('Non-face alexnet Guided-CAM (Conv5 channel%d)' %channel)
    plt.imshow(Guided_CAM.convert_to_grayscale(cam_gb, threshold=99)[0,:])
    plt.axis('off')
    plt.savefig('/home/dell/Desktop/channel%d' %channel)
    


## weights of representation for face channel
# Non-face alexnet
target_layer = 10
target_class = 0
Location_nonface = np.array([])
for n in range(len(picdataset)):
    picimg = picdataset[n][0].unsqueeze(0)
    gcv2 = Guided_CAM.GradCam(net, target_layer=target_layer)
    cam, weights = gcv2.generate_cam(picimg, target_class=target_class)
    for channel in face_channel:
        location = np.max(np.where(np.sort(weights)==weights[channel])[0]).item()
        Location_nonface = np.append(Location_nonface, location)
Location_nonface = Location_nonface.reshape(-1, len(face_channel))


plt.figure()
Location_alexnet_avg = np.mean(Location_alexnet, axis=0)
std_err = np.std(Location_alexnet, axis=0)
plt.bar(['59','66','93','124','184','186'], Location_alexnet_avg, yerr=std_err, label='Alexnet')
Location_nonface_avg = np.mean(Location_nonface, axis=0)
std_err = np.std(Location_nonface, axis=0)
plt.bar(['46','48','139','157'], Location_nonface_avg, yerr=std_err, label='Non-face alexnet')
plt.legend()
plt.xlabel('channel number')
plt.ylabel('The importance of weight in this channel')
plt.title('Compare face channels in Alexnet and Non-face alexnet')




'12. Compare face unit to brain'
###############################################################################
###############################################################################
'12.1 Extract activation of each frame in a video'
import copy
import cv2
from PIL import Image
import nibabel as nib
import SimpleITK as sitk
from nibabel import freesurfer
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from os.path import join as pjoin
import pickle
import scipy.stats as stats
import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import core
from dnnbrain.brain import io as brain_io


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alexnet = torchvision.models.alexnet(pretrained=True).to(device)
classification_alexnet = torch.load('/home/dell/Desktop/205cat/classification/Alexnet/model_epoch99')
nonface_net = torch.load('/home/dell/Desktop/data_sony\win7\win10\dell/model_epoch89')
classification_nonface = torch.load('/home/dell/Desktop/205cat/classification/Nonface_net/model_epoch99')


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
            self.conv_output = torch.mean(layer_out, (2,3))
        self.model.features[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if index == self.selected_layer:
                break
selected_layer = 11
net_truncate = NET(nonface_net, selected_layer=selected_layer)
alexnet_truncate = NET(alexnet, selected_layer=selected_layer)


parpath = '/home/dell/Desktop/DNN2Brain/movie'
video_name = ['seg'+str(i)+'.mp4' for i in np.arange(2,19,1)]
for vn in video_name:
    print('Now execute video {}'.format(vn))
    vidcap = cv2.VideoCapture(pjoin(parpath, vn))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    framecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # framecount = 1

    transform = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    output_act = np.array([])
    for i in range(framecount):
        if (i+1)%1000 == 0:
            print('    Frame count {}'.format(i+1))
        ifpic, picimg = vidcap.read()
        if ifpic:
            picimg = Image.fromarray(cv2.cvtColor(picimg, cv2.COLOR_BGR2RGB))
            picimg = transform(picimg).to(device)
            alexnet_truncate.layeract(picimg[None,...])
            #net_truncate.layeract(picimg[None,...])
            truncate = alexnet_truncate.feature_map.view(-1,256*13*13)
            #truncate = net_truncate.feature_map.view(-1,256*13*13)
            out = classification_alexnet(truncate) 
            #out = classification_nonface(truncate) 
            output_tmp = out[0][204].item()                          # 204 is the face unit
            output_act = np.append(output_act, output_tmp)
    vidcap.release()

    # Save array as .pkl.
    with open('/home/dell/Desktop/DNN2Brain/faceunit_data/Alexnet_'+vn.split('.')[0]+'.pkl', 'wb') as f:
        pickle.dump(output_act, f)
#    with open('/home/dell/Desktop/DNN2Brain/faceunit_data/Nonface_net'+vn.split('.')[0]+'.pkl', 'wb') as f:
#        pickle.dump(output_act, f)




'12.2 Caculate correlation by each voxel and block permutation test'
def get_cnn_hrf_signal(actdata, tr=2, fps=1.0/30):
    timept = actdata.shape[0]
    actlength = (timept)*(fps)
    onset = np.linspace(0, (timept)*fps, timept)
    duration = np.array([1.0/30]*timept)    
    cnn_hrf_signal = core.convolve_hrf(actdata, onset, duration, int(actlength/tr), tr)[1:]
    return cnn_hrf_signal


def prepare_fMRI_image(video_type, subject_name='all'):
    parpath = '/home/dell/Desktop/DNN2Brain/video_fmri_dataset'
    if subject_name == 'all':
        subjects = ['subject1', 'subject2', 'subject3']
    else: 
        subjects = [subject_name]
    run_num = np.array([1, 2])    
    img_allsubj = []
    for subj in subjects:
        img = np.zeros((239, 91282))
        for rn in run_num:
            img_tmp, header = brain_io.load_brainimg(os.path.join(parpath, subj, 'fmri', video_type, 'cifti', video_type+'_'+str(rn)+'_Atlas.dtseries.nii'))
            img += img_tmp[:239,:,0,0]
        img_avg = img/len(run_num)
        img_allsubj.append(img_avg)
    img_allsubj = np.array(img_allsubj)
    img_allsubj = np.mean(img_allsubj,axis=0)
    return img_allsubj, header


## generate R map
seg_type = ['seg'+str(i) for i in np.arange(1,19,1)]
R_avgseg_alexnet = []
R_avgseg_net = []

for i, st in enumerate(seg_type):
    print('Segment {}'.format(st))
    
    actdata_alexnet = np.load('/home/dell/Desktop/DNN2Brain/faceunit_data/Alexnet_'+st+'.pkl', allow_pickle=True)
    actdata_alexnet[actdata_alexnet<0] = 0
    actdata_alexnet = np.log(actdata_alexnet+0.01)    
    actdata_net = np.load('/home/dell/Desktop/DNN2Brain/faceunit_data/Nonface_net'+st+'.pkl', allow_pickle=True)
    actdata_net[actdata_net<0] = 0
    actdata_net = np.log(actdata_net+0.01)

    hrfsignal_alexnet = get_cnn_hrf_signal(actdata_alexnet.reshape(-1,1))
    hrfsignal_alexnet = stats.zscore(hrfsignal_alexnet)
    hrfsignal_net = get_cnn_hrf_signal(actdata_net.reshape(-1,1))
    hrfsignal_net = stats.zscore(hrfsignal_net)

    outimg, header = prepare_fMRI_image(st) 
    outimg = stats.zscore(outimg, axis=0)
    
    R_alexnet = []
    for voxel_timeseries in outimg.T:
        r = stats.pearsonr(voxel_timeseries, hrfsignal_alexnet.reshape(239))[0]
        R_alexnet.append(r)        
    R_avgseg_alexnet.append(R_alexnet)
    
    
    R_net = []
    for voxel_timeseries in outimg.T:
        r = stats.pearsonr(voxel_timeseries, hrfsignal_net.reshape(239))[0]
        R_net.append(r)
    R_avgseg_net.append(R_net)

R_avgseg_alexnet = np.array(R_avgseg_alexnet)
R_avgseg_net = np.array(R_avgseg_net)

Rmap_alexnet = np.mean(R_avgseg_alexnet, axis=0)
Rmap_net = np.mean(R_avgseg_net, axis=0)

plt.figure()
plt.plot(Rmap_alexnet, alpha=0.7)
plt.plot(Rmap_net, alpha=0.7)


data_nii = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')
# Rmap_alexnet
img = np.array([])
for i in range(245):
    img = np.append(img, Rmap_alexnet)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Yeo_templates/surface/Rmap_alexnet.dtseries.nii')
# Rmap_net
img = np.array([])
for i in range(245):
    img = np.append(img, Rmap_net)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Yeo_templates/surface/Rmap_net.dtseries.nii')



## block permutation test
seg_type = ['seg'+str(i) for i in np.arange(1,19,1)]
R_avgseg_alexnet = []
R_avgseg_net = []

for i, st in enumerate(seg_type):
    print('Segment {}'.format(st))   
    actdata_alexnet = np.load('/home/dell/Desktop/DNN2Brain/faceunit_data/Alexnet_'+st+'.pkl', allow_pickle=True)
    actdata_alexnet[actdata_alexnet<0] = 0
    actdata_alexnet = np.log(actdata_alexnet+0.01)    
    actdata_net = np.load('/home/dell/Desktop/DNN2Brain/faceunit_data/Nonface_net'+st+'.pkl', allow_pickle=True)
    actdata_net[actdata_net<0] = 0
    actdata_net = np.log(actdata_net+0.01)

    hrfsignal_alexnet = get_cnn_hrf_signal(actdata_alexnet.reshape(-1,1))
    hrfsignal_alexnet = stats.zscore(hrfsignal_alexnet)
    hrfsignal_net = get_cnn_hrf_signal(actdata_net.reshape(-1,1))
    hrfsignal_net = stats.zscore(hrfsignal_net)

    outimg, header = prepare_fMRI_image(st) 
    outimg = stats.zscore(outimg, axis=0)
    
    
    R_alexnet = []
    for n,voxel_timeseries in enumerate(outimg.T):
        print('voxel:', n)
        # calculate r of a voxel and face unit
        r = stats.pearsonr(voxel_timeseries, hrfsignal_alexnet.reshape(239))[0]
        
        # make this voxel's corresponding null-distribution
        faceunit_timeseries = hrfsignal_alexnet.reshape(239)
        R_null = np.array([])
        for t in range(100000):
            shift_number = int(np.random.uniform(0,faceunit_timeseries.shape[0],1))
            faceunit_timeseries_timeshift = np.append(faceunit_timeseries[shift_number:], faceunit_timeseries[:shift_number])
            blocks = []
            for time_point in np.arange(0,239,25):
                blocks.append(faceunit_timeseries_timeshift[time_point:time_point+25])
            blocks = np.array(blocks)
            np.random.shuffle(blocks)
            # a new timeseries of face unit
            new_timeseries = np.array([])
            for t in blocks:
                new_timeseries = np.append(new_timeseries, t)
            # calculate r of the voxel and new face unit timeseries
            r_permutation = stats.pearsonr(voxel_timeseries, new_timeseries)[0]
            R_null = np.append(R_null, r_permutation)
        alpha_top = np.percentile(R_null, 100-2.5/91282)
        alpha_down = np.percentile(R_null, 2.5/91282)
        if r>alpha_top or r<alpha_down:
            R_alexnet.append(r)
        else:
            R_alexnet.append(0)       
    R_avgseg_alexnet.append(R_alexnet)
    
    
    R_net = []
    for n,voxel_timeseries in enumerate(outimg.T):
        print('voxel:', n)
        # calculate r of a voxel and face unit
        r = stats.pearsonr(voxel_timeseries, hrfsignal_net.reshape(239))[0]
        
        # make this voxel's corresponding null-distribution
        faceunit_timeseries = hrfsignal_net.reshape(239)
        R_null = np.array([])
        for t in range(1000):
            shift_number = int(np.random.uniform(0,faceunit_timeseries.shape[0],1))
            faceunit_timeseries_timeshift = np.append(faceunit_timeseries[shift_number:], faceunit_timeseries[:shift_number])
            blocks = []
            for time_point in np.arange(0,239,25):
                blocks.append(faceunit_timeseries_timeshift[time_point:time_point+25])
            blocks = np.array(blocks)
            np.random.shuffle(blocks)
            # a new timeseries of face unit
            new_timeseries = np.array([])
            for t in blocks:
                new_timeseries = np.append(new_timeseries, t)
            # calculate r of the voxel and new face unit timeseries
            r_permutation = stats.pearsonr(voxel_timeseries, new_timeseries)[0]
            R_null = np.append(R_null, r_permutation)
        alpha_top = np.percentile(R_null, 100-2.5/91282)
        alpha_down = np.percentile(R_null, 2.5/91282)
        if r>alpha_top or r<alpha_down:
            R_net.append(r)
        else:
            R_net.append(0)       
    R_avgseg_net.append(R_net)


R_avgseg_alexnet = np.array(R_avgseg_alexnet)
R_avgseg_net = np.array(R_avgseg_net)

Rmap_alexnet = np.mean(R_avgseg_alexnet, axis=0)
Rmap_net = np.mean(R_avgseg_net, axis=0)

plt.figure()
plt.plot(Rmap_alexnet, alpha=0.7)
plt.plot(Rmap_net, alpha=0.7)    

            

'12.3 ROI analysis by mask (In MMP, FFA is 18/198, OFA is 22/202, STS is 128129130/308309310)'
mask = nib.load('/home/dell/Desktop/DNN2Brain/MMP_mask/surface/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
mask_ffa = np.where((mask==18)|(mask==198))[0]
mask_ofa = np.where((mask==22)|(mask==202))[0]
mask_sts = np.where((mask==128)|(mask==129)|(mask==130)|(mask==308)|(mask==309)|(mask==310))[0]
ffa_ofa = np.append(mask_ffa, mask_ofa)
ffa_ofa_sts = np.append(ffa_ofa, mask_sts)


seg_type = ['seg'+str(i) for i in np.arange(1,19,1)]
R_avgseg_alexnet = []
for i, st in enumerate(seg_type):
    print('Segment {}'.format(st))   
    actdata_alexnet = np.load('/home/dell/Desktop/DNN2Brain/faceunit_data/Alexnet_'+st+'.pkl', allow_pickle=True)
    actdata_alexnet[actdata_alexnet<0] = 0
    actdata_alexnet = np.log(actdata_alexnet+0.01)    

    hrfsignal_alexnet = get_cnn_hrf_signal(actdata_alexnet.reshape(-1,1))
    hrfsignal_alexnet = stats.zscore(hrfsignal_alexnet)

    outimg, header = prepare_fMRI_image(st) 
    outimg = stats.zscore(outimg, axis=0)
    ffa_voxel = outimg.T[mask_ffa]
    ofa_voxel = outimg.T[mask_ofa]
    sts_voxel = outimg.T[mask_sts]
    selected_voxel = np.vstack((ffa_voxel, ofa_voxel))
    
    R_alexnet = []
    for n,voxel_timeseries in enumerate(selected_voxel):
        print('seg/voxel:', st, n)
        # calculate r of a voxel and face unit
        r = stats.pearsonr(voxel_timeseries, hrfsignal_alexnet.reshape(239))[0]
        
        # make this voxel's corresponding null-distribution
        faceunit_timeseries = hrfsignal_alexnet.reshape(239)
        R_null = np.array([])
        for t in range(1000):
            shift_number = int(np.random.uniform(0,faceunit_timeseries.shape[0],1))
            faceunit_timeseries_timeshift = np.append(faceunit_timeseries[shift_number:], faceunit_timeseries[:shift_number])
            blocks = []
            for time_point in np.arange(0,239,25):
                blocks.append(faceunit_timeseries_timeshift[time_point:time_point+25])
            blocks = np.array(blocks)
            np.random.shuffle(blocks)
            # a new timeseries of face unit
            new_timeseries = np.array([])
            for t in blocks:
                new_timeseries = np.append(new_timeseries, t)
            # calculate r of the voxel and new face unit timeseries
            r_permutation = stats.pearsonr(voxel_timeseries, new_timeseries)[0]
            R_null = np.append(R_null, r_permutation)
        alpha_top = np.percentile(R_null, 100-2.5/selected_voxel.shape[0])
        alpha_down = np.percentile(R_null, 2.5/selected_voxel.shape[0])
        if r>alpha_top or r<alpha_down:
            R_alexnet.append(r)
        else:
            R_alexnet.append(0)       
    R_avgseg_alexnet.append(R_alexnet)
    

seg_type = ['seg'+str(i) for i in np.arange(1,19,1)]
R_avgseg_net = []
for i, st in enumerate(seg_type):
    print('Segment {}'.format(st))   
    actdata_net = np.load('/home/dell/Desktop/DNN2Brain/faceunit_data/Nonface_net'+st+'.pkl', allow_pickle=True)
    actdata_net[actdata_net<0] = 0
    actdata_net = np.log(actdata_net+0.01)

    hrfsignal_net = get_cnn_hrf_signal(actdata_net.reshape(-1,1))
    hrfsignal_net = stats.zscore(hrfsignal_net)

    outimg, header = prepare_fMRI_image(st) 
    outimg = stats.zscore(outimg, axis=0)
    ffa_voxel = outimg.T[mask_ffa]
    ofa_voxel = outimg.T[mask_ofa]
    sts_voxel = outimg.T[mask_sts]
    selected_voxel = np.vstack((ffa_voxel, ofa_voxel))
    
    R_net = []
    for n,voxel_timeseries in enumerate(selected_voxel):
        print('seg/voxel:', st, n)
        # calculate r of a voxel and face unit
        r = stats.pearsonr(voxel_timeseries, hrfsignal_net.reshape(239))[0]
        
        # make this voxel's corresponding null-distribution
        faceunit_timeseries = hrfsignal_net.reshape(239)
        R_null = np.array([])
        for t in range(1000):
            shift_number = int(np.random.uniform(0,faceunit_timeseries.shape[0],1))
            faceunit_timeseries_timeshift = np.append(faceunit_timeseries[shift_number:], faceunit_timeseries[:shift_number])
            blocks = []
            for time_point in np.arange(0,239,25):
                blocks.append(faceunit_timeseries_timeshift[time_point:time_point+25])
            blocks = np.array(blocks)
            np.random.shuffle(blocks)
            # a new timeseries of face unit
            new_timeseries = np.array([])
            for t in blocks:
                new_timeseries = np.append(new_timeseries, t)
            # calculate r of the voxel and new face unit timeseries
            r_permutation = stats.pearsonr(voxel_timeseries, new_timeseries)[0]
            R_null = np.append(R_null, r_permutation)
        alpha_top = np.percentile(R_null, 100-2.5/selected_voxel.shape[0])
        alpha_down = np.percentile(R_null, 2.5/selected_voxel.shape[0])
        if r>alpha_top or r<alpha_down:
            R_net.append(r)
        else:
            R_net.append(0)       
    R_avgseg_net.append(R_net)


R_avgseg_alexnet = np.array(R_avgseg_alexnet)
R_avgseg_net = np.array(R_avgseg_net)

Rmap_alexnet = np.mean(R_avgseg_alexnet, axis=0)
Zero = np.zeros(outimg.shape[1])
Zero[ffa_ofa] = Rmap_alexnet
Rmap_alexnet = Zero
Rmap_net = np.mean(R_avgseg_net, axis=0)
Zero = np.zeros(outimg.shape[1])
Zero[ffa_ofa] = Rmap_net
Rmap_net = Zero


data_nii = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')
# Rmap_alexnet
img = np.array([])
for i in range(245):
    img = np.append(img, Rmap_alexnet)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Yeo_templates/surface/Rmap_alexnet_ffa_ofa_sts.dtseries.nii')

# Rmap_net
img = np.array([])
for i in range(245):
    img = np.append(img, Rmap_net)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Yeo_templates/surface/Rmap_net_ffa_ofa_sts.dtseries.nii')





'12.4 project face unit timeseries to Liuzm Brain'
data_nii = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')

# Rmap_alexnet
img = np.array([])
for i in range(245):
    img = np.append(img, Rmap_alexnet)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Yeo_templates/surface/Rmap_alexnet.dtseries.nii')

# Rmap_net
img = np.array([])
for i in range(245):
    img = np.append(img, Rmap_net)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Yeo_templates/surface/Rmap_net.dtseries.nii')

# Rmap_diff
Rmap_diff = Rmap_alexnet-Rmap_net
img = np.array([])
for i in range(245):
    img = np.append(img, Rmap_diff)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Yeo_templates/surface/Rmap_diff.dtseries.nii')


# Rmap_diff_test based on roi(ffa,ofa,sts)
from scipy.stats import norm
from scipy.stats import ttest_ind
#n1 = 245
#n2 = 245
#F_r1 = 0.5*np.log(np.abs((1+Rmap_alexnet)/(1-Rmap_alexnet)))
#F_r2 = 0.5*np.log(np.abs((1+Rmap_net)/(1-Rmap_net)))
#Z = np.abs((F_r1 - F_r2)/np.sqrt((1/(n1-3))+(1/(n2-3))))
#p_value = 1-norm.cdf(Z)
def Fisher_z(r):
    return 0.5*np.log(np.abs((1+r)/(1-r)))

def R_ttest(A,B):  
    return ttest_ind(Fisher_z(A), Fisher_z(B), equal_var = True)

mask = nib.load('/home/dell/Desktop/DNN2Brain/MMP_mask/surface/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
mask_ffa = np.where((mask==18)|(mask==198))[0]
mask_ofa = np.where((mask==22)|(mask==202))[0]
mask_sts = np.where((mask==128)|(mask==129)|(mask==130)|(mask==308)|(mask==309)|(mask==310))[0]

R_ffa_alexnet = Rmap_alexnet[mask_ffa]
R_ofa_alexnet = Rmap_alexnet[mask_ofa]
R_sts_alexnet = Rmap_alexnet[mask_sts]
R_ffa_net = Rmap_net[mask_ffa]
R_ofa_net = Rmap_net[mask_ofa]
R_sts_net = Rmap_net[mask_sts]

R_ttest(R_ffa_alexnet, R_ffa_net)
R_ttest(R_ofa_alexnet, R_ofa_net)
R_ttest(R_sts_alexnet, R_sts_net)

Rmap_diff = Rmap_alexnet-Rmap_net
Zero = np.zeros(Rmap_diff.shape[0])
Zero[mask_ffa] = Rmap_diff[mask_ffa]
Zero[mask_ofa] = Rmap_diff[mask_ofa]
Zero[mask_sts] = Rmap_diff[mask_sts]
Rmap_diff = Zero

np.where(Rmap_diff )
img = np.array([])
for i in range(245):
    img = np.append(img, Rmap_diff)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Yeo_templates/surface/Rmap_diff.dtseries.nii')






