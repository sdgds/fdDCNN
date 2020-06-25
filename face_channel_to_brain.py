"13. Compare Conv5's face channel to brain"
###############################################################################
###############################################################################
import os
import copy
import cv2
from PIL import Image
import nibabel as nib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from os.path import join as pjoin
import pickle
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import core
from dnnbrain.brain import io as brain_io


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alexnet = torchvision.models.alexnet(pretrained=True).to(device)
nonface_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')


class NET(torch.nn.Module):
    def __init__(self, model, selected_layer):
        super(NET, self).__init__()
        self.model = model
        self.selected_layer = selected_layer
        self.conv_output = 0
    def hook_layer(self):
        def hook_function(module, layer_in, layer_out):
            self.conv_output = torch.mean(layer_out, (2,3))
        self.model.features[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if index == self.selected_layer:
                break
selected_layer = 11
alexnet_truncate = NET(alexnet, selected_layer=selected_layer)
net_truncate = NET(nonface_net, selected_layer=selected_layer)


parpath = '/home/dell/Desktop/DNN2Brain/movie'
video_name = ['seg'+str(i)+'.mp4' for i in np.arange(1,19,1)]
for vn in video_name:
    print('Now execute video {}'.format(vn))
    vidcap = cv2.VideoCapture(pjoin(parpath, vn))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    framecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
    output_act = []
    for i in range(framecount):
        if (i+1)%1000 == 0:
            print('    Frame count {}'.format(i+1))
        ifpic, picimg = vidcap.read()
        if ifpic:
            picimg = Image.fromarray(cv2.cvtColor(picimg, cv2.COLOR_BGR2RGB))
            picimg = transform(picimg).to(device)
            #alexnet_truncate.layeract(picimg[None,...])
            net_truncate.layeract(picimg[None,...])
            #truncate = alexnet_truncate.conv_output.reshape(256)
            truncate = net_truncate.conv_output.reshape(256)
            output_act.append(truncate.cpu().data.tolist())
    output_act = np.array(output_act).T
    vidcap.release()

    # Save array as .pkl.
#    with open('/home/dell/Desktop/DNN2Brain/Conv5_data/Alexnet_'+vn.split('.')[0]+'.pkl', 'wb') as f:
#        pickle.dump(output_act, f)
    with open('/home/dell/Desktop/DNN2Brain/Conv5_data/Nonface_lrdecay_'+vn.split('.')[0]+'.pkl', 'wb') as f:
        pickle.dump(output_act, f)
        



import os
import copy
import cv2
from PIL import Image
import nibabel as nib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from os.path import join as pjoin
import pickle
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from scipy.stats import ttest_1samp
import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import core
from dnnbrain.brain import io as brain_io


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alexnet = torchvision.models.alexnet(pretrained=True).to(device)
nonface_net = torch.load('/home/dell/Desktop/Nonface_alexnet/model_epoch89')


class NET(torch.nn.Module):
    def __init__(self, model, selected_layer):
        super(NET, self).__init__()
        self.model = model
        self.selected_layer = selected_layer
        self.conv_output = 0
    def hook_layer(self):
        def hook_function(module, layer_in, layer_out):
            self.conv_output = torch.mean(layer_out, (2,3))
        self.model.features[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if index == self.selected_layer:
                break
selected_layer = 11
alexnet_truncate = NET(alexnet, selected_layer=selected_layer)
net_truncate = NET(nonface_net, selected_layer=selected_layer)


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



seg_type = ['seg'+str(i) for i in np.arange(1,19,1)]
Coef_avgseg_alexnet = []
R2_avgseg_alexnet = []
for i, st in enumerate(seg_type):
    print('Segment {}'.format(st))    
    actdata_alexnet = np.load('/home/dell/Desktop/DNN2Brain/Conv5_data/Alexnet_'+st+'.pkl', allow_pickle=True)
    actdata_alexnet = np.vstack((actdata_alexnet[184], actdata_alexnet[124]))
    actdata_alexnet[actdata_alexnet<0] = 0
    actdata_alexnet = np.log(actdata_alexnet+0.01)    

    hrfsignal_alexnet = get_cnn_hrf_signal(actdata_alexnet.T)
    hrfsignal_alexnet = stats.zscore(hrfsignal_alexnet)

    outimg, header = prepare_fMRI_image(st) 
    outimg = stats.zscore(outimg, axis=0)
    
    Coef_alexnet = []
    R2_alexnet = []
    for voxel_timeseries in outimg.T:
        # OLS method to estimate params for Linear regresion
        lineareg = sm.OLS(voxel_timeseries, hrfsignal_alexnet)
        result = lineareg.fit()
        # calulate R2
        R2 = metrics.r2_score(voxel_timeseries, result.predict(hrfsignal_alexnet))
        R2_alexnet.append(R2) 
        # p values for betas
#        for i in [0,1]:
#            if result.pvalues[i]>(0.025/91282):
#                result.params[i] = 0
#            if result.pvalues[i]<(0.025/91282):
#                result.params[i] = result.params[i]
        Coef_alexnet.append(result.params)
    Coef_avgseg_alexnet.append(Coef_alexnet)
    R2_avgseg_alexnet.append(R2_alexnet)

# VIF
seg_type = ['seg'+str(i) for i in np.arange(1,19,1)]
VIF_alexnet = []
for i, st in enumerate(seg_type):
    actdata_alexnet = np.load('/home/dell/Desktop/DNN2Brain/Conv5_data/Alexnet_'+st+'.pkl', allow_pickle=True)
    actdata_alexnet = np.vstack((actdata_alexnet[184], actdata_alexnet[124]))
    actdata_alexnet[actdata_alexnet<0] = 0
    actdata_alexnet = np.log(actdata_alexnet+0.01)   
    hrfsignal_alexnet = get_cnn_hrf_signal(actdata_alexnet.T)
    hrfsignal_alexnet = stats.zscore(hrfsignal_alexnet)
    # OLS method to estimate params for Linear regresion
    lineareg = sm.OLS(hrfsignal_alexnet[:,0], hrfsignal_alexnet[:,1])
    result = lineareg.fit()
    # calulate R2
    R2 = metrics.r2_score(hrfsignal_alexnet[:,0], result.predict(hrfsignal_alexnet[:,1]))
    VIF = 1/(1-R2)
    VIF_alexnet.append(VIF) 
plt.plot(VIF_alexnet)
    



seg_type = ['seg'+str(i) for i in np.arange(1,19,1)]
Coef_avgseg_net = []
R2_avgseg_net = []
for i, st in enumerate(seg_type):
    print('Segment {}'.format(st)) 
    actdata_net = np.load('/home/dell/Desktop/DNN2Brain/Conv5_data/Nonface_lrdecay_'+st+'.pkl', allow_pickle=True)
    actdata_net = np.vstack((actdata_net[28], actdata_net[49]))
    actdata_net[actdata_net<0] = 0
    actdata_net = np.log(actdata_net+0.01)

    hrfsignal_net = get_cnn_hrf_signal(actdata_net.T)
    hrfsignal_net = stats.zscore(hrfsignal_net)

    outimg, header = prepare_fMRI_image(st) 
    outimg = stats.zscore(outimg, axis=0)
    
    Coef_net = []
    R2_net = []
    for voxel_timeseries in outimg.T:
        # OLS method to estimate params for Linear regresion
        lineareg = sm.OLS(voxel_timeseries, hrfsignal_net)
        result = lineareg.fit()
        # calulate R2
        R2 = metrics.r2_score(voxel_timeseries, result.predict(hrfsignal_net))
        R2_net.append(R2) 
        # p values for betas
#        for i in [0,1]:
#            if result.pvalues[i]>(0.025/91282):
#                result.params[i] = 0
#            if result.pvalues[i]<(0.025/91282):
#                result.params[i] = result.params[i]
        Coef_net.append(result.params)       
    Coef_avgseg_net.append(Coef_net)
    R2_avgseg_net.append(R2_net)
    
# VIF
seg_type = ['seg'+str(i) for i in np.arange(1,19,1)]
VIF_net = []
for i, st in enumerate(seg_type):
    actdata_net = np.load('/home/dell/Desktop/DNN2Brain/Conv5_data/Nonface_lrdecay_'+st+'.pkl', allow_pickle=True)
    actdata_net = np.vstack((actdata_net[28], actdata_net[49]))
    actdata_net[actdata_net<0] = 0
    actdata_net = np.log(actdata_net+0.01)  
    hrfsignal_net = get_cnn_hrf_signal(actdata_net.T)
    hrfsignal_net = stats.zscore(hrfsignal_net)
    # OLS method to estimate params for Linear regresion
    lineareg = sm.OLS(hrfsignal_net[:,0], hrfsignal_net[:,1])
    result = lineareg.fit()
    # calulate R2
    R2 = metrics.r2_score(hrfsignal_net[:,0], result.predict(hrfsignal_net[:,1]))
    VIF = 1/(1-R2)
    VIF_net.append(VIF) 
plt.plot(VIF_net)
    


from scipy.stats import f
F_critical = f.ppf(1-0.05/91282, 2, 236)

Coef_avgseg_alexnet = np.array(Coef_avgseg_alexnet) 
np.save('/home/dell/Desktop/DNN2Brain/Temp_results/Coef_avgseg_alexnet.npy', Coef_avgseg_alexnet)       
Coefmap_alexnet = np.mean(Coef_avgseg_alexnet, axis=0)
Coefmap_alexnet_orignal = copy.deepcopy(Coefmap_alexnet)
R2_avgseg_alexnet = np.array(R2_avgseg_alexnet)
np.save('/home/dell/Desktop/DNN2Brain/Temp_results/R2_avgseg_alexnet.npy', R2_avgseg_alexnet) 
R2map_alexnet = np.mean(R2_avgseg_alexnet, axis=0)

R2map_alexnet_Ftest = copy.deepcopy(R2map_alexnet)
for voxel in range(91282):                 # Bonferroni R2 test
    r2 = R2map_alexnet_Ftest[voxel]
    F = (r2/2)/((1-r2)/(239-3))
    if F<F_critical:
        R2map_alexnet_Ftest[voxel] = 0
    if F>F_critical:
        R2map_alexnet_Ftest[voxel] = R2map_alexnet_Ftest[voxel]
    
#for voxel in range(91282):                 # Bonferroni t test for Beta
#    for beta in [0,1]:
#        p_value = ttest_1samp(Coef_avgseg_alexnet[:,voxel,beta], 0)[1]
#        if p_value > (0.025/(91282*2)):
#            Coefmap_alexnet[voxel,beta] = 0
#        if p_value < (0.025/(91282*2)):
#            Coefmap_alexnet[voxel,beta] = Coefmap_alexnet[voxel,beta]



Coef_avgseg_net = np.array(Coef_avgseg_net)
np.save('/home/dell/Desktop/DNN2Brain/Temp_results/Coef_avgseg_net.npy', Coef_avgseg_net) 
Coefmap_net = np.mean(Coef_avgseg_net, axis=0)
Coefmap_net_orignal = copy.deepcopy(Coefmap_net)
R2_avgseg_net = np.array(R2_avgseg_net)
np.save('/home/dell/Desktop/DNN2Brain/Temp_results/R2_avgseg_net.npy', R2_avgseg_net) 
R2map_net = np.mean(R2_avgseg_net, axis=0)

R2map_net_Ftest = copy.deepcopy(R2map_net)
for voxel in range(91282):                 # Bonferroni R2 test
    r2 = R2map_net_Ftest[voxel]
    F = (r2/2)/((1-r2)/(239-3))
    if F<F_critical:
        R2map_net_Ftest[voxel] = 0
    if F>F_critical:
        R2map_net_Ftest[voxel] = R2map_net_Ftest[voxel]
        
#for voxel in range(91282):                 # Bonferroni t test for Beta
#    for beta in [0,1]:
#        p_value = ttest_1samp(Coef_avgseg_net[:,voxel,beta], 0)[1]
#        if p_value > (0.025/(91282*2)):
#            Coefmap_net[voxel,beta] = 0
#        if p_value < (0.025/(91282*2)):
#            Coefmap_net[voxel,beta] = Coefmap_net[voxel,beta]

plt.figure()
plt.plot(R2map_alexnet, alpha=0.7)
plt.plot(R2map_net, alpha=0.7)



data_nii = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')
# R2map_alexnet
img = np.array([])
for i in range(245):
    img = np.append(img, R2map_alexnet)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/R2map_alexnet_top2.dtseries.nii')
# R2map_alexnet F_test
img = np.array([])
for i in range(245):
    img = np.append(img, R2map_alexnet_Ftest)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/R2map_alexnet_Ftest_top2.dtseries.nii')
# R2map_net
img = np.array([])
for i in range(245):
    img = np.append(img, R2map_net)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/R2map_net_lrdecay_top2.dtseries.nii')
# R2map_net F_test
img = np.array([])
for i in range(245):
    img = np.append(img, R2map_net_Ftest)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/R2map_net_Ftest_top2.dtseries.nii')
# R2map_diff
R2map_diff = R2map_alexnet-R2map_net
img = np.array([])
for i in range(245):
    img = np.append(img, R2map_diff)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/R2map_diff_top2.dtseries.nii')


data_nii = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')
# Coefmap_alexnet
img = np.array([])
for i in range(245):
    img = np.append(img, Coefmap_alexnet[:,0])
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/Coefmap_alexnet_beta1_top2.dtseries.nii')
img = np.array([])
for i in range(245):
    img = np.append(img, Coefmap_alexnet[:,1])
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/Coefmap_alexnet_beta2_top2.dtseries.nii')
# Coefmap_net
img = np.array([])
for i in range(245):
    img = np.append(img, Coefmap_net[:,0])
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/Coefmap_net_lrdecay_beta1_top2.dtseries.nii')
img = np.array([])
for i in range(245):
    img = np.append(img, Coefmap_net[:,1])
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/Coefmap_net_lrdecay_beta2_top2.dtseries.nii')





########################### R2: FFA vs OFA
from scipy.stats import ttest_ind
import nibabel as nib

def Mask_mapping_32k2wen(mask_32k, Dict_32k_to_wen):
    mask_wen = np.zeros(len(Dict_32k_to_wen))
    for i in list(Dict_32k_to_wen.keys()):
        mask_wen[Dict_32k_to_wen[i]] = mask_32k[i]
    mask_wen = np.where(mask_wen!=0, 1, 0)
    return mask_wen

seg1_1_Atlas = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')
list_of_block = list(seg1_1_Atlas.header.get_index_map(1).brain_models)
CORTEX_Left = list_of_block[0]
CORTEX_Right = list_of_block[1]
Dict_wen_to_32kL = dict()
for vertex in range(CORTEX_Left.index_count):
    Dict_wen_to_32kL[vertex] = CORTEX_Left.vertex_indices[vertex]
Dict_32kL_to_wen = {v:k for k,v in Dict_wen_to_32kL.items()}
Dict_wen_to_32kR = dict()
for vertex in range(CORTEX_Right.index_count):
    Dict_wen_to_32kR[vertex] = CORTEX_Right.vertex_indices[vertex]
Dict_32kR_to_wen = {v:k for k,v in Dict_wen_to_32kR.items()}


ther = -200
mask_lofa = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-lOFA-PRM-fsaverage.func.gii').darrays[0].data
mask_lofa[np.argsort(mask_lofa)[ther:]] = 1
mask_lofa = np.where(mask_lofa==1, 1, 0)
mask_lofa_wen = Mask_mapping_32k2wen(mask_lofa, Dict_32kL_to_wen)
mask_rofa = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-rOFA-PRM-fsaverage.func.gii').darrays[0].data
mask_rofa[np.argsort(mask_rofa)[ther:]] = 1
mask_rofa = np.where(mask_rofa==1, 1, 0)
mask_rofa_wen = Mask_mapping_32k2wen(mask_rofa, Dict_32kR_to_wen)
mask_lpFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-lpFus-PRM-fsaverage.func.gii').darrays[0].data
mask_lpFFA[np.argsort(mask_lpFFA)[ther:]] = 1
mask_lpFFA = np.where(mask_lpFFA==1, 1, 0)
mask_lpFFA_wen = Mask_mapping_32k2wen(mask_lpFFA, Dict_32kL_to_wen)
mask_laFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-laFus-PRM-fsaverage.func.gii').darrays[0].data
mask_laFFA[np.argsort(mask_laFFA)[ther:]] = 1
mask_laFFA = np.where(mask_laFFA==1, 1, 0)
mask_laFFA_wen = Mask_mapping_32k2wen(mask_laFFA, Dict_32kL_to_wen)
mask_rpFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-rpFus-PRM-fsaverage.func.gii').darrays[0].data
mask_rpFFA[np.argsort(mask_rpFFA)[ther:]] = 1
mask_rpFFA = np.where(mask_rpFFA==1, 1, 0)
mask_rpFFA_wen = Mask_mapping_32k2wen(mask_rpFFA, Dict_32kR_to_wen)
mask_raFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-raFus-PRM-fsaverage.func.gii').darrays[0].data
mask_raFFA[np.argsort(mask_raFFA)[ther:]] = 1
mask_raFFA = np.where(mask_raFFA==1, 1, 0)
mask_raFFA_wen = Mask_mapping_32k2wen(mask_raFFA, Dict_32kR_to_wen)


R2_affa_alexnet = R2map_alexnet[np.hstack((np.where(mask_laFFA_wen!=0)[0], np.where(mask_raFFA_wen!=0)[0]+29696))]  
R2_pffa_alexnet = R2map_alexnet[np.hstack((np.where(mask_lpFFA_wen!=0)[0], np.where(mask_rpFFA_wen!=0)[0]+29696))] 
R2_affa_net = R2map_net[np.hstack((np.where(mask_laFFA_wen!=0)[0], np.where(mask_raFFA_wen!=0)[0]+29696))]  
R2_pffa_net = R2map_net[np.hstack((np.where(mask_lpFFA_wen!=0)[0], np.where(mask_rpFFA_wen!=0)[0]+29696))] 
R2_ofa_alexnet = R2map_alexnet[np.hstack((np.where(mask_lofa_wen!=0)[0], np.where(mask_rofa_wen!=0)[0]+29696))]
R2_ofa_net = R2map_net[np.hstack((np.where(mask_lofa_wen!=0)[0], np.where(mask_rofa_wen!=0)[0]+29696))]

plt.figure()
plt.ylabel('R2')
bar_width=0.3
index = np.arange(2) 
plt.bar(index, [np.mean(R2_affa_alexnet), 
                np.mean(R2_affa_net)], 
        yerr=[np.std(R2_affa_alexnet)/R2_affa_alexnet.shape[0],
              np.std(R2_affa_net)/R2_affa_net.shape[0]],
        width=bar_width, label='aFFA')
plt.bar(index+bar_width, [np.mean(R2_pffa_alexnet), 
                np.mean(R2_pffa_net)], 
        yerr=[np.std(R2_pffa_alexnet)/R2_pffa_alexnet.shape[0],
              np.std(R2_pffa_net)/R2_pffa_net.shape[0]],
        width=bar_width, label='pFFA')
plt.bar(index+2*bar_width, [np.mean(R2_ofa_alexnet), 
                          np.mean(R2_ofa_net)],
        yerr=[np.std(R2_ofa_alexnet)/R2_ofa_alexnet.shape[0],
              np.std(R2_ofa_net)/R2_ofa_net.shape[0]],
        width=bar_width, label='OFA')
plt.legend() 
plt.ylim((0,0.15))
plt.xticks([])

import csv
with open('/home/dell/Desktop/DNN2Brain/Temp_results/Area_FFA_OFA.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for i in range(400):
        csv_write.writerow(['aFFA', 
                            R2_affa_alexnet[i], 
                            R2_affa_net[i]])
    for i in range(400):
        csv_write.writerow(['pFFA', 
                            R2_pffa_alexnet[i], 
                            R2_pffa_net[i]]) 
    for i in range(400):
        csv_write.writerow(['OFA', 
                            R2_ofa_alexnet[i], 
                            R2_ofa_net[i]]) 


    
    

######################## Beta: FFA vs OFA
from scipy.stats import ttest_ind

def Mask_mapping_32k2wen(mask_32k, Dict_32k_to_wen):
    mask_wen = np.zeros(len(Dict_32k_to_wen))
    for i in list(Dict_32k_to_wen.keys()):
        mask_wen[Dict_32k_to_wen[i]] = mask_32k[i]
    mask_wen = np.where(mask_wen!=0, 1, 0)
    return mask_wen


seg1_1_Atlas = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')
list_of_block = list(seg1_1_Atlas.header.get_index_map(1).brain_models)
CORTEX_Left = list_of_block[0]
CORTEX_Right = list_of_block[1]
Dict_wen_to_32kL = dict()
for vertex in range(CORTEX_Left.index_count):
    Dict_wen_to_32kL[vertex] = CORTEX_Left.vertex_indices[vertex]
Dict_32kL_to_wen = {v:k for k,v in Dict_wen_to_32kL.items()}
Dict_wen_to_32kR = dict()
for vertex in range(CORTEX_Right.index_count):
    Dict_wen_to_32kR[vertex] = CORTEX_Right.vertex_indices[vertex]
Dict_32kR_to_wen = {v:k for k,v in Dict_wen_to_32kR.items()}


ther = -200
mask_lofa = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-lOFA-PRM-fsaverage.func.gii').darrays[0].data
#mask_lofa = np.where(mask_lofa>=0.1, mask_lofa, 0)
mask_lofa[np.argsort(mask_lofa)[ther:]] = 1
mask_lofa = np.where(mask_lofa==1, 1, 0)
mask_lofa_wen = Mask_mapping_32k2wen(mask_lofa, Dict_32kL_to_wen)
mask_rofa = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-rOFA-PRM-fsaverage.func.gii').darrays[0].data
#mask_rofa = np.where(mask_rofa>=0.1, mask_rofa, 0)
mask_rofa[np.argsort(mask_rofa)[ther:]] = 1
mask_rofa = np.where(mask_rofa==1, 1, 0)
mask_rofa_wen = Mask_mapping_32k2wen(mask_rofa, Dict_32kR_to_wen)
mask_lpFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-lpFus-PRM-fsaverage.func.gii').darrays[0].data
#mask_lpFFA = np.where(mask_lpFFA>=0.1, mask_lpFFA, 0)
mask_lpFFA[np.argsort(mask_lpFFA)[ther:]] = 1
mask_lpFFA = np.where(mask_lpFFA==1, 1, 0)
mask_lpFFA_wen = Mask_mapping_32k2wen(mask_lpFFA, Dict_32kL_to_wen)
mask_laFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-laFus-PRM-fsaverage.func.gii').darrays[0].data
#mask_laFFA = np.where(mask_laFFA>=0.1, mask_laFFA, 0)
mask_laFFA[np.argsort(mask_laFFA)[ther:]] = 1
mask_laFFA = np.where(mask_laFFA==1, 1, 0)
mask_laFFA_wen = Mask_mapping_32k2wen(mask_laFFA, Dict_32kL_to_wen)
mask_rpFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-rpFus-PRM-fsaverage.func.gii').darrays[0].data
#mask_rpFFA = np.where(mask_rpFFA>=0.1, mask_rpFFA, 0)
mask_rpFFA[np.argsort(mask_rpFFA)[ther:]] = 1
mask_rpFFA = np.where(mask_rpFFA==1, 1, 0)
mask_rpFFA_wen = Mask_mapping_32k2wen(mask_rpFFA, Dict_32kR_to_wen)
mask_raFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-raFus-PRM-fsaverage.func.gii').darrays[0].data
#mask_raFFA = np.where(mask_raFFA>=0.1, mask_raFFA, 0)
mask_raFFA[np.argsort(mask_raFFA)[ther:]] = 1
mask_raFFA = np.where(mask_raFFA==1, 1, 0)
mask_raFFA_wen = Mask_mapping_32k2wen(mask_raFFA, Dict_32kR_to_wen)


Beta_lpFFA_alexnet = Coefmap_alexnet_orignal[np.where(mask_lpFFA_wen!=0)[0],:]
Beta_rpFFA_alexnet = Coefmap_alexnet_orignal[np.where(mask_rpFFA_wen!=0)[0]+29696,:]
Beta_pFFA_alexnet = np.vstack((Beta_lpFFA_alexnet, Beta_rpFFA_alexnet))

Beta_laFFA_alexnet = Coefmap_alexnet_orignal[np.where(mask_laFFA_wen!=0)[0],:]
Beta_raFFA_alexnet = Coefmap_alexnet_orignal[np.where(mask_raFFA_wen!=0)[0]+29696,:]
Beta_aFFA_alexnet = np.vstack((Beta_laFFA_alexnet, Beta_raFFA_alexnet))

Beta_lofa_alexnet = Coefmap_alexnet_orignal[np.where(mask_lofa_wen!=0)[0],:]
Beta_rofa_alexnet = Coefmap_alexnet_orignal[np.where(mask_rofa_wen!=0)[0]+29696,:]
Beta_ofa_alexnet = np.vstack((Beta_lofa_alexnet, Beta_rofa_alexnet))


Beta_lpFFA_net = Coefmap_net_orignal[np.where(mask_lpFFA_wen!=0)[0],:]
Beta_rpFFA_net = Coefmap_net_orignal[np.where(mask_rpFFA_wen!=0)[0]+29696,:]
Beta_pFFA_net = np.vstack((Beta_lpFFA_net, Beta_rpFFA_net))

Beta_laFFA_net = Coefmap_net_orignal[np.where(mask_laFFA_wen!=0)[0],:]
Beta_raFFA_net = Coefmap_net_orignal[np.where(mask_raFFA_wen!=0)[0]+29696,:]
Beta_aFFA_net = np.vstack((Beta_laFFA_net, Beta_raFFA_net))

Beta_lofa_net = Coefmap_net_orignal[np.where(mask_lofa_wen!=0)[0],:]
Beta_rofa_net = Coefmap_net_orignal[np.where(mask_rofa_wen!=0)[0]+29696,:]
Beta_ofa_net = np.vstack((Beta_lofa_net, Beta_rofa_net))


plt.figure()
plt.ylabel('Beta')
bar_width=0.3
index = np.arange(6) 
plt.bar(index, [np.mean(Beta_pFFA_alexnet[:,0]), 
                np.mean(Beta_aFFA_alexnet[:,0]),
                np.mean(Beta_ofa_alexnet[:,0]),
                np.mean(Beta_pFFA_net[:,0]),
                np.mean(Beta_aFFA_net[:,0]), 
                np.mean(Beta_ofa_net[:,0])],
        yerr=[np.std(Beta_pFFA_alexnet[:,0])/Beta_pFFA_alexnet.shape[0],
              np.std(Beta_aFFA_alexnet[:,0])/Beta_aFFA_alexnet.shape[0],
              np.std(Beta_ofa_alexnet[:,0])/Beta_ofa_alexnet.shape[0],
              np.std(Beta_pFFA_net[:,0])/Beta_pFFA_net.shape[0],
              np.std(Beta_aFFA_net[:,0])/Beta_aFFA_net.shape[0],
              np.std(Beta_ofa_net[:,0])/Beta_ofa_net.shape[0]],
        width=0.3, label='Beta1')
plt.bar(index+bar_width, [np.mean(Beta_pFFA_alexnet[:,1]), 
                          np.mean(Beta_aFFA_alexnet[:,1]),
                          np.mean(Beta_ofa_alexnet[:,1]),
                          np.mean(Beta_pFFA_net[:,1]),
                          np.mean(Beta_aFFA_net[:,1]),
                          np.mean(Beta_ofa_net[:,1])],
        yerr=[np.std(Beta_pFFA_alexnet[:,1])/Beta_pFFA_alexnet.shape[0],
              np.std(Beta_aFFA_alexnet[:,1])/Beta_aFFA_alexnet.shape[0],
              np.std(Beta_ofa_alexnet[:,1])/Beta_ofa_alexnet.shape[0],
              np.std(Beta_pFFA_net[:,1])/Beta_pFFA_net.shape[0],
              np.std(Beta_aFFA_net[:,1])/Beta_aFFA_net.shape[0],
              np.std(Beta_ofa_net[:,1])/Beta_ofa_net.shape[0]],
        width=0.3, label='Beta2')
plt.legend() 
plt.ylim((0,0.25))
plt.xticks([])

import csv
with open('/home/dell/Desktop/DNN2Brain/Temp_results/DN_Beta_FFA_OFA.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for i in range(400):
        csv_write.writerow(['pFFA', 
                            Beta_pFFA_net[:,0][i], 
                            Beta_pFFA_net[:,1][i]])
    for i in range(400):
        csv_write.writerow(['aFFA', 
                            Beta_aFFA_net[:,0][i], 
                            Beta_aFFA_net[:,1][i]]) 
    for i in range(400):
        csv_write.writerow(['OFA', 
                            Beta_ofa_net[:,0][i], 
                            Beta_ofa_net[:,1][i]]) 
    
with open('/home/dell/Desktop/DNN2Brain/Temp_results/CN_Beta_FFA_OFA.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for i in range(400):
        csv_write.writerow(['pFFA', 
                            Beta_pFFA_alexnet[:,0][i], 
                            Beta_pFFA_alexnet[:,1][i]])
    for i in range(400):
        csv_write.writerow(['aFFA', 
                            Beta_aFFA_alexnet[:,0][i], 
                            Beta_aFFA_alexnet[:,1][i]]) 
    for i in range(400):
        csv_write.writerow(['OFA', 
                            Beta_ofa_alexnet[:,0][i], 
                            Beta_ofa_alexnet[:,1][i]]) 
    


from scipy.stats import ttest_ind
from scipy.stats import zscore
from scipy.stats import norm
def Fisher_z(r):
    return 0.5*np.log((1+r)/(1-r))
def StargZ(fishered_r1,n1, fishered_r2,n2):
    return (fishered_r1-fishered_r2)/np.sqrt((1/(n1-3))+(1/(n2-3)))

    
plt.figure()
plt.bar(np.array([0,1,2,3,4,5]), [np.corrcoef(zscore(Beta_pFFA_alexnet,axis=0).T)[0,1], 
                                  np.corrcoef(zscore(Beta_aFFA_alexnet,axis=0).T)[0,1],
                                  np.corrcoef(zscore(Beta_ofa_alexnet,axis=0).T)[0,1],
                                  np.corrcoef(zscore(Beta_pFFA_net,axis=0).T)[0,1],
                                  np.corrcoef(zscore(Beta_aFFA_net,axis=0).T)[0,1], 
                                  np.corrcoef(zscore(Beta_ofa_net,axis=0).T)[0,1]])
    
# test
p_value = 1-norm.cdf(StargZ(Fisher_z(np.corrcoef(zscore(Beta_pFFA_alexnet,axis=0).T)[0,1]),400,
                            Fisher_z(np.corrcoef(zscore(Beta_pFFA_net,axis=0).T)[0,1]),400))
print(p_value<0.05)

p_value = 1-norm.cdf(StargZ(Fisher_z(np.corrcoef(zscore(Beta_aFFA_alexnet,axis=0).T)[0,1]),400,
                            Fisher_z(np.corrcoef(zscore(Beta_aFFA_net,axis=0).T)[0,1]),400))
print(p_value<0.05)

p_value = 1-norm.cdf(StargZ(Fisher_z(np.corrcoef(zscore(Beta_ofa_alexnet,axis=0).T)[0,1]),400,
                            Fisher_z(np.corrcoef(zscore(Beta_ofa_net,axis=0).T)[0,1]),400))
print(p_value<0.05)



BetaR_pFFA_alexnet = []
BetaR_aFFA_alexnet = []
BetaR_ofa_alexnet = []
BetaR_pFFA_net = []
BetaR_aFFA_net = []
BetaR_ofa_net = []
for seg in range(18):
    Beta_lpFFA_alexnet = Coef_avgseg_alexnet[seg, :, :][np.where(mask_lpFFA_wen!=0)[0],:]
    Beta_rpFFA_alexnet = Coef_avgseg_alexnet[seg, :, :][np.where(mask_rpFFA_wen!=0)[0]+29696,:]
    Beta_pFFA_alexnet = np.vstack((Beta_lpFFA_alexnet, Beta_rpFFA_alexnet))
    
    Beta_laFFA_alexnet = Coef_avgseg_alexnet[seg, :, :][np.where(mask_laFFA_wen!=0)[0],:]
    Beta_raFFA_alexnet = Coef_avgseg_alexnet[seg, :, :][np.where(mask_raFFA_wen!=0)[0]+29696,:]
    Beta_aFFA_alexnet = np.vstack((Beta_laFFA_alexnet, Beta_raFFA_alexnet))
    
    Beta_lofa_alexnet = Coef_avgseg_alexnet[seg, :, :][np.where(mask_lofa_wen!=0)[0],:]
    Beta_rofa_alexnet = Coef_avgseg_alexnet[seg, :, :][np.where(mask_rofa_wen!=0)[0]+29696,:]
    Beta_ofa_alexnet = np.vstack((Beta_lofa_alexnet, Beta_rofa_alexnet))
    
    Beta_lpFFA_net = Coef_avgseg_net[seg, :, :][np.where(mask_lpFFA_wen!=0)[0],:]
    Beta_rpFFA_net = Coef_avgseg_net[seg, :, :][np.where(mask_rpFFA_wen!=0)[0]+29696,:]
    Beta_pFFA_net = np.vstack((Beta_lpFFA_net, Beta_rpFFA_net))
    
    Beta_laFFA_net = Coef_avgseg_net[seg, :, :][np.where(mask_laFFA_wen!=0)[0],:]
    Beta_raFFA_net = Coef_avgseg_net[seg, :, :][np.where(mask_raFFA_wen!=0)[0]+29696,:]
    Beta_aFFA_net = np.vstack((Beta_laFFA_net, Beta_raFFA_net))
    
    Beta_lofa_net = Coef_avgseg_net[seg, :, :][np.where(mask_lofa_wen!=0)[0],:]
    Beta_rofa_net = Coef_avgseg_net[seg, :, :][np.where(mask_rofa_wen!=0)[0]+29696,:]
    Beta_ofa_net = np.vstack((Beta_lofa_net, Beta_rofa_net))
    
    BetaR_pFFA_alexnet.append(np.corrcoef(Beta_pFFA_alexnet.T)[0,1])
    BetaR_aFFA_alexnet.append(np.corrcoef(Beta_aFFA_alexnet.T)[0,1])
    BetaR_ofa_alexnet.append(np.corrcoef(Beta_ofa_alexnet.T)[0,1])
    BetaR_pFFA_net.append(np.corrcoef(Beta_pFFA_net.T)[0,1])
    BetaR_aFFA_net.append(np.corrcoef(Beta_aFFA_net.T)[0,1])
    BetaR_ofa_net.append(np.corrcoef(Beta_ofa_net.T)[0,1])
    
plt.figure()
plt.bar(np.array([0,1,2,3,4,5]), [np.mean(BetaR_pFFA_alexnet), 
                                  np.mean(BetaR_aFFA_alexnet),
                                  np.mean(BetaR_ofa_alexnet),
                                  np.mean(BetaR_pFFA_net),
                                  np.mean(BetaR_aFFA_net), 
                                  np.mean(BetaR_ofa_net)])

    
    

####################### R2: Face vs Obj
from scipy.stats import ttest_ind

ther = -200
mask_lofa = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-lOFA-PRM-fsaverage.func.gii').darrays[0].data
mask_lofa[np.argsort(mask_lofa)[ther:]] = 1
mask_lofa = np.where(mask_lofa==1, 1, 0)
mask_lofa_wen = Mask_mapping_32k2wen(mask_lofa, Dict_32kL_to_wen)
mask_rofa = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-rOFA-PRM-fsaverage.func.gii').darrays[0].data
mask_rofa[np.argsort(mask_rofa)[ther:]] = 1
mask_rofa = np.where(mask_rofa==1, 1, 0)
mask_rofa_wen = Mask_mapping_32k2wen(mask_rofa, Dict_32kR_to_wen)
mask_lpFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-lpFus-PRM-fsaverage.func.gii').darrays[0].data
mask_lpFFA[np.argsort(mask_lpFFA)[ther:]] = 1
mask_lpFFA = np.where(mask_lpFFA==1, 1, 0)
mask_lpFFA_wen = Mask_mapping_32k2wen(mask_lpFFA, Dict_32kL_to_wen)
mask_laFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-laFus-PRM-fsaverage.func.gii').darrays[0].data
mask_laFFA[np.argsort(mask_laFFA)[ther:]] = 1
mask_laFFA = np.where(mask_laFFA==1, 1, 0)
mask_laFFA_wen = Mask_mapping_32k2wen(mask_laFFA, Dict_32kL_to_wen)
mask_rpFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-rpFus-PRM-fsaverage.func.gii').darrays[0].data
mask_rpFFA[np.argsort(mask_rpFFA)[ther:]] = 1
mask_rpFFA = np.where(mask_rpFFA==1, 1, 0)
mask_rpFFA_wen = Mask_mapping_32k2wen(mask_rpFFA, Dict_32kR_to_wen)
mask_raFFA = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-raFus-PRM-fsaverage.func.gii').darrays[0].data
mask_raFFA[np.argsort(mask_raFFA)[ther:]] = 1
mask_raFFA = np.where(mask_raFFA==1, 1, 0)
mask_raFFA_wen = Mask_mapping_32k2wen(mask_raFFA, Dict_32kR_to_wen)

R2_lpFFA_alexnet = R2map_alexnet[np.where(mask_lpFFA_wen!=0)[0]]
R2_rpFFA_alexnet = R2map_alexnet[np.where(mask_rpFFA_wen!=0)[0]+29696]
R2_pFFA_alexnet = np.vstack((R2_lpFFA_alexnet, R2_rpFFA_alexnet))

R2_laFFA_alexnet = R2map_alexnet[np.where(mask_laFFA_wen!=0)[0]]
R2_raFFA_alexnet = R2map_alexnet[np.where(mask_raFFA_wen!=0)[0]+29696]
R2_aFFA_alexnet = np.vstack((R2_laFFA_alexnet, R2_raFFA_alexnet))

R2_lofa_alexnet = R2map_alexnet[np.where(mask_lofa_wen!=0)[0]]
R2_rofa_alexnet = R2map_alexnet[np.where(mask_rofa_wen!=0)[0]+29696]
R2_ofa_alexnet = np.vstack((R2_lofa_alexnet, R2_rofa_alexnet))

R2_lpFFA_net = R2map_net[np.where(mask_lpFFA_wen!=0)[0]]
R2_rpFFA_net = R2map_net[np.where(mask_rpFFA_wen!=0)[0]+29696]
R2_pFFA_net = np.vstack((R2_lpFFA_net, R2_rpFFA_net))

R2_laFFA_net = R2map_net[np.where(mask_laFFA_wen!=0)[0]]
R2_raFFA_net = R2map_net[np.where(mask_raFFA_wen!=0)[0]+29696]
R2_aFFA_net = np.vstack((R2_laFFA_net, R2_raFFA_net))

R2_lofa_net = R2map_net[np.where(mask_lofa_wen!=0)[0]]
R2_rofa_net = R2map_net[np.where(mask_rofa_wen!=0)[0]+29696]
R2_ofa_net = np.vstack((R2_lofa_net, R2_rofa_net))


ther = -300
mask_lLO = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Object/32K/BAA-OR-OvS-lLO-PRM-fsaverage.func.gii').darrays[0].data
mask_lLO[np.argsort(mask_lLO)[ther:]] = 1
mask_lLO = np.where(mask_lLO==1, 1, 0)
mask_lLO_wen = Mask_mapping_32k2wen(mask_lLO, Dict_32kL_to_wen)
mask_lpFs = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Object/32K/BAA-OR-OvS-lpFs-PRM-fsaverage.func.gii').darrays[0].data
mask_lpFs[np.argsort(mask_lpFs)[ther:]] = 1
mask_lpFs = np.where(mask_lpFs==1, 1, 0)
mask_lpFs_wen = Mask_mapping_32k2wen(mask_lpFs, Dict_32kL_to_wen)
mask_rLO = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Object/32K/BAA-OR-OvS-rLO-PRM-fsaverage.func.gii').darrays[0].data
mask_rLO[np.argsort(mask_rLO)[ther:]] = 1
mask_rLO = np.where(mask_rLO==1, 1, 0)
mask_rLO_wen = Mask_mapping_32k2wen(mask_rLO, Dict_32kL_to_wen)
mask_rpFs = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Object/32K/BAA-OR-OvS-rpFs-PRM-fsaverage.func.gii').darrays[0].data
mask_rpFs[np.argsort(mask_rpFs)[ther:]] = 1
mask_rpFs = np.where(mask_rpFs==1, 1, 0)
mask_rpFs_wen = Mask_mapping_32k2wen(mask_rpFs, Dict_32kL_to_wen)

R2_lLO_alexnet = R2map_alexnet[np.where(mask_lLO_wen!=0)[0]]
R2_rLO_alexnet = R2map_alexnet[np.where(mask_rLO_wen!=0)[0]+29696]
R2_LO_alexnet = np.vstack((R2_lLO_alexnet, R2_rLO_alexnet))

R2_lpFs_alexnet = R2map_alexnet[np.where(mask_lpFs_wen!=0)[0]]
R2_rpFs_alexnet = R2map_alexnet[np.where(mask_rpFs_wen!=0)[0]+29696]
R2_pFs_alexnet = np.vstack((R2_lpFs_alexnet, R2_rpFs_alexnet))

R2_lLO_net = R2map_net[np.where(mask_lLO_wen!=0)[0]]
R2_rLO_net = R2map_net[np.where(mask_rLO_wen!=0)[0]+29696]
R2_LO_net = np.vstack((R2_lLO_net, R2_rLO_net))

R2_lpFs_net = R2map_net[np.where(mask_lpFs_wen!=0)[0]]
R2_rpFs_net = R2map_net[np.where(mask_rpFs_wen!=0)[0]+29696]
R2_pFs_net = np.vstack((R2_lpFs_net, R2_rpFs_net))


R2_face_alexnet = np.vstack((R2_pFFA_alexnet, R2_aFFA_alexnet, R2_ofa_alexnet))
R2_face_net = np.vstack((R2_pFFA_net, R2_aFFA_net, R2_ofa_net))
R2_obj_alexnet = np.vstack((R2_LO_alexnet, R2_pFs_alexnet))
R2_obj_net = np.vstack((R2_LO_net, R2_pFs_net))


plt.figure()
plt.ylabel('R2')
bar_width=0.3
index = np.arange(2) 
plt.bar(index, [np.mean(R2_face_alexnet), 
                np.mean(R2_face_net)], 
        yerr=[np.std(R2_face_alexnet)/R2_face_alexnet.shape[0],
              np.std(R2_face_net)/R2_face_net.shape[0]],
        width=0.3, label='Face')
plt.bar(index+bar_width, [np.mean(R2_obj_alexnet), 
                          np.mean(R2_obj_net)],
        yerr=[np.std(R2_obj_alexnet)/R2_obj_alexnet.shape[0],
              np.std(R2_obj_net)/R2_obj_net.shape[0]],
        width=0.3, label='Obj')
plt.legend() 
plt.ylim((0,0.125))
plt.xticks([])

import csv
with open('/home/dell/Desktop/DNN2Brain/Temp_results/Area_Face_Obj.csv', 'a+') as csvfile:
    csv_write = csv.writer(csvfile)
    for i in range(1200):
        csv_write.writerow(['Face_area', 
                            R2_face_alexnet.reshape(-1)[i], 
                            R2_face_net.reshape(-1)[i]])
    for i in range(1200):
        csv_write.writerow(['Obj_area', 
                            R2_obj_alexnet.reshape(-1)[i], 
                            R2_obj_net.reshape(-1)[i]])    


# if we use PMAP in Face and Obj area
mask_lface = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-lMPRM-fsaverage-thr10.func.gii').darrays[0].data
mask_lface_wen = Mask_mapping_32k2wen(mask_lface, Dict_32kL_to_wen)
mask_rface = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-rMPRM-fsaverage-thr10.func.gii').darrays[0].data
mask_rface_wen = Mask_mapping_32k2wen(mask_rface, Dict_32kR_to_wen)
mask_lobj = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Object/32K/BAA-OR-OvS-lMPRM-fsaverage-thr10.func.gii').darrays[0].data
mask_lobj_wen = Mask_mapping_32k2wen(mask_lobj, Dict_32kL_to_wen)
mask_robj = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Object/32K/BAA-OR-OvS-rMPRM-fsaverage-thr10.func.gii').darrays[0].data
mask_robj_wen = Mask_mapping_32k2wen(mask_robj, Dict_32kR_to_wen)

R2_lface_alexnet = R2map_alexnet[np.where(mask_lface_wen!=0)[0]]
R2_rface_alexnet = R2map_alexnet[np.where(mask_rface_wen!=0)[0]+29696]
R2_face_alexnet = np.hstack((R2_lface_alexnet, R2_rface_alexnet))

R2_lobj_alexnet = R2map_alexnet[np.where(mask_lobj_wen!=0)[0]]
R2_robj_alexnet = R2map_alexnet[np.where(mask_robj_wen!=0)[0]+29696]
R2_obj_alexnet = np.hstack((R2_lobj_alexnet, R2_robj_alexnet))

R2_lface_net = R2map_net[np.where(mask_lface_wen!=0)[0]]
R2_rface_net = R2map_net[np.where(mask_rface_wen!=0)[0]+29696]
R2_face_net = np.hstack((R2_lface_net, R2_rface_net))

R2_lobj_net = R2map_net[np.where(mask_lobj_wen!=0)[0]]
R2_robj_net = R2map_net[np.where(mask_robj_wen!=0)[0]+29696]
R2_obj_net = np.hstack((R2_lobj_net, R2_robj_net))


plt.figure()
plt.ylabel('R2')
bar_width=0.3
index = np.arange(2) 
plt.bar(index, [np.mean(R2_face_alexnet), 
                np.mean(R2_face_net)], 
        yerr=[np.std(R2_face_alexnet)/R2_face_alexnet.shape[0],
              np.std(R2_face_net)/R2_face_net.shape[0]],
        width=0.3, label='Face')
plt.bar(index+bar_width, [np.mean(R2_obj_alexnet), 
                          np.mean(R2_obj_net)],
        yerr=[np.std(R2_obj_alexnet)/R2_obj_alexnet.shape[0],
              np.std(R2_obj_net)/R2_obj_net.shape[0]],
        width=0.3, label='Obj')
plt.legend() 
plt.ylim((0,0.125))
plt.xticks([])




#################### Beta: Face vs Obj
mask_lface = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-lMPRM-fsaverage-thr10.func.gii').darrays[0].data
mask_lface_wen = Mask_mapping_32k2wen(mask_lface, Dict_32kL_to_wen)
mask_rface = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-rMPRM-fsaverage-thr10.func.gii').darrays[0].data
mask_rface_wen = Mask_mapping_32k2wen(mask_rface, Dict_32kR_to_wen)
mask_lobj = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Object/32K/BAA-OR-OvS-lMPRM-fsaverage-thr10.func.gii').darrays[0].data
mask_lobj_wen = Mask_mapping_32k2wen(mask_lobj, Dict_32kL_to_wen)
mask_robj = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Object/32K/BAA-OR-OvS-rMPRM-fsaverage-thr10.func.gii').darrays[0].data
mask_robj_wen = Mask_mapping_32k2wen(mask_robj, Dict_32kR_to_wen)

Beta_lface_alexnet = Coefmap_alexnet_orignal[np.where(mask_lface_wen!=0)[0],:]
Beta_rface_alexnet = Coefmap_alexnet_orignal[np.where(mask_rface_wen!=0)[0]+29696,:]
Beta_face_alexnet = np.vstack((Beta_lface_alexnet, Beta_rface_alexnet))

Beta_lobj_alexnet = Coefmap_alexnet_orignal[np.where(mask_lobj_wen!=0)[0],:]
Beta_robj_alexnet = Coefmap_alexnet_orignal[np.where(mask_robj_wen!=0)[0]+29696,:]
Beta_obj_alexnet = np.vstack((Beta_lobj_alexnet, Beta_robj_alexnet))

Beta_lface_net = Coefmap_net_orignal[np.where(mask_lface_wen!=0)[0],:]
Beta_rface_net = Coefmap_net_orignal[np.where(mask_rface_wen!=0)[0]+29696,:]
Beta_face_net = np.vstack((Beta_lface_net, Beta_rface_net))

Beta_lobj_net = Coefmap_net_orignal[np.where(mask_lobj_wen!=0)[0],:]
Beta_robj_net = Coefmap_net_orignal[np.where(mask_robj_wen!=0)[0]+29696,:]
Beta_obj_net = np.vstack((Beta_lobj_net, Beta_robj_net))


plt.figure()
plt.ylabel('Beta')
bar_width=0.3
index = np.arange(4) 
plt.bar(index, [np.mean(Beta_face_alexnet[:,0]), 
                np.mean(Beta_obj_alexnet[:,0]),
                np.mean(Beta_face_net[:,0]),
                np.mean(Beta_obj_net[:,0])], 
        yerr=[np.std(Beta_face_alexnet)/Beta_face_alexnet.shape[0],
              np.std(Beta_face_alexnet)/Beta_face_alexnet.shape[0],
              np.std(Beta_face_net)/Beta_face_net.shape[0],
              np.std(Beta_face_net)/Beta_face_net.shape[0]],
        width=0.3, label='Beta1')
plt.bar(index+bar_width, [np.mean(Beta_face_alexnet[:,1]), 
                          np.mean(Beta_obj_alexnet[:,1]),
                          np.mean(Beta_face_net[:,1]),
                          np.mean(Beta_obj_net[:,1])],
        yerr=[np.std(Beta_obj_alexnet)/Beta_obj_alexnet.shape[0],
              np.std(Beta_obj_alexnet)/Beta_obj_alexnet.shape[0],
              np.std(Beta_obj_net)/Beta_obj_net.shape[0],
              np.std(Beta_obj_net)/Beta_obj_net.shape[0]],
        width=0.3, label='Beta2')
plt.legend() 
plt.ylim((0,0.15))
plt.xticks([])
plt.savefig('/home/dell/Desktop/Beta_avgsegment')


plt.bar(np.array([0,1,2,3]), [np.corrcoef(Beta_face_alexnet.T)[0,1], 
                              np.corrcoef(Beta_obj_alexnet.T)[0,1],
                              np.corrcoef(Beta_face_net.T)[0,1],
                              np.corrcoef(Beta_obj_net.T)[0,1]])












 