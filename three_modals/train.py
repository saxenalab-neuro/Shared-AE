import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from collections import deque
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import h5py
import random
from data import image_neural_Dataset
from multi_loss import SupConLoss,csLoss,multicsLoss
from model import SupConClipResNet

from model_1d import SupConClipResNet1d
from modeldecodeorth2 import Decoder
from util import  AverageMeter,save_model,warmup_learning_rate,adjust_learning_rate,DropTransform_threemodal
import time 
from sklearn.utils import shuffle


jobid=os.getenv('SLURM_ARRAY_TASK_ID')

jobid=int(jobid)
param={i:[] for i in range(2)}
param[0]=['image',256,1,0.07,3,120,5]#####weight:5, 3
param[1]=['image',256,1,0.07,3,110,5]#####weight:5, 3
param[2]=['image',256,1,0.07,3,100,5]#####weight:5, 3
param[3]=['image',256,1,0.07,3,90,5]#####weight:5, 3
param[4]=['image',256,1,0.07,3,50,5]#####weight:5, 3
param[5]=['image',256,1,0.07,3,55,5]#####weight:5, 3
###########################################
param[6]=['image',256,1,0.07,3,60,5]#####weight:5, 3
param[7]=['image',256,1,0.07,3,65,5]#####weight:5, 3
param[8]=['image',256,1,0.07,3,70,5]#####weight:5, 3
param[9]=['image',256,1,0.07,3,75,5]#####weight:5, 3
param[10]=['image',256,1,0.07,3,80,5]#####weight:5, 3
param[11]=['image',256,1,0.07,3,85,5]#####weight:5, 3


weight=param[jobid][4]
dims=param[jobid][5]
KS=param[jobid][6]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if param[jobid][2]==1:
    torch.manual_seed(1)
else:
    torch.manual_seed(2)
######edit this for model path####
modelpath='headfixed/'
#######
W=9

hdf5_file=modelpath+'dataw9.hdf5'
data1 = h5py.File(hdf5_file, 'r')
hdf5_file=modelpath+'neu_9_all.hdf5'
data2 = h5py.File(hdf5_file, 'r')
hdf5_file=modelpath+'posedata1d.hdf5'
data3 = h5py.File(hdf5_file, 'r')
# data.keys()
arg=1
alllabels=deque()
# alllabels=list(data1['pose'].keys())
NN=388
# trainidx=np.load('trainidx.npy')
for tt in range(100):
    for ww in range (189-W):
        alllabels.append(str(tt)+'_'+str(ww))
for tt in range(516-288,516):
    for ww in range (189-W):
        alllabels.append(str(tt)+'_'+str(ww))
        
alllabels=alllabels*arg
drop=2
newW=W-drop
batch_size=param[jobid][1]
temperature=param[jobid][3]#0.07
flag='models/10newall_'+str(temperature)+'randomseed_'+str(param[jobid][2])+'weight_'+str(weight)+'dim_'+str(dims)+'_KS_'+str(dims)
if_label=False
######setup label####
# GN=47
# label=np.zeros((95))
# v=95//GN

# for j in range (GN):
#     label[v*j:v*(j+1)]=np.ones((len(label[v*j:v*(j+1)])))*j
    
# label[-1]=47
####################

dataset=image_neural_Dataset(alllabels,data1,data2,data3,DropTransform_threemodal(W,newW))

    
    
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


def set_model(model,temperature,flag):    
    criterion = SupConLoss(temperature=temperature)

    # enable synchronized Batch Normalization


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

                
        model = model.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True

    return model, criterion
image_private_dim,neural_private_dim,pose_private_dim=dims,dims,dims

embedding_dim, channels,bottleneck_dim,image_latent_dim,neural_latent_dim,pose_latent_dim=512,newW,dims+image_private_dim,dims,dims,dims
neural_dim=np.array(data2['neural'][alllabels[0]]).shape[-1]
pose_dim=7#np.array(data1['pose'][alllabels[0]]).shape[-1]


model_image=SupConClipResNet(in_channel=newW,name1='resnet18', name2='neural_resnet18',flag='image')
model_neural=SupConClipResNet1d(in_channel=newW,name1='resnet18', name2='neural_resnet18',flag='neural')
model_pose=SupConClipResNet1d(in_channel=newW,name1='resnet18', name2='neural_resnet18',flag='pose')

model_decoder=Decoder(embedding_dim, channels,bottleneck_dim,image_latent_dim,neural_latent_dim,neural_dim,image_private_dim,neural_private_dim,pose_latent_dim,pose_dim,pose_private_dim)


optimizer_image = optim.Adam(model_image.parameters(), lr=1e-4)
optimizer_neural = optim.Adam(model_neural.parameters(), lr=1e-4)
optimizer_pose = optim.Adam(model_pose.parameters(), lr=1e-4)

optimizer = optim.Adam(model_decoder.parameters(), lr=1e-4)


model_image,_=set_model(model_image,temperature,'image')
model_neural, criterion=set_model(model_neural,temperature,'neural')
model_pose, criterion=set_model(model_pose,temperature,'pose')

model_decoder, _=set_model(model_decoder,temperature,'decoder')


mse_criterion=nn.MSELoss()
mse_criterion = mse_criterion.to(device)


cs_criterion=csLoss(15)
cs_criterion = cs_criterion.to(device)

multics_criterion=multicsLoss(15)
multics_criterion = multics_criterion.to(device)

epochs=500
save_freq=20
save_folder1=flag+'/image_enocder_'+str(batch_size)+'_'+str(param[jobid][2])+'_'+str(temperature)+'_model/'
os.makedirs(save_folder1, exist_ok=True)
save_folder2=flag+'/neural_encoder_'+str(batch_size)+'_'+str(param[jobid][2])+'_'+str(temperature)+'_model/'
os.makedirs(save_folder2, exist_ok=True)
save_folder3=flag+'/pose_encoder_'+str(batch_size)+'_'+str(param[jobid][2])+'_'+str(temperature)+'_model/'
os.makedirs(save_folder3, exist_ok=True)

save_folderdecoder=flag+'/decoder_'+str(batch_size)+'_'+str(param[jobid][2])+'_'+str(temperature)+'_model/'
os.makedirs(save_folderdecoder, exist_ok=True)

warm_epochs,warmup_from,warmup_to=30,1e-2,1e-3
warm=False
fff=True
ccc=True

def train(loader, model_image,model_neural,model_pose, model_decoder,mse_criterion,cs_criterion,multics_criterion, optimizer,epoch,epochs,warm,warm_epochs,warmup_from,warmup_to,optimizer_image,optimizer_neural,optimizer_pose,learning_rate=1e-4,cosine=True,lr_decay_rate=1e-7,print_freq=50):


    model_decoder.train()
    model_image.train()
    model_neural.train()
    model_pose.train()
    
    
    losses = AverageMeter()
    cslosses = AverageMeter()
    cslosses2 = AverageMeter()
    cslosses3 = AverageMeter()
    cslosses4 = AverageMeter()
    cslosses5 = AverageMeter()
    
    
    
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    end = time.time()
    label=None

    for batch_idx, (data) in enumerate(loader):###same image with different argumentation 
        
        # adjust_learning_rate(learning_rate,cosine, optimizer_image, epoch,epochs,lr_decay_rate)
        # adjust_learning_rate(learning_rate,cosine, optimizer_neural, epoch,epochs,lr_decay_rate)
        

        optimizer.zero_grad()
        optimizer_image.zero_grad()
        optimizer_neural.zero_grad()
        optimizer_pose.zero_grad()
        
        
        
        data_time.update(time.time() - end)
        images=data[0]
        neural=data[1]
        pose=data[2]
        
        
        images=images.to(device).float()
        neural=neural.to(device).float()
        pose=pose.to(device).float()
        
        
        features_image = model_image(images)
        features_neural = model_neural(neural)
        features_pose = model_pose(pose)
        
        

        global_latent,image_feature,neural_feature,pose_feature,image_pred,neural_pred,pose_pred,image_previate,neural_previate,pose_previate=model_decoder(features_image,features_neural,features_pose)
        
        bs=images.shape[0]

        # features1 = torch.cat([features_image.unsqueeze(1), features_neural.unsqueeze(1)], dim=1)
        # features2 = torch.cat([features_neural.unsqueeze(1), features_image.unsqueeze(1)], dim=1)
        loss1=mse_criterion(image_pred,images)
        # print(neural.shape,neural_pred.shape)
        # loss2=mse_criterion(neural_pred[:,-6:],neural[:,-6:])
        loss2=mse_criterion(neural_pred[:,:],neural[:,:])
        losspose=mse_criterion(pose_pred[:,:],pose[:,:])
        
        
        loss3=multics_criterion(image_feature,neural_feature,pose_feature)
        
        if ccc and fff:
            loss5=cs_criterion(image_feature,image_previate)
            cslosses3.update(loss5.item(), bs)
            
            loss6=cs_criterion(neural_feature,neural_previate)
            cslosses4.update(loss6.item(), bs)
            loss7=cs_criterion(pose_feature,pose_previate)
            cslosses5.update(loss7.item(), bs)
            
            loss4=multics_criterion(image_previate,neural_previate,pose_previate)
            cslosses2.update(loss4.item(), bs)
            
            loss=loss1*128*64+losspose*pose_dim*newW+loss2*neural_dim*newW+loss3*weight*10+weight/loss4+weight/loss5+weight/loss6+weight/loss7
        #######################################
        elif fff and not ccc:
            loss4=cs_criterion(image_previate,neural_previate)
            cslosses2.update(loss4.item(), bs)
            # loss=loss1*28*28*newW+loss2*neural_dim*newW+loss3*weight+weight/loss4+weight/loss5+weight/loss6
            
            loss=loss1*pose_dim*newW+loss2*neural_dim*newW+loss3*weight+weight/loss4
            # loss=loss1*28*28*newW+loss2*neural_dim*newW+1/loss3+loss4
        
        else:
            loss=loss1*pose_dim*newW+loss2*neural_dim*newW+loss3
            
        losses.update(loss.item(), bs)
        cslosses.update(loss3.item(), bs)
        
        

        loss.backward()
        optimizer.step()
        optimizer_image.step()
        optimizer_neural.step()
        optimizer_pose.step()
        

        
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        ###print
        if fff and not ccc:
            if (batch_idx + 1) % print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'
                      'csloss {csloss.val:.3f} ({csloss.avg:.3f})'
                      'csloss2 {csloss2.val:.3f} ({csloss2.avg:.3f})'.format(
                       epoch, batch_idx + 1, len(loader), batch_time=batch_time,
                       data_time=data_time, loss=losses,csloss=cslosses,csloss2=cslosses2))
                sys.stdout.flush()
        elif ccc and fff:
            if (batch_idx + 1) % print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'
                      'csloss {csloss.val:.3f} ({csloss.avg:.3f})'
                      'csloss2 {csloss2.val:.3f} ({csloss2.avg:.3f})'
                      'csloss3 {csloss3.val:.3f} ({csloss3.avg:.3f})'
                      'csloss4 {csloss4.val:.3f} ({csloss4.avg:.3f})'.format(
                       epoch, batch_idx + 1, len(loader), batch_time=batch_time,
                       data_time=data_time, loss=losses,csloss=cslosses,csloss2=cslosses2,csloss3=cslosses3,csloss4=cslosses4))
                sys.stdout.flush()
        else:
            if (batch_idx + 1) % print_freq == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'loss {loss.val:.3f} ({loss.avg:.3f})'
                          'csloss {csloss.val:.3f} ({csloss.avg:.3f})'.format(
                           epoch, batch_idx + 1, len(loader), batch_time=batch_time,
                           data_time=data_time, loss=losses,csloss=cslosses))
                    sys.stdout.flush()

    return losses.avg

# checkpoint1 = torch.load(modelpath+'ckpt_epoch_27.pth')##30,10

# checkpoint2 = torch.load(modelpath+'cl/models/neural_'+str(batch_size)+'_seed'+str(param[jobid][2])+'/'+'ckpt_epoch_10.pth')


# model_image.load_state_dict(checkpoint1['model'], strict=False)
# model_neural.load_state_dict(checkpoint2['model'], strict=False)
# optimizer_image.load_state_dict(checkpoint1['optimizer'])
# optimizer_neural.load_state_dict(checkpoint2['optimizer'])

for param in model_image.parameters():
    param.requires_grad = True
for param in model_neural.parameters():
    param.requires_grad = True

for epoch in range(1, epochs + 1):
    # adjust_learning_rate(opt, optimizer, epoch) ###later

    # train for one epoch
    time1 = time.time()
    loss = train(loader, model_image,model_neural,model_pose, model_decoder,mse_criterion,cs_criterion,multics_criterion, optimizer,epoch,epochs,warm,warm_epochs,warmup_from,warmup_to,optimizer_image,optimizer_neural,optimizer_pose)
    time2 = time.time()
    print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


    if epoch % save_freq == 0:
        save_file1 = os.path.join(save_folder1, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        save_file2 = os.path.join(save_folder2, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        save_file4 = os.path.join(save_folder3, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        
        save_file3 = os.path.join(save_folderdecoder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        
        
        save_model(model_image, optimizer_image, epoch, save_file1)
        save_model(model_neural, optimizer_neural, epoch, save_file2)
        save_model(model_decoder, optimizer, epoch, save_file3)
        save_model(model_pose, optimizer_pose, epoch, save_file4)
        
        
        

# save the last model
save_file1 = os.path.join(save_folder1, 'last.pth')
save_file2 = os.path.join(save_folder2, 'last.pth')
save_file3 = os.path.join(save_folderdecoder, 'last.pth')
save_file4 = os.path.join(save_folder3, 'last.pth')



save_model(model_image, optimizer_image, epochs, save_file1)
save_model(model_neural, optimizer_neural, epoch, save_file2)
save_model(model_decoder, optimizer_decoder, epoch, save_file3)
save_model(model_decoder, optimizer_pose, epoch, save_file4)


