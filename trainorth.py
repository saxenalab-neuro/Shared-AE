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
from data import pose_neural_Dataset
from multi_loss import SupConLoss,csLoss

from model1d import SupConClipResNet1d
from modeldecodeorth1 import Decoder
from util import  AverageMeter,save_model,warmup_learning_rate,adjust_learning_rate,TwoDropTransform_twomodal
import time 
from sklearn.utils import shuffle


jobid=os.getenv('SLURM_ARRAY_TASK_ID')

jobid=int(jobid)
param={i:[] for i in range(2)}
# param[0]=['image',256,1,0.07,5,50,15]#####weight:5, 3
# param[1]=['image',256,1,0.07,5,40,15]#####weight:5, 3
# param[2]=['image',256,1,0.07,5,60,15]#####weight:5, 3
# param[3]=['image',256,1,0.07,5,70,15]#####weight:5, 3
# param[4]=['image',256,1,0.07,5,80,15]#####weight:5, 3
# param[5]=['image',256,1,0.07,5,90,15]#####weight:5, 3
# param[6]=['image',256,1,0.07,5,30,15]#####weight:5, 3
# param[7]=['image',256,1,0.07,5,20,15]#####weight:5, 3
# param[8]=['image',256,1,0.07,5,10,15]#####weight:5, 3
# param[0]=['image',256,1,0.07,5,90,5]#####weight:5, 3
# param[1]=['image',256,1,0.07,5,90,10]#####weight:5, 3
# param[2]=['image',256,1,0.07,10,90,10]#####weight:5, 3
# param[3]=['image',256,1,0.07,10,90,15]#####weight:5, 3

# param[0]=['image',256,1,0.07,5,90,20]#####weight:5, 3
# param[1]=['image',256,1,0.07,5,90,15]#####weight:5, 3
# param[2]=['image',256,1,0.07,10,70,15]#####weight:5, 3
# param[3]=['image',256,1,0.07,10,80,15]#####weight:5, 3
# param[4]=['image',256,1,0.07,10,60,15]#####weight:5, 3
# param[5]=['image',256,1,0.07,10,50,15]#####weight:5, 3

# param[6]=['image',256,1,0.07,5,70,15]#####weight:5, 3
# param[7]=['image',256,1,0.07,5,80,15]#####weight:5, 3
# param[8]=['image',256,1,0.07,5,60,15]#####weight:5, 3
# param[9]=['image',256,1,0.07,5,50,15]#####weight:5, 3
param[0]=['image',256,1,0.07,10,10,15]#####weight:5, 3
param[1]=['image',256,1,0.07,10,20,15]#####weight:5, 3
param[2]=['image',256,1,0.07,10,30,15]#####weight:5, 3
param[3]=['image',256,1,0.07,10,40,15]#####weight:5, 3
param[4]=['image',256,1,0.07,10,50,15]#####weight:5, 3
param[5]=['image',256,1,0.07,10,60,15]#####weight:5, 3

param[6]=['image',256,1,0.07,10,70,15]#####weight:5, 3
param[7]=['image',256,1,0.07,10,80,15]#####weight:5, 3
param[8]=['image',256,1,0.07,10,90,15]#####weight:5, 3
# param[9]=['image',256,1,0.07,5,50,15]#####weight:5, 3
weight=param[jobid][4]
dims=param[jobid][5]
KS=param[jobid][6]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if param[jobid][2]==1:
    torch.manual_seed(1)
else:
    torch.manual_seed(2)

# path='/blue/npadillacoreano/yidaiyao/calms21/'
modelpath='/gpfs/radev/project/saxena/dy274/social/'
W=8
# hdf5_file ='zs_td_resize_seq_mouse_30_window_9_516session.hdf5'#'seq_mouse_30_window_9.hdf5'
# hdf5_file=modelpath+'mnist25more.hdf5'
# hdf5_file=modelpath+'mnist25more_resize_sine.hdf5'
# hdf5_file=modelpath+'mnist_all_more_resize_sine.hdf5'
hdf5_file=modelpath+'social_pose_new.hdf5'
data1 = h5py.File(hdf5_file, 'r')
hdf5_file=modelpath+'social_neural_new.hdf5'
data2 = h5py.File(hdf5_file, 'r')


# data.keys()
arg=50

# alllabels=list(data['neural'].keys())
alllabels=[]

for i in range (0,32620-20,100):
    for j in range(i,i+63):
        alllabels.append(str(j))
    
alllabels=alllabels*arg
drop=2
newW=W-drop
batch_size=param[jobid][1]
temperature=param[jobid][3]#0.07
flag='model/more_new_argtemperature_'+str(temperature)+'randomseed_'+str(param[jobid][2])+'weight_'+str(weight)+'dim_'+str(dims)+'_KS_'+str(KS)
if_label=False
######setup label####
# GN=47
# label=np.zeros((95))
# v=95//GN

# for j in range (GN):
#     label[v*j:v*(j+1)]=np.ones((len(label[v*j:v*(j+1)])))*j
    
# label[-1]=47
####################

dataset=pose_neural_Dataset(alllabels,data1,data2,TwoDropTransform_twomodal(W,newW))

    
    
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
image_private_dim,neural_private_dim=dims,dims

embedding_dim, channels,bottleneck_dim,image_latent_dim,neural_latent_dim=512,newW,dims+image_private_dim,dims,dims
neural_dim=np.array(data2['neural'][alllabels[0]]).shape[-1]
pose_dim=np.array(data1['pose'][alllabels[0]]).shape[-1]
model_image=SupConClipResNet1d(in_channel=newW,name1='resnet18', name2='neural_resnet18',flag='image')
model_neural=SupConClipResNet1d(in_channel=newW,name1='resnet18', name2='neural_resnet18',flag='neural')

model_decoder=Decoder(embedding_dim, channels,bottleneck_dim,image_latent_dim,neural_latent_dim,neural_dim,image_private_dim,neural_private_dim,pose_dim)


optimizer_image = optim.Adam(model_image.parameters(), lr=1e-4)
optimizer_neural = optim.Adam(model_neural.parameters(), lr=1e-4)
optimizer = optim.Adam(model_decoder.parameters(), lr=1e-4)


model_image,_=set_model(model_image,temperature,'image')
model_neural, criterion=set_model(model_neural,temperature,'neural')
model_decoder, _=set_model(model_decoder,temperature,'decoder')


mse_criterion=nn.MSELoss()
mse_criterion = mse_criterion.to(device)


cs_criterion=csLoss(KS)
cs_criterion = cs_criterion.to(device)

epochs=500
save_freq=50
save_folder1=flag+'/pose_enocder_'+str(batch_size)+'_'+str(param[jobid][2])+'_'+str(temperature)+'_model/'
os.makedirs(save_folder1, exist_ok=True)
save_folder2=flag+'/neural_encoder_'+str(batch_size)+'_'+str(param[jobid][2])+'_'+str(temperature)+'_model/'
os.makedirs(save_folder2, exist_ok=True)
save_folderdecoder=flag+'/decoder_'+str(batch_size)+'_'+str(param[jobid][2])+'_'+str(temperature)+'_model/'
os.makedirs(save_folderdecoder, exist_ok=True)

warm_epochs,warmup_from,warmup_to=30,1e-2,1e-3
warm=False
fff=True
ccc=True
def train(loader, model_image,model_neural, model_decoder,mse_criterion, optimizer,epoch,epochs,warm,warm_epochs,warmup_from,warmup_to,optimizer_image,optimizer_neural,learning_rate=5e-3,cosine=True,lr_decay_rate=1e-7,print_freq=5):


    model_decoder.train()
    model_image.train()
    model_neural.train()
    
    
    losses = AverageMeter()
    cslosses = AverageMeter()
    cslosses2 = AverageMeter()
    cslosses3 = AverageMeter()
    cslosses4 = AverageMeter()
    
    
    
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
        
        data_time.update(time.time() - end)
        images=data[0]
        neural=data[1]
        
        images=images.to(device).float()
        neural=neural.to(device).float()
        
        features_image = model_image(images)
        features_neural = model_neural(neural)
        

        global_latent,image_feature,neural_feature,image_pred,neural_pred,image_previate,neural_previate=model_decoder(features_image,features_neural)
        
        bs=images.shape[0]

        # features1 = torch.cat([features_image.unsqueeze(1), features_neural.unsqueeze(1)], dim=1)
        # features2 = torch.cat([features_neural.unsqueeze(1), features_image.unsqueeze(1)], dim=1)
        loss1=mse_criterion(image_pred,images)
        # print(neural.shape,neural_pred.shape)
        # loss2=mse_criterion(neural_pred[:,-6:],neural[:,-6:])
        loss2=mse_criterion(neural_pred[:,:],neural[:,:])
        
        loss3=cs_criterion(image_feature,neural_feature)
        if ccc and fff:
            loss5=cs_criterion(image_feature,image_previate)
            cslosses3.update(loss5.item(), bs)
            
            loss6=cs_criterion(neural_feature,neural_previate)
            cslosses4.update(loss6.item(), bs)
            
            loss4=cs_criterion(image_previate,neural_previate)
            cslosses2.update(loss4.item(), bs)
            
            loss=loss1*pose_dim*newW+loss2*neural_dim*newW+loss3*weight+weight/loss4+weight/loss5+weight/loss6
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


# checkpoint1 = torch.load(modelpath+'cl/models/'+'pose_'+str(batch_size)+'_seed'+str(param[jobid][2])+'_model/'+'last.pth')##30,10
# checkpoint2 = torch.load(modelpath+'cl/models/neural_'+str(batch_size)+'_seed'+str(param[jobid][2])+'_model/'+'ckpt_epoch_40.pth')


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
    loss = train(loader, model_image,model_neural,model_decoder,mse_criterion, optimizer, epoch,epochs,warm,warm_epochs,warmup_from,warmup_to,optimizer_image,optimizer_neural)
    time2 = time.time()
    print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


    if epoch % save_freq == 0:
        save_file1 = os.path.join(save_folder1, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        save_file2 = os.path.join(save_folder2, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        save_file3 = os.path.join(save_folderdecoder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        
        
        save_model(model_image, optimizer_image, epoch, save_file1)
        save_model(model_neural, optimizer_neural, epoch, save_file2)
        save_model(model_decoder, optimizer, epoch, save_file3)
        
        

# save the last model
save_file1 = os.path.join(save_folder1, 'last.pth')
save_file2 = os.path.join(save_folder2, 'last.pth')
save_file3 = os.path.join(save_folderdecoder, 'last.pth')


save_model(model_image, optimizer_image, epochs, save_file1)
save_model(model_neural, optimizer_neural, epoch, save_file2)
save_model(model_decoder, optimizer, epoch, save_file3)

