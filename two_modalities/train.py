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
from two_modalities.data import pose_neural_Dataset
from two_modalities.multi_loss import csLoss
from two_modalities.model_1d import ResNet1d
from two_modalities.modeldecodeorth1 import Decoder
from two_modalities.util import  AverageMeter,save_model,warmup_learning_rate,adjust_learning_rate,TwoDropTransform_twomodal
import time 
from sklearn.utils import shuffle



def train_model(param):

    weight1=param[4]
    weight2=param[5]
    weight3=param[6]
    weight4=param[7]

    dims=param[2]
    KS=param[3]
    W=param[8]
    drop=param[9]
    print_freq=param[13]
    epochs=param[14]
    save_freq=param[15]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if param[1]==1:
        torch.manual_seed(1)
    else:
        torch.manual_seed(2)

    # path='/blue/npadillacoreano/yidaiyao/calms21/'
    modelpath=param[10]#'/gpfs/radev/project/saxena/dy274/sharedae/headfixed/'

    hdf5_file=modelpath+param[11]#'posedata1d.hdf5'
    data1 = h5py.File(hdf5_file, 'r')
    hdf5_file=modelpath+param[12]#'neu_9_all.hdf5'
    data2 = h5py.File(hdf5_file, 'r')

    # data.keys()
    arg=1
    alllabels=list(data1['pose'].keys())


    alllabels=alllabels*arg
    newW=W-drop
    batch_size=param[0]

    flag='models/randomseed_'+str(param[0])+'dim_'+str(dims)+'_KS_'+str(dims)+'weight1_'+str(weight1)+'weight2_'+str(weight2)+'weight3_'+str(weight3)+'weight4_'+str(weight4)



    dataset=pose_neural_Dataset(alllabels,data1,data2,TwoDropTransform_twomodal(W,newW))

        
        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


    def set_model(model,flag):    


        # enable synchronized Batch Normalization


        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

                    
            model = model.to(device)
            cudnn.benchmark = True

        return model
    image_private_dim,neural_private_dim=dims,dims

    embedding_dim, channels,bottleneck_dim,image_latent_dim,neural_latent_dim=512,newW,dims+image_private_dim,dims,dims
    neural_dim=np.array(data2['neural'][alllabels[0]]).shape[-1]
    pose_dim=np.array(data1['pose'][alllabels[0]]).shape[-1]


    model_image=ResNet1d(in_channel=newW,name1='resnet18', name2='neural_resnet18',flag='pose')
    model_neural=ResNet1d(in_channel=newW,name1='resnet18', name2='neural_resnet18',flag='neural')

    model_decoder=Decoder(embedding_dim, channels,bottleneck_dim,image_latent_dim,neural_latent_dim,neural_dim,image_private_dim,neural_private_dim,pose_dim)



    optimizer_image = optim.Adam(model_image.parameters(), lr=1e-4)
    optimizer_neural = optim.Adam(model_neural.parameters(), lr=1e-4)
    optimizer = optim.Adam(model_decoder.parameters(), lr=1e-4)


    model_image=set_model(model_image,'image')
    model_neural=set_model(model_neural,'neural')
    model_decoder=set_model(model_decoder,'decoder')


    mse_criterion=nn.MSELoss()
    mse_criterion = mse_criterion.to(device)


    cs_criterion=csLoss(KS)
    cs_criterion = cs_criterion.to(device)

    save_folder1=flag+'/pose_enocder_'+str(batch_size)+'_'+str(param[2])+'_'+'_model/'
    os.makedirs(save_folder1, exist_ok=True)
    save_folder2=flag+'/neural_encoder_'+str(batch_size)+'_'+str(param[2])+'_'+'_model/'
    os.makedirs(save_folder2, exist_ok=True)
    save_folder3=flag+'/decoder_'+str(batch_size)+'_'+str(param[2])+'_'+'_model/'
    os.makedirs(save_folder3, exist_ok=True)

    fff=True
    ccc=True

    def train(loader, model_image,model_neural, model_decoder,mse_criterion, optimizer,epoch,optimizer_image,optimizer_neural,print_freq):


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
            

            _,image_feature,neural_feature,image_pred,neural_pred,image_previate,neural_previate=model_decoder(features_image,features_neural)
            
            bs=images.shape[0]

            # print(image_pred.shape,images.shape)
            loss1=mse_criterion(image_pred,images)
            loss2=mse_criterion(neural_pred[:,:],neural[:,:])
            
            loss3=cs_criterion(image_feature,neural_feature)
            if ccc and fff:
                loss5=cs_criterion(image_feature,image_previate)
                cslosses3.update(loss5.item(), bs)
                
                loss6=cs_criterion(neural_feature,neural_previate)
                cslosses4.update(loss6.item(), bs)
                
                loss4=cs_criterion(image_previate,neural_previate)
                cslosses2.update(loss4.item(), bs)
                
                loss=loss1*pose_dim*newW+loss2*neural_dim*newW+loss3*weight1+weight2/loss4+weight3/loss5+weight4/loss6
            #######################################
            elif fff and not ccc:
                loss4=cs_criterion(image_previate,neural_previate)
                cslosses2.update(loss4.item(), bs)
                # loss=loss1*28*28*newW+loss2*neural_dim*newW+loss3*weight+weight/loss4+weight/loss5+weight/loss6
                
                loss=loss1*pose_dim*newW+loss2*neural_dim*newW+loss3*weight1+weight2/loss4
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



    for param in model_image.parameters():
        param.requires_grad = True
    for param in model_neural.parameters():
        param.requires_grad = True

    for epoch in range(1, epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch) ###later

        # train for one epoch
        time1 = time.time()
        loss = train(loader, model_image,model_neural, model_decoder,mse_criterion, optimizer,epoch,optimizer_image,optimizer_neural,print_freq)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        if epoch % save_freq == 0:
            save_file1 = os.path.join(save_folder1, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_file2 = os.path.join(save_folder2, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_file3 = os.path.join(save_folder3, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            
            
            save_model(model_image, optimizer_image, epoch, save_file1)
            save_model(model_neural, optimizer_neural, epoch, save_file2)
            save_model(model_decoder, optimizer, epoch, save_file3)
            
            

    # save the last model
    save_file1 = os.path.join(save_folder1, 'last.pth')
    save_file2 = os.path.join(save_folder2, 'last.pth')
    save_file3 = os.path.join(save_folder3, 'last.pth')


    save_model(model_image, optimizer_image, epochs, save_file1)
    save_model(model_neural, optimizer_neural, epoch, save_file2)
    save_model(model_decoder, optimizer, epoch, save_file3)

