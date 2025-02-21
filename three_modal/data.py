import torch
import random 
import numpy as np
from collections import deque
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from util import TwoCropTransform

class image_neural_Dataset(torch.utils.data.Dataset):
    def __init__(self, allindex,data1,data2,data3,transform):
        'Initialization'
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        
        self.allindex = allindex
        self.transform=transform
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.allindex)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.allindex[index]

        # Load data and get label
        image1 = np.array(self.data1['image_v1'][ID])
        image2 = np.array(self.data1['image_v2'][ID])
        image = np.array(np.concatenate((image1,image2),axis=-1)/255)
        neural = np.array(self.data2['neural'][ID])
        pose = np.array(self.data3['pose'][ID])[:,[0,1,2,3,5,7,8]]
        
        
        
        return self.transform(image,neural,pose)
    
    #image[keep1],neural[keep1]

# class neuralDataset(torch.utils.data.Dataset):
#     def __init__(self, allindex,data,W,newW):
#         'Initialization'
#         self.data = data
#         self.allindex = allindex
#         self.W=W
#         self.newW=newW
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.allindex)

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.allindex[index]

#         # Load data and get label
#         neural = np.array(self.data['neural'][ID]) ##image size:(256,128)
#         # neural = neural.reshape(self.W,14,15)
#         # neural=np.expand_dims(neural,axis=-1)
#         # image2 = np.array(self.data['image_v2'][ID])
#         # image = np.array(np.concatenate((image1,image2),axis=-1)/255)
#         keep1=sorted(random.sample(deque(np.arange(self.W)),self.newW))
#         keep2=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        
#         return neural[keep1],neural[keep2]

