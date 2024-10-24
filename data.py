import torch
import random 
import numpy as np
from collections import deque
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler



    

class neuralDataset1d(torch.utils.data.Dataset):
    def __init__(self, allindex,data,TwoDropTransform):
        'Initialization'
        self.data = data
        self.allindex = allindex
        self.transform=TwoDropTransform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.allindex)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.allindex[index]

        # Load data and get label
        neural = np.array(self.data['neural'][ID]) ##image size:(256,128)
        # neural = neural.reshape(self.W,14,15)
        # neural=np.expand_dims(neural,axis=-1)
        # image2 = np.array(self.data['image_v2'][ID])
        # image = np.array(np.concatenate((image1,image2),axis=-1)/255)
        # keep1=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        # keep2=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        
        return self.transform(np.nan_to_num(neural))
        
class poseDataset1d(torch.utils.data.Dataset):
    def __init__(self, allindex,data,TwoDropTransform):
        'Initialization'
        self.data = data
        self.allindex = allindex
        self.transform=TwoDropTransform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.allindex)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.allindex[index]

        # Load data and get label
        neural = np.array(self.data['pose'][ID]) ##image size:(256,128)

        return self.transform(np.nan_to_num(neural))
 
class pose_neural_Dataset(torch.utils.data.Dataset):
    def __init__(self, allindex,data1,data2,transform):
        'Initialization'
        self.data1 = data1
        self.data2 = data2
        
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
        image = np.array(self.data1['pose'][ID])
        neural = np.array(self.data2['neural'][ID])
        
        # keep1=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        # keep2=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        
        return self.transform(image,neural)
  
class pose2_Dataset(torch.utils.data.Dataset):
    def __init__(self, allindex,data1,data2,transform):
        'Initialization'
        self.data1 = data1
        self.data2 = data2
        
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
        image = np.array(self.data1['pose'][ID])[:,:,:14]
        neural = np.array(self.data2['pose'][ID])[:,:,14:]
        
        # keep1=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        # keep2=sorted(random.sample(deque(np.arange(self.W)),self.newW))
        
        return self.transform(image,neural)