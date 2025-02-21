import torch
import random 
import numpy as np
from collections import deque
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from util import TwoCropTransform
class oldpose_neural_Dataset(torch.utils.data.Dataset):
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
        
        
        return self.transform(image,neural)
        
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
        image = np.array(self.data1[ID])/255

        neural = np.array(self.data2[ID])[:,[0,3,5]]
        
        
        return self.transform(image,neural)



class pose_neural_Dataset2(torch.utils.data.Dataset):
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
        image = np.array(self.data1[ID])/255

        neural = np.array(self.data2[ID])[:,[0,3,4]]
        
        
        return self.transform(image,neural)

class pose_neural_Dataset3(torch.utils.data.Dataset):
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
        image = np.array(self.data1[ID])/255

        neural = np.array(self.data2[ID])[:,[3,5,6]]
        
        
        return self.transform(image,neural)

class pose_neural_Dataset4(torch.utils.data.Dataset):
    def __init__(self, allindex,data1,data2):
        'Initialization'
        self.data1 = data1
        self.data2 = data2
        
        self.allindex = allindex
        # self.transform=transform
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.allindex)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.allindex[index]

        # Load data and get label
        image = np.array(self.data1[ID])/255

        neural = np.array(self.data2[ID])[:]
        
        
        return [image,neural]
        
class shuffle_pose_neural_Dataset(torch.utils.data.Dataset):
    def __init__(self, allindex1,allindex2,data1,data2,transform):
        'Initialization'
        self.data1 = data1
        self.data2 = data2
        
        self.allindex1 = allindex1
        self.allindex2 = allindex2
        
        self.transform=transform
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.allindex1)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID1 = self.allindex1[index]
        ID2 = self.allindex2[index]
        

        # Load data and get label
        image = np.array(self.data1[ID1])/255

        neural = np.array(self.data2[ID2])[:,[3,5,6]]
        
        return self.transform(image,neural)
