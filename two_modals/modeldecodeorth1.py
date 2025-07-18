import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import ortho_group

class Decoder(nn.Module):
    def __init__(self, embedding_dim, channels,bottleneck_dim,image_latent_dim,neural_latent_dim,neural_dim,image_private_dim,neural_private_dim,pose_dim):
        super(Decoder, self).__init__()
        m1 = ortho_group.rvs(dim=image_private_dim+image_latent_dim).astype('float32')
        
        # define fully connected layer to unflatten the embeddings
        self.lamdba1=nn.Linear(embedding_dim,bottleneck_dim)
        self.lamdba2=nn.Linear(embedding_dim,bottleneck_dim)
        self.fusion=nn.Linear(bottleneck_dim*2,image_latent_dim)
        

        self.imagedense=nn.Linear(bottleneck_dim,image_latent_dim)
        self.neuraldense=nn.Linear(bottleneck_dim,neural_latent_dim)
        
        self.neural_output=nn.Linear(7*7*4,neural_dim)
        self.imagepreviate=nn.Linear(bottleneck_dim,image_private_dim)
        self.neuralpreviate=nn.Linear(bottleneck_dim,neural_private_dim)
        self.pose_output=nn.Linear(7*7*4,pose_dim)
        self.neural_output=nn.Linear(7*7*4,neural_dim)
        
        # self.orthlayer1=nn.Linear(image_private_dim+image_latent_dim,image_private_dim)
        # self.orthlayer2=nn.Linear(image_private_dim+image_latent_dim,image_latent_dim)
        
        # with torch.no_grad():
        #     self.orthlayer2.weight = nn.Parameter(
        #         torch.from_numpy(m1[image_private_dim:image_private_dim+image_latent_dim,:]), requires_grad=False)
        #     self.orthlayer1.weight = nn.Parameter(
        #         torch.from_numpy(m1[:image_private_dim,:]), requires_grad=False)

        ###########################image############################
        self.fc1 = nn.Linear(image_private_dim+image_latent_dim, np.prod((128,7,7)))
        # store the shape before flattening
        self.reshape_dim1 = (128,7*7)
        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose1d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose1d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        self.conv1 = nn.Conv1d(64, channels, kernel_size=3, stride=1, padding=1)
        #############################neural###########################
        self.fc2 = nn.Linear(neural_private_dim+neural_latent_dim, np.prod((128,7,7)))
        self.reshape_dim2 = (128,7*7)
        
        self.deconv3 = nn.ConvTranspose1d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv4 = nn.ConvTranspose1d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        self.conv2 = nn.Conv1d(64, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x,y):
        #######################both##########################
        x1=self.lamdba1(x)
        y1=self.lamdba2(y)
        image_previate=self.imagepreviate(x1)
        neural_previate=self.neuralpreviate(y1)
        
        two=torch.cat((x1,y1),axis=-1)
        global_latent=self.fusion(two)

        # image_latent=self.imagedense(x1)
        
        # neural_latent=self.neuraldense(y1)
        
        temp1=torch.cat((image_previate,global_latent),axis=-1)
        temp2=torch.cat((neural_previate,global_latent),axis=-1)

        # temp11=self.orthlayer1(temp1)###pri
        # temp12=self.orthlayer2(temp1)###global
        
        # temp21=self.orthlayer1(temp2)
        # temp22=self.orthlayer2(temp2)
        
        # temp1=torch.cat((temp11,temp12),axis=-1)
        # temp2=torch.cat((temp21,temp22),axis=-1)
        #######################image###########################
        x= self.fc1(temp1)
        x = x.view(x.size(0), *self.reshape_dim1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.conv1(x)
        
        image_pred = self.pose_output(x)
        ######################neural##########################
        y= self.fc2(temp2)
        y = y.view(x.size(0), *self.reshape_dim2)
        y = F.relu(self.deconv3(y))
        y = F.relu(self.deconv4(y))
        neural_pred = self.conv2(y)
        neural_pred=self.neural_output(neural_pred)

        
        return global_latent,global_latent,global_latent,image_pred,neural_pred,image_previate,neural_previate
