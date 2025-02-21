import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#https://github.com/JiahongChen/ResNet-decoder
#https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        
        out = torch.flatten(out, 1)
        return out


def resnet18(in_channel,**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],in_channel=in_channel, **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def neural_resnet18(in_channel,**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channel=in_channel, **kwargs)


def neural_resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channel = 210,**kwargs)


def neural_resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channel =210, **kwargs)


def neural_resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], in_channel = 210,**kwargs)

def pose_resnet18(in_channel,**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channel=in_channel ,**kwargs)


def pose_resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channel = 7,**kwargs)


def pose_resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channel = 7, **kwargs)


def pose_resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], in_channel = 7,**kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'neural_resnet18': [neural_resnet18, 512],
    'neural_resnet34': [neural_resnet34, 512],
    'neural_resnet50': [neural_resnet50, 2048],
    'neural_resnet101': [neural_resnet101, 2048],
    'pose_resnet18': [pose_resnet18, 512],
    'pose_resnet34': [pose_resnet34, 512],
    'pose_resnet50': [pose_resnet50, 2048],
    'pose_resnet101': [pose_resnet101, 2048],
}

#model_dict = {
#    'neural_resnet18': [resnet18, 512],
#    'neural_resnet34': [resnet34, 512],
#    'neural_resnet50': [resnet50, 2048],
#    'neural_resnet101': [resnet101, 2048],
#}

class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

#we need to initialize cross_domain model
class SupConClipResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, flag,in_channel=3, name1='resnet50', name2='neural_resnet18', name3='pose_resnet18',head='mlp', feat_dim=512):
        super(SupConClipResNet, self).__init__()

        
        model_fun1, dim_in1 = model_dict[name1]
        model_fun2, dim_in2 = model_dict[name2]
        model_fun3, dim_in3 = model_dict[name3]
        
        self.imgencoder = model_fun1(in_channel=in_channel)
        self.poseencoder = model_fun1(in_channel=in_channel)
        self.pose2dencoder = model_fun3(in_channel=in_channel)
        
        self.neural2dencoder = model_fun2(in_channel=in_channel)
        
        self.toyencoder = model_fun1(in_channel=in_channel)
        self.neuralencoder = model_fun2(in_channel=in_channel)
        
        self.flag=flag
        if head == 'linear':
            self.head_img = nn.Linear(dim_in1, feat_dim)
            self.head_flow = nn.Linear(dim_in1, feat_dim)
            self.head_neural = nn.Linear(dim_in2, feat_dim)
            #self.head = nn.Linear(dim_in1, feat_dim)
            #self.head2 = nn.Linear(dim_in2, feat_dim)
        elif head == 'mlp':
            self.head_img = nn.Sequential(
                nn.Linear(dim_in1, dim_in1),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in1, feat_dim)
            )
            self.head_pose = nn.Sequential(
                nn.Linear(dim_in3, dim_in3),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in1, feat_dim)
            )
            self.head_toy = nn.Sequential(
                nn.Linear(dim_in1, dim_in1),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in1, feat_dim)
            )

            self.head_neural = nn.Sequential(
                nn.Linear(dim_in2, dim_in2),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in2, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, input_):
        # print(self.flag)
        if self.flag=='image':
            # neural=self.neuralencoder(neural)
            image=self.imgencoder(input_)
            # flow=self.flowencoder(flow)
            out=F.normalize(self.head_img(image), dim=1)
            
        elif self.flag=='neural':
            neural=self.neuralencoder(input_)
            # flow=self.flowencoder(flow)
            out=F.normalize(self.head_neural(neural), dim=1)

        elif self.flag=='pose':
            pose=self.poseencoder(input_)
            # flow=self.flowencoder(flow)
            out=F.normalize(self.head_pose(pose), dim=1)
        
        elif self.flag=='toy':
            toy=self.toyencoder(input_)
            # flow=self.flowencoder(flow)
            out=F.normalize(self.head_toy(toy), dim=1)
        elif self.flag=='2dpose':
            pose=self.pose2dencoder(input_)
            # flow=self.flowencoder(flow)
            out=F.normalize(self.head_pose(pose), dim=1)
        elif self.flag=='2dneural':
            n=self.neural2dencoder(input_)
            # flow=self.flowencoder(flow)
            out=F.normalize(self.head_neural(n), dim=1)
        elif self.flag=='2dmcneural':
            n=self.neural2dencoder(input_)
            # flow=self.flowencoder(flow)
            out=F.normalize(self.head_neural(n), dim=1)
        else:
            raise NotImplementedError(
                'Flag not supported: {}'.format(self.flag))
            
        return  out
    
    #self.head_neural(neural),self.head_img(image),self.head_neural(flow)
class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNNModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)//2

        #Initializing hidden state for first input using method defined below
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        print(hidden.shape,x.shape)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        print(out.shape)
        out = self.fc(out[:,-1,:])
        
        return out
class decoder(nn.Module):
    def __init__(self,newW):
        super(decoder, self).__init__()
        self.fc = nn.Linear(512, 128*2*2)

            # Upsample to the target size
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)  # Output: (batch_size, 64, 4, 4)
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)   # Output: (batch_size, 32, 8, 8)
        self.conv_transpose3 = nn.ConvTranspose2d(32, newW, kernel_size=2, stride=2, padding=1,output_padding=1)   # Output: (batch_size, 22, 16, 16)
        
        # Batch normalization (optional)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        self.newW=newW
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 2, 2)  # Reshape to (batch_size, 128, 2, 2)
        
        # Apply convolutional transpose layers
        x = F.relu(self.bn1(self.conv_transpose1(x)))
        x = F.relu(self.bn2(self.conv_transpose2(x)))
        x = torch.sigmoid(self.conv_transpose3(x))
        x = self.upsample(x, output_size=(self.newW,14,15))
        return x
    
class decoder_image(nn.Module):
    def __init__(self,newW):
        super(decoder_image, self).__init__()
        self.fc = nn.Linear(512, 128*2*2)

            # Upsample to the target size
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)  # Output: (batch_size, 64, 4, 4)
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)   # Output: (batch_size, 32, 8, 8)
        self.conv_transpose3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=1)   # Output: (batch_size, 32, 8, 8)
        
        self.conv_transpose4 = nn.ConvTranspose2d(32, newW, kernel_size=(2,3), stride=2, padding=1)   # Output: (batch_size, 22, 14,15)
        
        # Batch normalization (optional)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.newW=newW
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 2, 2)  # Reshape to (batch_size, 128, 2, 2)
        
        # Apply convolutional transpose layers
        x = F.relu((self.conv_transpose1(x)))
        x = F.relu((self.conv_transpose2(x)))
        x = F.relu((self.conv_transpose3(x)))
        
        x = self.conv_transpose4(x)
        # x = self.upsample(x, output_size=(self.newW,14,15))
        return x




class decoder_neural(nn.Module):
    def __init__(self,newW):
        super(decoder_neural, self).__init__()
        self.fc = nn.Linear(512, 128*4*4)

            # Upsample to the target size
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=(2,3), stride=2, padding=1)  # Output: (batch_size, 64, 4, 4)
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=1)   # Output: (batch_size, 32, 8, 8)
        self.conv_transpose3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=3, padding=1,output_padding=(1,0))   # Output: (batch_size, 32, 8, 8)
        self.conv_transpose4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=1, output_padding=1)   # Output: (batch_size, 32, 8, 8)
        self.conv_transpose5 = nn.ConvTranspose2d(32, newW, kernel_size=2, stride=2, padding=1)   # Output: (batch_size, 22, 14,15)
        
        # Batch normalization (optional)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        
        
        self.newW=newW
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4,4)  # Reshape to (batch_size, 128, 2, 2)
        
        # Apply convolutional transpose layers
        # x = F.relu(self.bn1(self.conv_transpose1(x)))
        # x = F.relu(self.bn2(self.conv_transpose2(x)))
        # x = F.relu(self.bn3(self.conv_transpose3(x)))
        # x = F.relu(self.bn4(self.conv_transpose4(x)))
        x = F.relu((self.conv_transpose1(x)))
        x = F.relu((self.conv_transpose2(x)))
        x = F.relu((self.conv_transpose3(x)))
        x = F.relu((self.conv_transpose4(x)))
        
        x = self.conv_transpose5(x)
        # x = self.upsample(x, output_size=(self.newW,14,15))
        return x

    
class linear_decoder_image(nn.Module):
    def __init__(self,newW):
        super(linear_decoder_image, self).__init__()
        self.newW=newW
        
        self.fc1 = nn.Linear(512, 128*2*2)
        self.fc2 = nn.Linear(128*2*2, self.newW*14*15)
        

        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        
        x = x.view(-1, self.newW,14,15)  # Reshape to (batch_size, 128, 2, 2)
        
        return x
    
class linear_decoder_neural(nn.Module):
    def __init__(self,newW):
        super(linear_decoder_neural, self).__init__()
        self.newW=newW
        
        self.fc1 = nn.Linear(512, 128*2*2)
        self.fc2 = nn.Linear(128*2*2, self.newW*108*128)
                
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        
        x = x.view(-1, self.newW,108,128)  # Reshape to (batch_size, 128, 2, 2)
        
        return x
class Decoder(nn.Module):
    def __init__(self, embedding_dim, channels):
        super(Decoder, self).__init__()
        # define fully connected layer to unflatten the embeddings
        self.fc = nn.Linear(embedding_dim, np.prod((128,7,7)))
        # store the shape before flattening
        self.reshape_dim = (128,7,7)
        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # self.deconv3 = nn.ConvTranspose2d(
        #     64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        # )
        # define final convolutional layer to generate output image
        self.conv1 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # apply fully connected layer to unflatten the embeddings
        x = self.fc(x)
        # reshape the tensor to match shape before flattening
        x = x.view(x.size(0), *self.reshape_dim)
        # apply ReLU activations after each transpose convolutional layer
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        # x = F.relu(self.deconv3(x))
        # apply sigmoid activation to the final convolutional layer to generate output image
        x = torch.sigmoid(self.conv1(x))
        return x
class Decoder_neural(nn.Module):
    def __init__(self, embedding_dim, channels):
        super(Decoder_neural, self).__init__()
        # define fully connected layer to unflatten the embeddings
        self.fc = nn.Linear(embedding_dim, np.prod((128,4,4)))
        # store the shape before flattening
        self.reshape_dim = (128,4,4)
        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # define final convolutional layer to generate output image
        self.conv1 = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # apply fully connected layer to unflatten the embeddings
        x = self.fc(x)
        # reshape the tensor to match shape before flattening
        x = x.view(x.size(0), *self.reshape_dim)
        # apply ReLU activations after each transpose convolutional layer
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        
        # apply sigmoid activation to the final convolutional layer to generate output image
        x = self.conv1(x)
        return x
