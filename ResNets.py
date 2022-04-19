#!/usr/bin/env python
# coding: utf-8

# In[12]:


"""
Special ResNet-18/34 architecture for 32x32 data in order to use on CIFAR-10
"""
import torch
import torch.nn as nn


# In[18]:


class conv_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        """
        Basic conv followed by bn
        
        params:
        
        in_channels: how many dimensions are in the input
        out_channels: how many dimensions we want on the output
        stride: assumed to be 1 but can be used for downsampling
        """
        
        super().__init__()
        
        #pytorch does not support same padding for strides more than 1
        if stride == 1:
            self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 'same', stride = stride, bias = False),
                     nn.BatchNorm2d(out_channels))
        else:
            self.padding =  (3 // 2, 3 // 2)
            self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = self.padding, stride = stride, bias = False),
                     nn.BatchNorm2d(out_channels))
        
        
    def forward(self, x):
        return self.block(x)
    

class ResNet_Projection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 2):
        """
        shortcut function if the original is downsampled by 2
        
        params:
        in_channels: how many dimensions are in the input
        out_channels: how many dimensions we want on the output
        kernel: kernel size which is usually 1 in resnets
        stride: assumed to be 1 but can be used for downsampling
        """
        super().__init__()
        
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 2, bias = False)
        self.norm = nn.BatchNorm2d(out_channels)
        
        
    def forward(self, x):
        x = self.layer(x)
        return self.norm(x)
    
class Residual_block(nn.Module):
    
    def __init__(self, in_channels, out_channels, first_stride = 1, use_projection = True):
        """
        residual block function
        
        params:
        in_channels: how many dimensions are in the input
        out_channels: how many dimensions we want on the output
        stride: assumed to be 1 but can be used for downsampling
        """
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_projection = use_projection
        self.first_stride = first_stride
        
        self.conv_block1 = conv_bn_block(in_channels, out_channels, stride = first_stride)
        self.conv_block2 = conv_bn_block(out_channels, out_channels, stride = 1)
        if self.first_stride > 1 and use_projection == True:
            self.proj = ResNet_Projection(in_channels, out_channels)
        
    def forward(self, x):
        
        residual = x
        if self.first_stride > 1 and self.use_projection:
            residual = self.proj(x)
            
        if self.first_stride > 1 and not self.use_projection:
            residual = 0
            
        x = self.conv_block1(x)
        x = nn.ReLU()(x)
        x = self.conv_block2(x)
        x += residual
        x = nn.ReLU()(x)
        return x
    
    
class ResNet_layer(nn.Module):
    
    def __init__(self, in_channels, out_channels, blocks, use_projection = True):
        """
        ResNet basic layer composed of x blocks
        
        params:
        in_channels: how many dimensions are in the input
        out_channels: how many dimensions we want on the output
        blocks: how many blocks will be used
        """
        super().__init__()
        self.layers = nn.Sequential(Residual_block(in_channels, out_channels, first_stride = 2, use_projection = use_projection),
                                   *[Residual_block(out_channels, out_channels) for _ in range(blocks - 1)])
        
    def forward(self, x):
        #print(x.shape)
        return self.layers(x)
    
    
    
class ResNet(nn.Module):
    
    def __init__(self, layers, blocks_per_layer, channels_per_layer, use_final_pool = True, cifar_data = True):
        """
        class to create a resnet
        
        params:
        layers: how many layers of ResNet layer we will have in our model
        blocks_per_layer: how many blocks we want in each layer
        channels_per_layer: list of how many channels will be in each layer
        """
        super().__init__()
        #no size per map will be 16x16
        self.initial_downsampling = nn.Conv2d(3, 64, kernel_size = 7, stride = 1, padding = 'same', bias = False)
        self.norm = nn.BatchNorm2d(64)
        
        if cifar_data:
            self.pool = nn.Identity()
        else:
            self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 3 // 2)
        
        #adds the first projection that we do so as to use in the first block
        channels_per_layer = [64] + channels_per_layer
        
        self.layers_to_representation = nn.Sequential(*[ResNet_layer(channels_per_layer[i], channels_per_layer[i + 1],
                                                                    blocks_per_layer[i], use_projection = False if i == 0 else True) for i in range(layers)])
        self.use_final_pool = use_final_pool
        if self.use_final_pool:
            self.final_pool = nn.AvgPool2d(3, stride = 2, padding = 3 //2)
        
    def forward(self, x):
        x = self.initial_downsampling(x)
        x = self.norm(x)
        x = nn.ReLU()(x)
        x = self.layers_to_representation(x)
        if self.use_final_pool:
            x = self.final_pool(x)
        return x
        
        


# In[19]:


#will have to introduce the bottleneck layer if hoping to go higher
import torchvision

def ResNet18(cifar = False):
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    if cifar:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
    return model


def ResNet34(cifar_data = False):
    return ResNet(4, [3, 4, 6, 3], [64, 128, 256, 512], cifar_data = cifar_data)

