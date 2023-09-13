import numpy as np
import torch.nn as nn
factors = [1,1,1,1,1/2,1/4,1/8,1/16, 1/32]


class WSConv2d(nn.Module): #includes equalized lr
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None #remove bias of conv layer

    #initaliaize conv weights and bias
        nn.init.normal_(self.conv.weight)  # init weights to standard guasiann
        nn.init.zeros_(self.bias) 

    def forward(self, x):
        return self.conv(x * self.scale)  + self.bias.view(1, self.bias.shape[0], 1,1)
    


class PixelNorm(nn.Module): # normalize across RGB channels
    def __init__(self):
        super().__init__() #call init of parent class nn.Module (super class)
        self.epsilon = 1e-8

        def forward ( self, x ):   # x is the input tensor to the forward function
            return x / torch.sqrt(torch.mean(x**2,  dim = 1, keepdim = True) + self.epsilon) #dim = 1 means we want to take the mean across the channels, keepdim = True means we want to keep the dimension of the mean


class ConvBlock(nn.Module): #convolutional block 3x3 conv, pixel norm, leaky relu
    pass
            


class Generator(nn.Module):
    pass


class Discriminator(nn.Module):
    pass