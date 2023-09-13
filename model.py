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
    


class PixelNorm(nn.Module):
    pass


class ConvBlock(nn.Module):
    pass

class Generator(nn.Module):
    pass

class Discriminator(nn.Module):
    pass