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
    def __init__(self, in_channels, out_channels, use_pn = True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)   #conv layer 1
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pn

        def forward (self, x ):
            x = self.leaky(self.conv1(x))
            x = self.pn(x) if self.use_pn else x #if use_pn is true, then we apply pixel norm
            x = self.leaky(self.conv2(x))
            return self.pn(x) if self.use_pn else x
            


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super.__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(z_dim, in_channels),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.initial_rbg = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1) #1x1 conv layer
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList(self.inital_rbg) #list of modules, list of rgb layers
        for i in range(len(factors) -1 ): # -1 because we dont want to include the last layer
            conv_in_c = int( in_channels * factors[i]) # multiply in_channels by the factor to get the number of channels for the next block
            conv_out_c = int( in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c)) # conv block is conv layer, pixel norm, leaky relu
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1)) # we want to add a 1x1 conv layer to the rgb layers list

        def fade_in(self, alpha, upscaled, generated):
            """
            alpha: fade in factor   
            upscaled: output of the previous block
            generated: output of the current block
            """
            pass

        def forward (self,x, alpha, steps): # steps * 4 
            pass


class Discriminator(nn.Module):
    pass