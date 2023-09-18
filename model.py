import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
factors = [1,1,1,1,1/2,1/4,1/8,1/16, 1/32]

class WSConv2d(nn.Module): # single convolutional layer
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None #remove bias of conv layer

        #initialize conv weights and bias
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
        super().__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0) #1x1 conv layer
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb]) #list of modules, list of rgb layers
        for i in range(len(factors) - 1 ) : # -1 because we dont want to include the last layer
            conv_in_c = int( in_channels * factors[i]) # multiply in_channels by the factor to get the number of channels for the next block
            conv_out_c = int( in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c)) # conv block is conv layer, pixel norm, leaky relu
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)) # we want to add a 1x1 conv layer to the rgb layers list

    def fade_in(self, alpha, upscaled, generated):
        """
        alpha: fade in factor   
        upscaled: output of the previous block
        generated: output of the current block
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward (self,x, alpha, steps): # steps * 4 
        out = self.initial(x) # pass x through the initial block (4x4)

        if steps == 0: 
            return self.initial_rgb(out) # if steps = 0, then we return the output of the initial rgb layer
        
        for step in range(steps):
            upscale = F.interpolate(out, scale_factor=2, mode="nearest") # upscale the output of the previous block
            out = self.prog_blocks[step](upscale) # pass the upscaled output through the conv block
        
        final_upscale = self.rgb_layers[steps - 1](upscale)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscale, final_out)
          
class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()   #list of modules, list of rgb layers
        self.leaky = nn.LeakyReLU(0.2)

        for i in range( len( factors ) - 1, 0 , 1 ): 
            conv_in_c = int( in_channels * factors[i] )# multiply in_channels by the factor to get the number of channels for the next block
            conv_out_c = int( in_channels * factors[ i - 1 ])  # multiply in_channels by the factor to get the number of channels for the next block
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pn=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))
        
        self.inital_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)  
        self.rgb_layers.append(self.inital_rgb) # add the initial rgb layer to the rgb layers list
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)   # average pooling layer

        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(
                in_channels, 1, kernel_size=1, padding=0, stride=1
            ),  # we use this instead of linear layer
        )

    def fade_in(self, alpha ,downscaled_apool, out_conv):
        return alpha * out_conv + (1 - alpha) * downscaled_apool
    
    def minibatch_std(self, x):
        batch_stats = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_stats], dim=1) # we want to add a new channel to the input tensor that contains the standard deviation of each feature map across the batch

    def forward(self, x, alpha,  steps):
        # steps * 4 but need last index
        cur_step = len( self.prog_blocks ) - steps 
        out = self.leaky(self.rgb_layers[cur_step](x)) # pass x through the rgb layer of the current step and apply leaky relu to the output 


        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        downscaled = self.leaky( self.rgb_layers[cur_step + 1] ( self.avg_pool(x) ) ) # pass x through the rgb layer of the next step by downscaling x by a factor of 2 using average pooling and apply leaky relu to the output
        out = self.prog_blocks[cur_step](out) # pass the output of the rgb layer through the conv block of the current step
        out = self.fade_in(alpha, downscaled, out)

        for step in range( cur_step + 1, len ( self.prog_blocks) ): # inital layer done, as we go up the prog factors, make smaller to 4x4
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1) # pass the output of the minibatch std layer through the final block and flatten the output


# train code
if __name__ == "__main__":
    Z_DIM = 100
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3)
    critic = Discriminator(IN_CHANNELS, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(np.log2(img_size / 4))
        print(f"Image size: {img_size}, steps: {num_steps}")
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        print(z.shape )
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")