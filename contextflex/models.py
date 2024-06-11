import torch
import torch.nn as nn

import einops

from . import layers
from .model_utils import * 


class ContextFlex(nn.Module): 
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 conv_kernel_size=3,
                 do_batchnorm=False,
                 aggregate_type="mean", 
                 enable_lastlayer_activation=False):

        super(ContextFlex, self).__init__() 

        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = (conv_kernel_size - 1) // 2

        out_channels_down = features 
        in_channels_down = [features[0]] + features[:-1]

        in_channels_bottleneck = features[-1]
        out_channels_bottleneck = features[-1]

        out_channels_up = features[::-1][1:] + [out_channels]
        in_channels_up = [out_channels_bottleneck + features[-1]] + \
            [out_channels_up[i] + out_channels_down[-2::-1][i] \
             for i in range(len(features) - 1)]

        conv = nn.Conv2d(in_channels, features[0], 
                         kernel_size=self.conv_kernel_size, padding=self.conv_padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm2d(features[0]))
        lst.append(nn.PReLU())
        self.init_conv = nn.Sequential(*lst)

        self.pool = layers.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = layers.UpsamplingBilinear2d(scale_factor=2)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for (in_channels, out_channels) in zip(in_channels_down, out_channels_down): 
            self.downs.append(layers.SetConv(in_channels, out_channels, 
                                             kernel_size=self.conv_kernel_size, 
                                             aggregate_type=aggregate_type, 
                                             enable_activation=True))
        
        self.bottleneck = layers.SetConv(in_channels_bottleneck, out_channels_bottleneck, 
                                             kernel_size=self.conv_kernel_size, 
                                             aggregate_type=aggregate_type, 
                                             enable_activation=True)
        
        for (in_channels, out_channels) in zip(in_channels_up[:-1], out_channels_up[:-1]): 
            self.ups.append(layers.SetConv(in_channels, out_channels, 
                                           kernel_size=self.conv_kernel_size, 
                                           aggregate_type=aggregate_type, 
                                           enable_activation=True))
        self.ups.append(layers.SetConv(in_channels_up[-1], out_channels_up[-1], 
                                       kernel_size=1, 
                                       aggregate_type=aggregate_type, 
                                       enable_activation=enable_lastlayer_activation))

    def forward(self, x):
        
        # Check the dimension of x
        check_dims(x, 5)
        gs = x.shape[1]

        # Initial conv
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.init_conv(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=gs)

        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]

        for (i, up) in enumerate(self.ups): 
            x = self.upsample(x)
            x = torch.cat((skip_connections[i], x), dim=2)
            x = up(x)
        
        return x

