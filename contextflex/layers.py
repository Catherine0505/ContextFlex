import torch
import torch.nn.functional as nnf
import torch.nn as nn
import torch.nn.functional as F

import einops
import math

from .model_utils import *


class SetConv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, 
                 aggregate_type, 
                 enable_activation): 
        
        super(SetConv, self).__init__()

        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.kernel_size = kernel_size 
        self.padding = (self.kernel_size - 1) // 2
        self.aggregate_type = aggregate_type
        self.enable_activation = enable_activation 

        lst = []

        if self.aggregate_type == "mean": 
            self.in_channels_conv = self.in_channels * 2
        elif self.aggregate_type == "none": 
            self.in_channels_conv = self.in_channels

        conv = nn.Conv2d(self.in_channels_conv, self.out_channels, 
            kernel_size=self.kernel_size, padding=self.padding)
        lst.append(conv)
        if self.enable_activation: 
            lst.append(nn.PReLU())
        
        self.fusion_conv = nn.Sequential(*lst)
    
    def forward_aggregate(self, x): 
        # Check dimension of x 
        check_dims(x, 5)
        gs = x.shape[1]

        set_context = None
        if self.aggregate_type == "mean": 
            set_context = torch.mean(x, dim=1, keepdim=False)
            set_context = einops.repeat(set_context, 'b c h w -> b n c h w', n=gs)
        return set_context
    
    def forward_fusion(self, x, set_context): 
        # Check dimension of x and set_context
        check_dims(x, 5)

        x_fused = x
        if set_context is not None: 
            check_dims(set_context, 5)
            x_fused = torch.cat([x_fused, set_context], dim=2)
        
        x_fused = einops.rearrange(x_fused, 'b n c h w -> (b n) c h w')
        x_fused = self.fusion_conv(x_fused)
        return x_fused
    
    def forward(self, x): 
        # Check dimension of x 
        check_dims(x, 5)
        gs = x.shape[1]

        set_context = self.forward_aggregate(x)  # b gs c_in h w
        x_fused = self.forward_fusion(x, set_context)  # (b gs) c_out h w
        x_fused = einops.rearrange(x_fused, '(b n) c h w -> b n c h w', n=gs)
        
        return x_fused


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Check dimension of x
        check_dims(x, 5)
        gs = x.shape[1]

        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.pool(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=gs)
        return x


class UpsamplingBilinear2d(nn.Module):
    def __init__(self, scale_factor):
        super(UpsamplingBilinear2d, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        # Check dimension of x
        check_dims(x, 5)
        gs = x.shape[1]

        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.upsample(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=gs)
        return x





