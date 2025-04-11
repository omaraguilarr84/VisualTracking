#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:20:33 2019
Modified to inject a MobileNetV2-style Inverted Residual Block
for a lighter, more efficient downsampling.

References:
    Original: https://github.com/ShusilDangi/DenseUNet-K
    MobileNetV2 paper for inverted residual design.
"""

### INVERTED BLOCK V1
# down block 2 is an inverted residual block

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# --- Inverted Residual Block Definition ---
class InvertedResidualBlock(nn.Module):
    """
    A simple MobileNetV2-style inverted residual block.
    It expands the number of channels, applies a depthwise convolution,
    and then projects back to a lower number of channels. If the input and output
    sizes match (and stride == 1), a skip connection is used.
    """
    def __init__(self, in_channels, out_channels, expansion_ratio=6, stride=1):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expansion_ratio
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        
        self.block = nn.Sequential(
            # Expansion: 1x1 convolution to increase channel dimension
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Depthwise convolution: separate filtering per channel, with stride for downsampling
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Projection: 1x1 convolution to reduce channel dimension
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

# --- Existing DenseNet2D Down and Up Blocks ---
class DenseNet2D_down_block(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3,3), padding=(1,1))
        self.conv21 = nn.Conv2d(input_channels+output_channels, output_channels, kernel_size=(1,1), padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3,3), padding=(1,1))
        self.conv31 = nn.Conv2d(input_channels+2*output_channels, output_channels, kernel_size=(1,1), padding=(0,0))
        self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=(3,3), padding=(1,1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)            
        
        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x):
        # If down_size is set, downsample with average pooling
        if self.down_size is not None:
            x = self.max_pool(x)
            
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
        return self.bn(out)
    
class DenseNet2D_up_block_concat(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels+input_channels, output_channels, kernel_size=(1,1), padding=(0,0))
        self.conv12 = nn.Conv2d(output_channels, output_channels, kernel_size=(3,3), padding=(1,1))
        self.conv21 = nn.Conv2d(skip_channels+input_channels+output_channels, output_channels,
                                kernel_size=(1,1), padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3,3), padding=(1,1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x):
        # Upsample the input feature map using nearest neighbor interpolation
        x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')
        # Concatenate the skip connection from the encoder with the upsampled features
        x = torch.cat((x, prev_feature_map), dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
        return out

# --- Modified DenseNet2D Model ---
class MobileNet2D_V2(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(MobileNet2D_V2, self).__init__()
        # Downsampling path with a mix of traditional and inverted blocks
        # self.down_block1 = DenseNet2D_down_block(input_channels=in_channels,
        #                                           output_channels=channel_size,
        #                                           down_size=None, dropout=dropout, prob=prob)

        # replaced block1 with Inverted Residual Block
        self.down_block1 = InvertedResidualBlock(in_channels=in_channels, 
                                                 out_channels=channel_size, expansion_ratio=6, stride=1,)
        # Instead of the usual dense block, we now use an Inverted Residual Block which does downsampling.
        # Setting stride=2 here reduces the spatial resolution like the original down block.
        self.down_block2 = InvertedResidualBlock(in_channels=channel_size,
                                                 out_channels=channel_size,
                                                 expansion_ratio=6, stride=2)
        # Continue with the original down blocks for further processing
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size,
                                                  output_channels=channel_size,
                                                  down_size=(2,2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size,
                                                  output_channels=channel_size,
                                                  down_size=(2,2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size,
                                                  output_channels=channel_size,
                                                  down_size=(2,2), dropout=dropout, prob=prob)

        # Upsampling path remains unchanged
        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size,
                                                    input_channels=channel_size,
                                                    output_channels=channel_size,
                                                    up_stride=(2,2), dropout=dropout, prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size,
                                                    input_channels=channel_size,
                                                    output_channels=channel_size,
                                                    up_stride=(2,2), dropout=dropout, prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size,
                                                    input_channels=channel_size,
                                                    output_channels=channel_size,
                                                    up_stride=(2,2), dropout=dropout, prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size,
                                                    input_channels=channel_size,
                                                    output_channels=channel_size,
                                                    up_stride=(2,2), dropout=dropout, prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)  # Now uses the inverted residual block!
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4, self.x5)
        self.x7 = self.up_block2(self.x3, self.x6)
        self.x8 = self.up_block3(self.x2, self.x7)
        self.x9 = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)
                       
        return out
