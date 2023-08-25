import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# Feature-wise Linear Modulation (FiLM) Layer
class FiLM(nn.Module):
    def __init__(self, num_features):
        super(FiLM, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        return self.gamma * x + self.beta

# Residual Block with FiLM
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.film = FiLM(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.film(x)
        x += residual
        return x

# UNet Block with FiLM + ResBlock + optional Self-Attention + optional Cross-Attention
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, attention_type=None):
        super(UNetBlock, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.blocks.append(ResBlock(in_channels, out_channels))
        if attention_type == 'self':
            self.blocks.append(SelfAttention(out_channels))
        elif attention_type == 'cross':
            self.blocks.append(CrossAttention(out_channels))

    def forward(self, x, y=None):
        for block in self.blocks:
            if isinstance(block, CrossAttention):
                x = block(x, y)
            else:
                x = block(x)
        return x

# Main Parallel-UNet Model
class ParallelUNet(nn.Module):
    def __init__(self):
        super(ParallelUNet, self).__init__()
        # Define the UNet blocks based on the given structure
        self.initial_conv = nn.Conv2d(6, 128, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            UNetBlock(128, 128, 3),
            UNetBlock(128, 64, 4),
            UNetBlock(64, 32, 6, attention_type='self'),
            UNetBlock(32, 16, 7, attention_type='cross'),
            UNetBlock(16, 16, 7, attention_type='cross'),
            UNetBlock(16, 32, 6),
            UNetBlock(32, 64, 4),
            UNetBlock(64, 128, 3)
        ])
        self.final_conv = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x

# Instantiate the model
model = ParallelUNet()






