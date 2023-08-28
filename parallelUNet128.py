import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# Pose Embedding
class PoseEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(PoseEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.clip_attention = nn.Softmax(dim=-1)  # Assuming 1D attention pooling

    def forward(self, x):
        x = self.fc(x)
        attention_weights = self.clip_attention(x)
        x = torch.sum(x * attention_weights, dim=1, keepdim=True)
        return x

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

    def forward(self, x, pose_embedding=None, garment_output=None):
        for block in self.blocks:
            if isinstance(block, SelfAttention) or isinstance(block, CrossAttention):
                x = block(x, pose_embedding, garment_output)
            else:
                x = block(x)
        return x

# Main Parallel-UNet Model
class ParallelUNet(nn.Module):
    def __init__(self, human_pose_dim, garment_pose_dim, embedding_dim=128):
        super(ParallelUNet, self).__init__()

        # Embedding layers for human and garment pose
        self.human_pose_embed = PoseEmbedding(human_pose_dim, embedding_dim)
        self.garment_pose_embed = PoseEmbedding(garment_pose_dim, embedding_dim)

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

        # garment_initial_conv의 입력 차원을 1로 설정합니다.
        self.garment_initial_conv = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),  # 입력 차원이 1입니다.
            nn.Swish(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Swish()
        )
        self.garment_blocks = nn.ModuleList([
            UNetBlock(128, 128, 3),
            UNetBlock(128, 64, 4),
            UNetBlock(64, 32, 6),
            UNetBlock(32, 16, 7),
            UNetBlock(16, 16, 7),
            UNetBlock(16, 32, 6),
        ])

    def integrate_pose(self, x, pose_embedding):
        B, C, H, W = x.size()
        pose_embedding = pose_embedding.view(B, -1, 1, 1).expand(B, -1, H, W)
        return torch.cat([x, pose_embedding], dim=1)

    def forward(self, x, human_pose, garment_pose, garment_segment):
        human_pose_embedding = self.human_pose_embed(human_pose)
        garment_pose_embedding = self.garment_pose_embed(garment_pose)

        x = self.initial_conv(x)
        garment_output = self.garment_initial_conv(garment_segment)
        for block in self.garment_blocks:
            garment_output = block(garment_output)
        for block in self.blocks:
            x = self.integrate_pose(x, human_pose_embedding)
            x = block(x, pose_embedding=human_pose_embedding, garment_output=garment_output)

        x = self.final_layer(x)
        return x


# Instantiate the model
model = ParallelUNet(human_pose_dim=1, garment_pose_dim=1)






