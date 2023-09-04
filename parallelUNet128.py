import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class PoseEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(PoseEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.clip_attention = nn.Softmax(dim=-1)  # Assuming 1D attention pooling

    def forward(self, x, pose_embedding=None):  # Modified to take pose_embedding
        if pose_embedding is None:
            return x  # or any other default behavior when pose_embedding is None
        gamma = self.gamma * pose_embedding  # Modulate gamma using pose_embedding
        beta = self.beta * pose_embedding  # Modulate beta using pose_embedding
        return gamma * x + beta  # Return modulated x

class FiLM(nn.Module):
    def __init__(self, num_features):
        super(FiLM, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x, pose_embedding=None):  # Modified to take pose_embedding
        if pose_embedding is None:
            return x  # or any other default behavior when pose_embedding is None
        gamma = self.gamma * pose_embedding  # Modulate gamma using pose_embedding
        beta = self.beta * pose_embedding  # Modulate beta using pose_embedding
        return gamma * x + beta  # Return modulated x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.film = FiLM(out_channels)
        self.activation = nn.ReLU()

        # Add a projection layer if in_channels != out_channels
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.projection = None

    def forward(self, x, pose_embedding=None):
        if self.projection:
            residual = self.projection(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.film(x, pose_embedding)
        x += residual
        return x


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, attention_types=[]):
        super(UNetBlock, self).__init__()
        self.blocks = nn.ModuleList()

        for _ in range(num_res_blocks):
            self.blocks.append(ResBlock(in_channels, out_channels))

        # 'attention_types'는 리스트 형태로 전달되어, 여러 attention 타입을 포함할 수 있습니다.
        for attention_type in attention_types:  # Modified
            if attention_type == 'self':
                self.blocks.append(SelfAttention(out_channels))
            elif attention_type == 'cross':
                self.blocks.append(CrossAttention(out_channels))

    def forward(self, x, garment_features=None, pose_embedding=None):
        for block in self.blocks:
            if isinstance(block, CrossAttention):
                x = block(x, garment_features)  # Passing garment_features as key-value pairs
            elif isinstance(block, SelfAttention):
                x = block(x, pose_embedding)
            else:
                x = block(x)
        return x

# Main Parallel-UNet Model
class ParallelUNet(nn.Module):
    def __init__(self, human_pose_dim, garment_pose_dim,num_channels_Ia,num_channels_zt, embedding_dim=128):
        super(ParallelUNet, self).__init__()

        # Embedding layers for human and garment pose
        self.human_pose_embed = PoseEmbedding(human_pose_dim, embedding_dim)
        self.garment_pose_embed = PoseEmbedding(garment_pose_dim, embedding_dim)

        # Define the UNet blocks based on the given structure
        self.initial_conv = nn.Conv2d(num_channels_Ia + num_channels_zt, 128, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            UNetBlock(128, 128, 3),
            UNetBlock(128, 64, 4),
            UNetBlock(64, 32, 6, attention_types=['self','cross']),
            UNetBlock(32, 16, 7, attention_types=['self','cross']),
            UNetBlock(16, 16, 7, attention_types=['self','cross']),
            UNetBlock(16, 32, 6),
            UNetBlock(32, 64, 4),
            UNetBlock(64, 128, 3)
        ])
        self.final_conv = nn.Conv2d(128, 3, kernel_size=3, padding=1)

        # garment_initial_conv의 입력 차원을 1로 설정합니다.
        self.garment_initial_conv = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),  # Change the input channels to 3
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU()
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

    def forward(self, Ia, zt, human_pose, garment_pose, garment_segment):
        human_pose_embedding = self.human_pose_embed(human_pose)
        garment_pose_embedding = self.garment_pose_embed(garment_pose)

        # Concatenating Ia and zt
        person_input = torch.cat([Ia, zt], dim=1)
        garment_features_list = []  # New list to store features for cross-attention
        garment_output = self.garment_initial_conv(garment_segment)

        for block in self.garment_blocks:  # Early stop at 32x32
            garment_output = block(garment_output)
            garment_features_list.append(garment_output)  # Store features

        for idx, block in enumerate(self.blocks):
            person_input = self.integrate_pose(person_input, human_pose_embedding)
            person_input = self.integrate_pose(person_input, garment_pose_embedding)
            person_output = block(person_input, pose_embedding=human_pose_embedding,
                                  garment_pose_embedding=garment_pose_embedding)
            person_input = person_output  # Update person_input for the next iteration
            # Now we start passing garment_features to the person UNet blocks
            for idx, block in enumerate(self.blocks):
                if idx >= 2 and idx <= 5:  # Indices where you want cross attention to occur
                    person_output = block(person_output, garment_features=garment_features_list[
                        idx - 2])  # garment_features_list comes from the forward pass of garment blocks
                else:
                    person_output = block(person_output)

        person_output = self.final_conv(person_output)  # Modified
        return person_output

# Instantiate the model
model = ParallelUNet(human_pose_dim=136, garment_pose_dim=16, num_channels_Ia=3, num_channels_zt=3)







