import torch.nn as nn
from utils import *
from Unet import *

class ParallelUNet256(nn.Module):
    def __init__(self, d_model=256):
        super(ParallelUNet256, self).__init__()
        self.garment_unet = GarmentUNet()
        self.person_unet = PersonUNet()
        self.feature_modulation = FeatureModulation(channels=d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.noise_augmentation = NoiseConditioningAugmentation(channels=3)
        self.efficient_unet = EfficientUNet()
        self.warp_and_blend_single_pass = WarpAndBlendSinglePass(d_model)
        self.clip_pooling = CLIP1DAttentionPooling(d_model=d_model)
        self.multi_head_cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=8)

    def forward(self, garment_input, person_input, pose_keypoints, time_step):
        # Apply noise conditioning augmentation
        garment_input = self.noise_augmentation(garment_input)
        person_input = self.noise_augmentation(person_input)

        garment_features = self.garment_unet(garment_input)
        person_features = self.person_unet(person_input)

        # Apply multi-head cross attention
        combined_features = self.multi_head_cross_attention(person_features, garment_features, garment_features)

        # Apply positional encoding for pose keypoints and time step
        pose_encoding = self.positional_encoding(pose_keypoints)
        time_step_encoding = self.positional_encoding(time_step)
        combined_features += pose_encoding + time_step_encoding

        # Apply feature modulation
        modulated_features = self.feature_modulation(combined_features)

        # Apply Efficient-UNet for super resolution
        output = self.efficient_unet(modulated_features)

        return output