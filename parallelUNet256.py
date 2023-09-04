import torch
import torch.nn as nn
import json


class PoseEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PoseEmbedding, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class CLIPAttentionPooling(nn.Module):
    def __init__(self, dim):
        super(CLIPAttentionPooling, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        attention_weights = torch.softmax(q @ k.T, dim=-1)
        out = attention_weights @ x
        return out

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, pose_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(in_dim, pose_dim)
        self.key = nn.Linear(pose_dim, pose_dim)
        self.value = nn.Linear(pose_dim, in_dim)

    def forward(self, x, pose_embedding):
        q = self.query(x)
        k = self.key(pose_embedding)
        v = self.value(pose_embedding)

        attention_weights = torch.softmax(q @ k.T, dim=-1)
        out = attention_weights @ v
        return out

# Cross-Attention Layer
class CrossAttentionLayer(nn.Module):
    def __init__(self, d, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.d = d // num_heads
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, zt, ic):
        # Multi-head attention preparation
        Q = self.query(zt).view(zt.size(0), -1, self.num_heads, self.d).permute(0, 2, 1, 3)
        K = self.key(ic).view(ic.size(0), -1, self.num_heads, self.d).permute(0, 2, 1, 3)
        V = self.value(ic).view(ic.size(0), -1, self.num_heads, self.d).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attention_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.d ** 0.5)
        attention_weights = self.softmax(attention_map)
        out = torch.matmul(attention_weights, V).permute(0, 2, 1, 3).contiguous().view(zt.size(0), -1,
                                                                                       self.d * self.num_heads)

        return out

# Read JSON file and create initial pose embeddings
with open('human_pose.json', 'r') as f:
    human_data = json.load(f)
human_pose = torch.Tensor(human_data['landmarks'])

with open('garment_pose.json', 'r') as f:
    garment_data = json.load(f)
garment_pose = torch.Tensor(garment_data['landmarks'])

# Convert raw pose data to embeddings
human_pose_embedding_layer = PoseEmbedding(human_pose.shape[0], 512)
garment_pose_embedding_layer = PoseEmbedding(garment_pose.shape[0], 512)

human_pose_embedding = human_pose_embedding_layer(human_pose)
garment_pose_embedding = garment_pose_embedding_layer(garment_pose)

class FiLM(nn.Module):
    def __init__(self, num_features):
        super(FiLM, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        return self.gamma * x + self.beta

class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, scale='down', num_groups=32):
        super(ResBlock, self).__init__()
        use_bias = True  # You can set this based on your needs
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 'up' 'down' 'same'"

        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=use_bias)
        if scale == 'up':
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=use_bias)
            )
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)

        self.film = FiLM(out_nc)

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, out_nc),  # GroupNorm
            nn.SiLU(inplace=True),  # Swish (SiLU)
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.GroupNorm(num_groups, out_nc),  # GroupNorm
            nn.SiLU(inplace=True),  # Swish (SiLU)
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, film_params=None):
        residual = self.scale(x)
        out = self.block(residual)
        if film_params is not None:
            gamma, beta = torch.chunk(film_params, 2, dim=1)
            out = gamma * out + beta
        return self.relu(residual + out)


class PersonUnet(nn.Module):
    def __init__(self, in_channels, out_channels, pose_dim, norm_layer=nn.BatchNorm2d):
        super(PersonUnet, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.enc1 = nn.ModuleList([ResBlock(128, 128, scale='same') for _ in range(3)])

        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.ModuleList([ResBlock(256, 256, scale='same') for _ in range(4)])

        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.ModuleList([ResBlock(512, 512, scale='same') for _ in range(6)])

        self.enc_conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.enc4 = nn.ModuleList([ResBlock(1024, 1024, scale='same') for _ in range(7)])

        # Cross Attention Layers
        self.cross_attn1 = CrossAttentionLayer(512)
        self.cross_attn2 = CrossAttentionLayer(1024)
        self.cross_attn3 = CrossAttentionLayer(1024)
        self.cross_attn4 = CrossAttentionLayer(512)

        self.attn1 = AttentionLayer(128, pose_dim)
        self.attn2 = AttentionLayer(256, pose_dim)
        self.attn3 = AttentionLayer(512, pose_dim)
        self.attn4 = AttentionLayer(1024, pose_dim)

        # Decoder
        self.dec_conv4 = nn.ConvTranspose2d(3072, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.ModuleList([ResBlock(1024, 1024, scale='same') for _ in range(7)])

        self.dec_conv3 = nn.ConvTranspose2d(1536, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ModuleList([ResBlock(512, 512, scale='same') for _ in range(6)])

        self.dec_conv2 = nn.ConvTranspose2d(768, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ModuleList([ResBlock(256, 256, scale='same') for _ in range(4)])

        self.dec_conv1 = nn.ConvTranspose2d(384, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.ModuleList([ResBlock(128, 128, scale='same') for _ in range(3)])

        # Final conv layer
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, Ia, zt, person_pose_embedding, garment_pose_embedding,
                enc3_garment, enc4_garment, dec4_garment, dec3_garment):
        # Combine person and garment pose embeddings
        x = torch.cat([Ia, zt], dim=1)
        pose_embedding = person_pose_embedding + garment_pose_embedding

        # Encoder with Attention
        x = self.enc_conv1(x)
        for block in self.enc1:
            x = block(x)
        film1_params = self.attn1(x, pose_embedding)

        x = self.enc_conv2(x)
        for block in self.enc2:
            x = block(x)
        film2_params = self.attn2(x, pose_embedding)

        x = self.enc_conv3(x)
        for block in self.enc3:
            x = block(x)
        film3_params = self.attn3(x, pose_embedding)
        x = self.cross_attn1(x, enc3_garment)  # Cross-Attention

        x = self.enc_conv4(x)
        for block in self.enc4:
            x = block(x)
        film4_params = self.attn4(x, pose_embedding)
        x = self.cross_attn2(x, enc4_garment)  # Cross-Attention

        # Decoder with skip connections and FiLM modulation
        x = self.dec_conv4(torch.cat([x, film4_params], dim=1))
        for block in self.dec4:
            x = block(x, film4_params)
        x = self.cross_attn3(x, dec4_garment)  # Cross-Attention

        x = self.dec_conv3(torch.cat([x, film3_params], dim=1))
        for block in self.dec3:
            x = block(x, film3_params)
        x = self.cross_attn4(x, dec3_garment) # Cross-Attention

        x = self.dec_conv2(torch.cat([x, film2_params], dim=1))
        for block in self.dec2:
            x = block(x, film2_params)

        x = self.dec_conv1(torch.cat([x, film1_params], dim=1))
        for block in self.dec1:
            x = block(x, film1_params)

        # Final Layer
        final = self.final_conv(x)

        return final

class GarmentUnet(nn.Module):
    def __init__(self, in_channels, out_channels, pose_dim, norm_layer=nn.BatchNorm2d):
        super(GarmentUnet, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.enc1 = nn.ModuleList([ResBlock(128, 128, scale='same') for _ in range(3)])

        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.ModuleList([ResBlock(256, 256, scale='same') for _ in range(4)])

        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.ModuleList([ResBlock(512, 512, scale='same') for _ in range(6)])

        self.enc_conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.enc4 = nn.ModuleList([ResBlock(1024, 1024, scale='same') for _ in range(7)])

        self.attn1 = AttentionLayer(128, pose_dim)
        self.attn2 = AttentionLayer(256, pose_dim)
        self.attn3 = AttentionLayer(512, pose_dim)
        self.attn4 = AttentionLayer(1024, pose_dim)

        # Decoder
        self.dec_conv4 = nn.ConvTranspose2d(3072, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.ModuleList([ResBlock(1024, 1024, scale='same') for _ in range(7)])

        self.dec_conv3 = nn.ConvTranspose2d(1536, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ModuleList([ResBlock(512, 512, scale='same') for _ in range(6)])

        self.dec_conv2 = nn.ConvTranspose2d(768, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ModuleList([ResBlock(256, 256, scale='same') for _ in range(4)])

        self.dec_conv1 = nn.ConvTranspose2d(384, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.ModuleList([ResBlock(128, 128, scale='same') for _ in range(3)])

        # Final conv layer
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, Ic, person_pose_embedding, garment_pose_embedding):
        # Combine person and garment pose embeddings
        x = Ic
        pose_embedding = person_pose_embedding + garment_pose_embedding

        # Encoder with Attention
        x = self.enc_conv1(x)
        for block in self.enc1:
            x = block(x)
        film1_params = self.attn1(x, pose_embedding)

        x = self.enc_conv2(x)
        for block in self.enc2:
            x = block(x)
        film2_params = self.attn2(x, pose_embedding)

        x = self.enc_conv3(x)
        for block in self.enc3:
            x = block(x)
        enc3_out = x
        film3_params = self.attn3(x, pose_embedding)

        x = self.enc_conv4(x)
        for block in self.enc4:
            x = block(x)
        enc4_out = x
        film4_params = self.attn4(x, pose_embedding)

        # Decoder with skip connections and FiLM modulation
        x = self.dec_conv4(torch.cat([x, film4_params], dim=1))
        for block in self.dec4:
            x = block(x, film4_params)
        dec4_out = x

        x = self.dec_conv3(torch.cat([x, film3_params], dim=1))
        for block in self.dec3:
            x = block(x, film3_params)
        dec3_out = x

        x = self.dec_conv2(torch.cat([x, film2_params], dim=1))
        for block in self.dec2:
            x = block(x, film2_params)

        x = self.dec_conv1(torch.cat([x, film1_params], dim=1))
        for block in self.dec1:
            x = block(x, film1_params)

        # Final Layer
        final = self.final_conv(x)

        return final, enc3_out, enc4_out, dec4_out, dec3_out


class UnifiedUNet(nn.Module):
    def __init__(self, in_channels_person, in_channels_garment, out_channels, pose_dim, norm_layer=nn.BatchNorm2d):
        super(UnifiedUNet, self).__init__()

        self.person_unet = PersonUnet(in_channels_person, out_channels, pose_dim, norm_layer)
        self.garment_unet = GarmentUnet(in_channels_garment, out_channels, pose_dim, norm_layer)

        # CLIP-style 1D attention pooling
        self.clip_pooling = CLIPAttentionPooling(pose_dim)

    def forward(self, x_person, x_garment, person_pose_embedding, garment_pose_embedding, diffusion_timestep,
                noise_level):
        # Combine person and garment pose embeddings
        pose_embedding = person_pose_embedding + garment_pose_embedding

        # CLIP-style 1D attention pooling
        pooled_embedding = self.clip_pooling(pose_embedding)

        # Add positional encoding of diffusion timestep and noise augmentation levels
        positional_encoding = diffusion_timestep + noise_level
        modulated_embedding = pooled_embedding + positional_encoding

        # Forward pass through Garment U-Net to get intermediate outputs
        out_garment, enc3_garment, enc4_garment, dec4_garment, dec3_garment = self.garment_unet(
            x_garment, modulated_embedding, modulated_embedding)

        # Forward pass through Person U-Net with shared cross-attention layers
        out_person = self.person_unet(x_person, modulated_embedding, modulated_embedding,
                                      enc3_garment, enc4_garment, dec4_garment, dec3_garment)

        # Combine 'out_person' and 'out_garment' using your blending logic here
        combined_out = out_person + out_garment

        return combined_out