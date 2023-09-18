import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()

        # First GroupNorm -> Swish -> Convolution sequence
        self.groupnorm1 = nn.GroupNorm(num_groups=8, num_channels=channels)  # groupnorm의 32값은 임의의 값이자 하이퍼파라미터 값 이므로, 1 or num_channels or 다른 수 총 3가지 옵션으로 조정 가능
        self.swish1 = Swish()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Second GroupNorm -> Swish -> Convolution sequence
        self.groupnorm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.swish2 = Swish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Skip connection
        self.skip = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.swish1(self.groupnorm1(x)))
        out = self.conv2(self.swish2(self.groupnorm2(out)))
        out += self.skip(x)
        return out
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ImageToLinear(nn.Module):
    def __init__(self):
        super(ImageToLinear, self).__init__()
        self.efficient_unet = ParallelUNet()  # image data(2D)를 vector data(1D)로 받을 때 기존에 사용했던 UNet의 종류를 입력해야 함
        self.flatten = Flatten()
        self.fc = nn.Linear(256*256*3, 1000)

    def forward(self, x):
        x = self.efficient_unet(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, channels, attention_heads=8):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.attention_heads = attention_heads
        self.hidden_size = 2 * channels

        self.query = nn.Linear(channels, self.hidden_size * self.attention_heads)
        self.key = nn.Linear(channels, self.hidden_size * self.attention_heads)
        self.value = nn.Linear(channels, self.hidden_size * self.attention_heads)
        self.out = nn.Linear(self.hidden_size * self.attention_heads, channels)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, self.attention_heads, self.hidden_size, H * W)
        k = self.key(x).view(B, self.attention_heads, self.hidden_size, H * W)
        v = self.value(x).view(B, self.attention_heads, self.hidden_size, H * W)

        attention = F.softmax(torch.bmm(q, k.transpose(2, 3)), dim=-1)
        out = torch.bmm(v, attention.transpose(2, 3))
        out = out.view(B, self.attention_heads * self.hidden_size, H, W)
        out = self.out(out)
        return out

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), numResNetBlocksPerBlock=1, use_self_attention=False):
        super(DBlock, self).__init__()

        # Adjust the number of ResNet blocks based on the channels
        if out_channels == 128:
            numResNetBlocksPerBlock *= 2
        elif out_channels == 256:
            numResNetBlocksPerBlock *= 4
        elif out_channels == 512 or out_channels == 1024:
            numResNetBlocksPerBlock *= 8

        # Convolution layer without downsampling
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(1, 1), padding=1, bias=False)

        # Downsampling using MaxPool2d
        self.downsample = nn.MaxPool2d(2, 2)

        # Skip connection for downsampling
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        # Multiple ResNetBlocks
        self.resblocks = nn.Sequential(*[ResNetBlock(out_channels) for _ in range(numResNetBlocksPerBlock)])

        # Optional SelfAttention layer
        self.use_self_attention = use_self_attention
        if self.use_self_attention:
            self.self_attention = SelfAttention(out_channels)

    def forward(self, x):
        downsampled_x = self.downsample(x)  # Downsampling before convolution
        skip_x = self.skip(downsampled_x)
        x = self.conv(downsampled_x)
        x = self.resblocks(x)
        x += skip_x
        if self.use_self_attention:
            x = self.self_attention(x)
        return x

    

class EfficientNetEncoder(nn.Module):
    def __init__(self, version="efficientnet_b0", pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        self.encoder = timm.create_model(version, pretrained=pretrained, num_classes=0, global_pool='')
        self.name = version  # name attribute로 사용하기 위해 model의 version을 name으로 하여 속성 추가

    def forward(self, x):
        x = self.encoder(x)
        return x    

class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), numResNetBlocksPerBlock=1, use_self_attention=False, use_upsampling=True):
        super(UBlock, self).__init__()
        
        # use_upsampling 속성 초기화
        self.use_upsampling = use_upsampling

        # Multiple ResNetBlocks
        self.resblocks = nn.Sequential(*[ResNetBlock(out_channels) for _ in range(numResNetBlocksPerBlock)])

        # Convolution layer without upsampling
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(1, 1), padding=1, bias=False)

        # Upsampling using Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Optional SelfAttention layer
        self.use_self_attention = use_self_attention
        if self.use_self_attention:
            self.self_attention = SelfAttention(out_channels)

        # Convolution layer with optional upsampling
        if self.use_upsampling:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    def forward(self, x):
        x = self.resblocks(x)
        if self.use_self_attention:
            x = self.self_attention(x)
        x = self.conv(x)
        if self.use_upsampling:  # 조건 추가
            x = self.upsample(x)  # Upsampling after convolution
        return x
    
class EfficientUNet(nn.Module):
    def __init__(self, version="efficientnet_b0"):
        super(EfficientUNet, self).__init__()

        # Use EfficientNet as the encoder
        self.encoder = EfficientNetEncoder(version=version)

        # Initialize concat_input to True
        self.concat_input = True

        # Initial Convolution layer
        self.init_conv = nn.Conv2d(3, 256, kernel_size=3, padding=1, bias=False)

         # DBlocks
        self.dblock_256 = DBlock(256, 128, stride=(2, 2), numResNetBlocksPerBlock=4)
        self.dblock_128 = DBlock(128, 64, stride=(2, 2), numResNetBlocksPerBlock=2)
        self.dblock_64 = DBlock(64, 32, stride=(2, 2))
        self.dblock_32 = DBlock(32, 16, stride=(2, 2))
        
        # UBlocks
        self.ublock_16 = UBlock(16, 32, stride=(2, 2))
        self.ublock_32 = UBlock(32, 64, stride=(2, 2))
        self.ublock_64 = UBlock(64, 128, stride=(2, 2))
        self.ublock_128 = UBlock(128, 256, stride=(2, 2))
        self.ublock_256 = UBlock(256, 512, stride=(2, 2))
        self.ublock_512 = UBlock(512, 1024, stride=(2, 2))


        # Final Dense layer
        self.dense = nn.Conv2d(1024, 3, kernel_size=1, stride=1, bias=True)

    @property  # 각 efficientnet version에 따른 채널 수를 할당하는 메소드
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property  # 각 efficientnet version에 따른 이미지 크기를 할당하는 메소드
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x  # save input image data
        x = self.encoder(x)
        x = self.init_conv(x)

        # DBlock forward passes with skip connections
        skip_256 = self.dblock_256(x)
        skip_128 = self.dblock_128(skip_256)
        skip_64 = self.dblock_64(skip_128)
        x = self.dblock_32(skip_64)
        
        # UBlock forward passes with skip connections
        x = self.ublock_32(x)
        x = self.ublock_64(x)
        x = torch.cat([x, skip_64], dim=1)  # Skip connection using concat
        x = self.ublock_128(x)
        x = torch.cat([x, skip_128], dim=1)  # Skip connection using concat
        x = self.ublock_256(x)
        x = torch.cat([x, skip_256], dim=1)  # Skip connection using concat
        x = self.ublock_512(x)


        # input image와 final output 출력 연결
        if self.concat_input:
            x = torch.cat([x, input_], dim=1)

        x = self.dense(x)
        return x
        

# 각 버전별 U-Net 모델을 반환하는 함수들
def get_efficientunet_b0():
    return EfficientUNet(version="efficientnet_b0")

def get_efficientunet_b1():
    return EfficientUNet(version="efficientnet_b1")

def get_efficientunet_b2():
    return EfficientUNet(version="efficientnet_b2")

def get_efficientunet_b3():
    return EfficientUNet(version="efficientnet_b3")

def get_efficientunet_b4():
    return EfficientUNet(version="efficientnet_b4")

def get_efficientunet_b5():
    return EfficientUNet(version="efficientnet_b5")

def get_efficientunet_b6():
    return EfficientUNet(version="efficientnet_b6")

def get_efficientunet_b7():
    return EfficientUNet(version="efficientnet_b7")
