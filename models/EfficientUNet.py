import torch
import torch.nn as nn
import timm

class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.activation(x)
        return x

class EfficientUNet(nn.Module):
    def __init__(self, version="efficientnet_b0"):
        super(EfficientUNet, self).__init__()
        self.encoder = timm.create_model(version, pretrained=True, num_classes=0, global_pool='')
        
        # UBlocks
        self.ublock1 = UBlock(1280, 1024)
        self.ublock2 = UBlock(1024, 512)
        self.ublock3 = UBlock(512, 256)
        self.ublock4 = UBlock(256, 128)
        self.ublock5 = UBlock(128, 64)
        self.ublock6 = UBlock(64, 32)
        self.ublock7 = UBlock(32, 16)
        
        # Final output layer
        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        
        x = self.ublock1(x)
        x = self.ublock2(x)
        x = self.ublock3(x)
        x = self.ublock4(x)
        x = self.ublock5(x)
        x = self.ublock6(x)
        x = self.ublock7(x)
        
        x = self.final_conv(x)
        return x



if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    
    model = EfficientUNet()
    
    out = model(x)
    
    print("Output tensor shape:", out.shape)
