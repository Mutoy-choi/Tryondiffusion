import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.sqrt_d = d_model ** 0.5

    def forward(self, zt, Ic):
        Q = self.query(zt)
        K = self.key(Ic)
        V = self.value(Ic)
        attention_weights = F.softmax(Q @ K.transpose(-2, -1) / self.sqrt_d, dim=-1)
        return attention_weights @ V

class GarmentUNet(nn.Module):
    def __init__(self, d_model=64):
        super(GarmentUNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            CrossAttention(d_model),
            nn.ConvTranspose2d(d_model, 3, kernel_size=2, stride=2)
        )

    def forward(self, garment_input):
        return self.layers(garment_input)

class PersonUNet(nn.Module):
    def __init__(self, d_model=64):
        super(PersonUNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            CrossAttention(d_model),
            nn.ConvTranspose2d(d_model, 3, kernel_size=2, stride=2)
        )
        self.pose_embedding = nn.Linear(17 * 2, d_model)  # Assuming 17 keypoints for 2D pose

    def forward(self, person_input, pose):
        pose_embed = self.pose_embedding(pose)
        person_features = self.layers(person_input)
        return person_features + pose_embed

class EfficientUNet(nn.Module):
    def __init__(self):
        super(EfficientUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
