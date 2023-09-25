from PIL import Image
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

class EfficientUNetDataLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path)

        # 1024x1024 이미지에서 256x256 크기로 랜덤 크롭
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        image_crop = transforms.functional.crop(image, i, j, h, w)

        # 256x256 크기의 이미지를 64x64 크기로 다운샘플링
        image_downsample = transforms.Resize((64, 64))(image_crop)

        # 64x64 크기의 이미지에 약간의 노이즈 추가
        noise = torch.randn_like(transforms.ToTensor()(image_downsample)) * 0.05
        noisy_image = transforms.ToTensor()(image_downsample) + noise

        if self.transform:
            image_crop = self.transform(image_crop)

        return noisy_image, image_crop