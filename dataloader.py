import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import glob
from PIL import Image

# Define the custom dataset class
class VirtualTryOnDataset(Dataset):
    def __init__(self, ia_dir, ic_dir, jp_dir, jg_dir, transform=None):
        self.ia_paths = glob.glob(os.path.join(ia_dir, "*.jpg"))
        self.ic_paths = glob.glob(os.path.join(ic_dir, "*.jpg"))
        self.jp_paths = glob.glob(os.path.join(jp_dir, "*.json"))
        self.jg_paths = glob.glob(os.path.join(jg_dir, "*.json"))
        self.transform = transform

    def __len__(self):
        return min(len(self.ia_paths), len(self.ic_paths), len(self.jp_paths), len(self.jg_paths))

    def __getitem__(self, idx):
        ia_image = Image.open(self.ia_paths[idx]).convert("RGB")
        ic_image = Image.open(self.ic_paths[idx]).convert("RGB")
        jp_keypoints = json.load(open(self.jp_paths[idx], 'r'))['landmarks']
        jg_keypoints = json.load(open(self.jg_paths[idx], 'r'))['landmarks']

        # Convert flat keypoints list to an array of shape (num_keypoints, 3)
        jp_keypoints = np.array(jp_keypoints).reshape(-1, 3)[:, :2]  # Only keep x, y coordinates
        jg_keypoints = np.array(jg_keypoints).reshape(-1, 3)[:, :2]  # Only keep x, y coordinates

        # Apply transforms to images if specified
        if self.transform:
            ia_image = self.transform(ia_image)
            ic_image = self.transform(ic_image)

        return ia_image, jp_keypoints, ic_image, jg_keypoints
