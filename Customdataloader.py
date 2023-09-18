from PIL import Image
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        while True:
            try:
                data = self.data_list[idx]
                wearing_img_path = os.path.join("trainexample/blur_model", data["wearing"])
                ia_img_path = os.path.join("trainexample/Ia", data["wearing"])
                org_img_path = os.path.join("trainexample/resized_org_model", data["wearing"])  # Original person image


                target_top = data['inner_top'] if data['inner_top'] is not None else data['main_top']

                jg_json_path = os.path.join("trainexample/Jg", f"{target_top}_F.json")
                ic_img_path = os.path.join("trainexample/Ic", f"{target_top}_F.jpg")

                jp_json_path = os.path.join("trainexample/Jp", data["wearing"].replace(".jpg", ".json"))

                wearing_img = Image.open(wearing_img_path).convert('RGB')
                ia_img = Image.open(ia_img_path).convert('RGB')
                ic_img = Image.open(ic_img_path).convert('RGB')

                combined_img = Image.fromarray(np.hstack((np.array(wearing_img), np.array(ia_img))))

                with open(jp_json_path, 'r') as f:
                    jp_data = json.load(f)
                    person_pose = torch.tensor(jp_data['landmarks'])

                with open(jg_json_path, 'r') as f:
                    jg_data = json.load(f)
                    garment_pose = torch.tensor(jg_data['landmarks'])

                org_img = Image.open(org_img_path).convert('RGB')  # Load original person image

                if self.transform:
                    combined_img = self.transform(combined_img)
                    ic_img = self.transform(ic_img)
                    org_img = self.transform(org_img)  # Transform original person image

                return combined_img, person_pose, garment_pose, ic_img, org_img

            except FileNotFoundError:
                idx = (idx + 1) % len(self.data_list)  # Move to the next item

class SuperResolutionDataset(Dataset):
    def __init__(self, image_dir, crop_size=256, downscale_factor=4):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self.crop_size = crop_size
        self.downscale_factor = downscale_factor

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        img = Image.open(img_path).convert('RGB')

        # Random cropping to get ground-truth patch
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.crop_size, self.crop_size))
        ground_truth = TF.crop(img, i, j, h, w)

        # Downsample ground-truth patch to create input image
        input_image = ground_truth.resize(
            (self.crop_size // self.downscale_factor, self.crop_size // self.downscale_factor),
            Image.BICUBIC)

        # Apply noise conditioning augmentation if needed
        # ...

        # Convert to tensor
        ground_truth = TF.to_tensor(ground_truth)
        input_image = TF.to_tensor(input_image)

        return input_image, ground_truth


