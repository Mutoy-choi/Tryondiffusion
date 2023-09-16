from PIL import Image
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
