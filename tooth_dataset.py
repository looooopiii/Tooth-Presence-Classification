import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os


class ToothDataset(Dataset):
    def __init__(self, json_path, image_size=224):
        """
        Args:
            json_path (str or Path): path to dataset.json
            image_size (int): target image size (square), default 224
        """
        self.json_path = Path(json_path)
        with open(self.json_path, "r") as f:
            self.data = json.load(f)

        # Grayscale + resize + to tensor + normalize
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convert to [0,1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_paths = item["images"]
        # print the path
        for p in img_paths:
            if not os.path.exists(p):
                print(f"Image not found: {p}")
            else:
                print(f"Loading: {p}")
    
        imgs = [self.transform(Image.open(p).convert('L')) for p in img_paths]  # Grayscale image
        imgs = torch.stack(imgs)  # [5, 1, 224, 224]

        label = torch.tensor(item["label"]).float()
        return imgs, label