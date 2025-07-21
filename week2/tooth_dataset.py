import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import ast 

class ToothPresenceDataset(Dataset):
    def __init__(self, csv_file, transform=None, debug=False):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.debug = debug

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Read image path and label
        img_path = self.data_frame.iloc[idx]['image_path']
        label_raw = self.data_frame.iloc[idx]['label']

        # Change string into list[int]
        if isinstance(label_raw, str):
            try:
                label_list = ast.literal_eval(label_raw)
            except:
                raise ValueError(f"Invalid label format at index {idx}: {label_raw}")
        else:
            label_list = label_raw

        # Check label
        assert len(label_list) == 32, f"Label at index {idx} has {len(label_list)} elements, expected 32."

        # Loading images (grayscale plots)
        image = Image.open(img_path).convert("L")

        # Apply transform
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label_tensor = torch.tensor(label_list, dtype=torch.float32)

        # Debug
        if self.debug:
            print(f"[DEBUG] Path: {img_path}")
            print(f"[DEBUG] Label: {label_list}")

        return image, label_tensor
