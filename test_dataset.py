from torch.utils.data import DataLoader
from tooth_dataset import ToothDataset
import os

dataset = ToothDataset("/home/user/lzhou/5views-json/dataset.json")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for imgs, labels in loader:
    print("Image batch shape:", imgs.shape)      # [B, 5, 1, 224, 224]
    print("Label batch shape:", labels.shape)    # [B, 32]
    break