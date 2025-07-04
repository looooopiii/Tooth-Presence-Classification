import torch
import torch.nn as nn
import torchvision.models as models
import os

class MultiViewModel(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        
        # With pre-trained ResNet18 (grayscale input)
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()  # Remove the classification layer and keep the feature
        self.backbone = backbone

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 5, 512),
            nn.ReLU(),
            nn.Linear(512, 32),  # output 32-bit lable
        )

    def forward(self, x):
        # inout x shape: [B, 5, 1, 224, 224]
        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W)             # merge batch and view
        features = self.backbone(x)            # [B*5, 512]
        features = features.view(B, V * 512)   # [B, 5×512]
        out = self.fusion(features)            # [B, 32]
        return out