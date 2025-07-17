import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from tooth_dataset import ToothPresenceDataset

# ===== parameter =====
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "/home/user/lzhou/baseline_dataset_front_only.csv"
MODEL_PATH = "baseline_model.pth"

# ===== image preprocessing =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== dataset =====
dataset = ToothPresenceDataset(CSV_PATH, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# ===== Model construction (taking ResNet18 as an example）=====
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 改成灰度图输入
model.fc = nn.Linear(model.fc.in_features, 32)  # 输出32位标签
model = model.to(DEVICE)

# ===== Loss Function and Optimizer =====
weights = torch.tensor([1.0]*32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# ===== Training =====
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ===== validation =====
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.numel()

    avg_val_loss = val_loss / len(val_loader)
    acc = correct / total

    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")

    # ===== Precision / Recall / F1-score =====
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            y_pred.append(preds)
            y_true.append(labels.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    print("Per-tooth metrics:")
    for i in range(32):
        print(f"Tooth {i+1:02d} | P: {precision[i]:.3f} R: {recall[i]:.3f} F1: {f1[i]:.3f}")

# ===== save model =====
torch.save(model.state_dict(), MODEL_PATH)
print("Saved model as baseline_model.pth")

# Check whether the model file is saved successfully
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
    print(f"Model file saved successfully: {MODEL_PATH} ({os.path.getsize(MODEL_PATH) // 1024} KB)")
else:
    print("Warning: Model file is empty or not saved properly.")