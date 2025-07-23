
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import copy
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === CONFIGURATION ===
CSV_FILE = Path("/home/user/tbrighton/blender-scripts/tooth_labels.csv")
IMAGE_DIR = Path("/home/user/tbrighton/blender_outputs/multi_views")
PLOTS_DIR = Path("./plots") # Directory to save generated plots

# --- Model & Training Parameters ---
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Standard tooth numbering (FDI notation) for the entire mouth
UPPER_TEETH_L = [18, 17, 16, 15, 14, 13, 12, 11]
UPPER_TEETH_R = [21, 22, 23, 24, 25, 26, 27, 28]
LOWER_TEETH_L = [48, 47, 46, 45, 44, 43, 42, 41]
LOWER_TEETH_R = [31, 32, 33, 34, 35, 36, 37, 38]
ALL_TEETH = sorted(UPPER_TEETH_L + UPPER_TEETH_R + LOWER_TEETH_L + LOWER_TEETH_R)
NUM_CLASSES = len(ALL_TEETH)


PLOTS_DIR.mkdir(exist_ok=True)

# === 1. FOCAL LOSS IMPLEMENTATION ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        elif self.reduction == 'sum': return torch.sum(F_loss)
        else: return F_loss


# === 2. CUSTOM PYTORCH DATASET ===
class ToothDataset(Dataset):
    def __init__(self, df, all_teeth_map, transform=None):
        self.df = df
        self.transform = transform
        self.all_teeth_map = {tooth: i for i, tooth in enumerate(all_teeth_map)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Create a 32-element label vector for the entire mouth
        labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        # Populate the vector based on which teeth are present
        for tooth_col in self.df.columns:
            if tooth_col.startswith('tooth_') and row[tooth_col] == 1:
                tooth_num = int(tooth_col.split('_')[1])
                if tooth_num in self.all_teeth_map:
                    labels[self.all_teeth_map[tooth_num]] = 1.0
        
        return image, labels

# === 3. MODEL DEFINITION ===
def create_resnet50_multilabel(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

# === 4. METRICS & PLOTTING ===
def calculate_and_print_metrics(y_true, y_pred, tooth_labels):
    metrics_data = []
    for i, tooth in enumerate(tooth_labels):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics_data.append({
            'Tooth': tooth,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F1-Score': f1
        })

    metrics_df = pd.DataFrame(metrics_data).set_index('Tooth')
    print("\n--- Per-Tooth Performance Metrics (Validation Set) ---")
    print(metrics_df)
    
    # Print overall metrics
    print("\n--- Overall Performance Metrics (Validation Set) ---")
    print(f"Macro F1-Score:    {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro Precision:   {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro Recall:      {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    return metrics_df

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Model Training History')

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.savefig(PLOTS_DIR / 'training_history.png')
    print(f"\nSaved training history plot to {PLOTS_DIR / 'training_history.png'}")
    plt.close()

def plot_final_metrics(metrics_df):
    metrics_to_plot = metrics_df[['F1-Score', 'Precision', 'Sensitivity']]
    metrics_to_plot.plot(kind='bar', figsize=(20, 8), width=0.8)
    plt.title('Per-Tooth Performance Metrics on Validation Set')
    plt.ylabel('Score')
    plt.xlabel('Tooth Number (FDI)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'final_metrics_per_tooth.png')
    print(f"Saved final metrics plot to {PLOTS_DIR / 'final_metrics_per_tooth.png'}")
    plt.close()


# === 5. TRAINING & VALIDATION LOOP (Corrected) ===
def train_model(model, criterion, optimizer, dataloaders, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects, total_labels = 0.0, 0, 0
            
            # This part remains the same
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs) > 0.5
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_labels += labels.numel()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / total_labels
            
            # --- FIX IS HERE ---
            # Move loss and accuracy to CPU before appending for plotting
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.cpu()) # <-- Added .cpu()
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_unified_model.pth')
                print(f'New best model saved with accuracy: {best_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


# === 6. MAIN EXECUTION BLOCK ===
def main():
    print("Starting Unified Tooth Classification Model Training...")
    print(f"Device: {DEVICE}, Total Classes (Teeth): {NUM_CLASSES}")

    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} total samples from {CSV_FILE}")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': ToothDataset(train_df, ALL_TEETH, data_transforms['train']),
        'val': ToothDataset(val_df, ALL_TEETH, data_transforms['val'])
    }
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'val']}

    model = create_resnet50_multilabel(num_classes=NUM_CLASSES).to(DEVICE)
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # --- Start Training ---
    trained_model, history = train_model(model, criterion, optimizer, dataloaders, num_epochs=NUM_EPOCHS)
    
    # --- Final Evaluation and Reporting ---
    print("\n--- Evaluating final model on validation data... ---")
    trained_model.eval()
    all_val_labels, all_val_preds = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = trained_model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            all_val_labels.append(labels.cpu().numpy())
            all_val_preds.append(preds.cpu().numpy())
    
    y_true = np.concatenate(all_val_labels)
    y_pred = np.concatenate(all_val_preds)
    
    # Calculate, print, and plot metrics
    metrics_df = calculate_and_print_metrics(y_true, y_pred, ALL_TEETH)
    plot_training_history(history)
    plot_final_metrics(metrics_df)

    print("\nTraining Finished!")
    print(f"The best unified model has been saved as 'best_unified_model.pth'")

if __name__ == "__main__":
    main()