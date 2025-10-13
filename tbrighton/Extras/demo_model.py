
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path
import copy
import time

# === CONFIGURATION ===

CSV_FILE = Path("/home/user/tbrighton/blender-scripts/tooth_labels.csv")
IMAGE_DIR = Path("/home/user/tbrighton/blender_outputs/multi_views")


JAW_TYPE_TO_TRAIN = 'upper'
NUM_EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Focal Loss parameters
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if JAW_TYPE_TO_TRAIN == 'upper':
    TOOTH_COLS = [f'tooth_{i}' for i in [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]]
else:
    TOOTH_COLS = [f'tooth_{i}' for i in [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]]

NUM_CLASSES = len(TOOTH_COLS)


# === 1. FOCAL LOSS IMPLEMENTATION ===
class FocalLoss(nn.Module):
    """
    Custom Focal Loss implementation for multi-label classification.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C), logits from the model
        # targets: (N, C), binary ground truth
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt
        pt = torch.exp(-BCE_loss)
        
        # Calculate Focal Loss
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# === 2. CUSTOM PYTORCH DATASET ===
class ToothDataset(Dataset):
    """
    Dataset for loading tooth images and their corresponding presence labels.
    """
    def __init__(self, df, tooth_columns, transform=None):
        self.df = df
        self.transform = transform
        self.tooth_columns = tooth_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.df.iloc[idx]['image_path']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}. Please check your IMAGE_DIR path.")
            # Return a dummy tensor if image not found to avoid crashing the loader
            return torch.zeros((3, 224, 224)), torch.zeros((len(self.tooth_columns)))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = self.df.iloc[idx][self.tooth_columns].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels

# === 3. MODEL DEFINITION ===
def create_resnet50_multilabel(num_classes):

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze all layers except the final one
    for param in model.parameters():
        param.requires_grad = False
        
    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

# === 4. TRAINING & VALIDATION LOOP ===
def train_model(model, criterion, optimizer, dataloaders, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            total_labels = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # For accuracy calculation
                    preds = torch.sigmoid(outputs) > 0.5
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                total_labels += labels.size(0) * labels.size(1) # samples * num_classes

            epoch_loss = running_loss / total_samples
            # Accuracy is calculated as the total number of correct label predictions
            # divided by the total number of labels (samples * 16)
            epoch_acc = running_corrects.double() / total_labels

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                torch.save(model.state_dict(), f'best_model_{JAW_TYPE_TO_TRAIN}.pth')
                print(f'New best model saved with accuracy: {best_acc:.4f}')


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# === 5. MAIN EXECUTION BLOCK ===
def main():
    print("Starting Tooth Classification Model Training...")
    print(f"Device: {DEVICE}")
    print(f"Training for: {JAW_TYPE_TO_TRAIN.capitalize()} Jaw")
    print(f"Number of classes (teeth): {NUM_CLASSES}")

    # --- Load and Prepare Data ---
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Label file not found at {CSV_FILE}. Please run tooth_label.py first.")
    
    df = pd.read_csv(CSV_FILE)
    
    # Filter for the specific jaw type
    df_jaw = df[df['jaw'] == JAW_TYPE_TO_TRAIN].copy()
    print(f"Found {len(df_jaw)} samples for the {JAW_TYPE_TO_TRAIN} jaw.")
    
    if len(df_jaw) == 0:
        print("No data found for the selected jaw type. Exiting.")
        return

    # Split data into training and validation sets
    train_df, val_df = train_test_split(df_jaw, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # --- Define Data Transformations ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- Create Datasets and Dataloaders ---
    image_datasets = {
        'train': ToothDataset(train_df, TOOTH_COLS, data_transforms['train']),
        'val': ToothDataset(val_df, TOOTH_COLS, data_transforms['val'])
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    # --- Initialize Model, Loss, and Optimizer ---
    model = create_resnet50_multilabel(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print("Model created and moved to device.")

    # Unfreeze final layer to be trained
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # --- Start Training ---
    trained_model = train_model(model, criterion, optimizer, dataloaders, num_epochs=NUM_EPOCHS)
    
    print("\nTraining Finished!")
    print(f"The best model for the '{JAW_TYPE_TO_TRAIN}' jaw has been saved as 'best_model_{JAW_TYPE_TO_TRAIN}.pth'")


if __name__ == "__main__":
    main()