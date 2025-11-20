import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import subprocess
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet50

# ==================== CONFIGURATION ====================
# 3D JSON label roots
JSON_ROOT_LOWER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower"
JSON_ROOT_UPPER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"

# 2D rendered image roots
IMG_ROOT_LOWER = "/home/user/lzhou/week4/multi_views/lowerjaw"
IMG_ROOT_UPPER = "/home/user/lzhou/week4/multi_views/upperjaw"

# Outputs
OUTPUT_DIR = "/home/user/lzhou/week12/output/Train2D/normal"
PLOT_DIR = "/home/user/lzhou/week12/output/Train2D/normal/plots"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILENAME = "resnet18_bce_best_2d.pth"
LAST_MODEL_FILENAME = "resnet18_bce_last_2d.pth"
PLOT_FILENAME = "resnet18_bce_training_metrics_2d.png"
METRICS_FILENAME = "detailed_metrics_2d.json"

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 35
LEARNING_RATE = 1e-3
IMG_SIZE = 320
# NUM_TEETH = 32
NUM_OUTPUTS = 17  # 16 teeth + 1 jaw
NUM_TEETH_PER_JAW = 16
SEED = 41
BACKBONE = "resnet18"  # or "resnet50"

EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 1e-4

# FDI Notation
# new mapping for 16 teeth per jaw
# upperjaw: 18-11, 21-28 reflect to 0-15
UPPER_FDI_TO_IDX16 = {
    18: 0, 17: 1, 16: 2, 15: 3, 14: 4, 13: 5, 12: 6, 11: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15
}
# lowerjaw: 48-41, 31-38 reflect to 0-15
LOWER_FDI_TO_IDX16 = {
    48: 0, 47: 1, 46: 2, 45: 3, 44: 4, 43: 5, 42: 6, 41: 7,
    31: 8, 32: 9, 33: 10, 34: 11, 35: 12, 36: 13, 37: 14, 38: 15
}
# GPU config
available_gpus = []
device = None
# =======================================================

def get_free_gpus(threshold_mb=1000, max_gpus=2):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        free_gpus = [int(line.split(', ')[0]) for line in result.stdout.strip().split('\n') if int(line.split(', ')[1]) < threshold_mb]
        if len(free_gpus) > max_gpus:
            free_gpus = free_gpus[:max_gpus]
        if not free_gpus:
            print("Warning: No free GPUs found, using GPU 0"); return [0]
        print(f" Free GPUs detected: {free_gpus}"); return free_gpus
    except Exception as e:
        print(f"Error detecting free GPUs: {e}\nFalling back to GPU 0"); return [0]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ==================== DATASET (2D) ====================
class Tooth2DDataset(Dataset):
    """
    Reads rendered top-view PNGs from lower/upper folders and finds matching 3D JSON
    labels by case_id + jaw. File naming assumed: {case_id}_{jaw}_top.png
    """
    def __init__(self, img_roots, json_roots, transform=None):
        self.samples = []
        self.transform = transform
        lower_img_root, upper_img_root = [Path(p) for p in img_roots]
        lower_json_root, upper_json_root = [Path(p) for p in json_roots]

        def add_samples(img_root, jaw):
            if not img_root.exists():
                return
            for png in sorted(img_root.glob("*_top.png")):
                name = png.stem  # e.g., 0AAQ6BO3_lower_top
                # parse case_id and jaw
                if name.endswith("_top"):
                    core = name[:-4]
                else:
                    core = name
                # split from right: caseid_jaw
                if core.endswith("_lower"):
                    case_id = core[:-6]
                    cur_jaw = "lower"
                elif core.endswith("_upper"):
                    case_id = core[:-6]
                    cur_jaw = "upper"
                else:
                    # fallback: try provided jaw
                    parts = core.split("_")
                    if len(parts) >= 2:
                        case_id, cur_jaw = parts[0], parts[1]
                    else:
                        continue
                if cur_jaw != jaw:
                    continue
                # JSON path
                if jaw == "lower":
                    json_path = lower_json_root / case_id / f"{case_id}_lower.json"
                else:
                    json_path = upper_json_root / case_id / f"{case_id}_upper.json"
                if json_path.exists():
                    self.samples.append({
                        'img': str(png),
                        'json': str(json_path),
                        'case_id': case_id,
                        'jaw': jaw
                    })
        add_samples(lower_img_root, "lower")
        add_samples(upper_img_root, "upper")
        print(f"[Info] Loaded {len(self.samples)} 2D samples")

    def __len__(self):
        return len(self.samples)

    def load_labels(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return set(data.get('labels', []))

    def create_tooth_missing_vector(self, vertex_labels_set,jaw):
        # Initialize 17 outputs: 16 teeth + 1 jaw
        # 1 for missing, 0 for present
        output_vector = np.zeros(NUM_OUTPUTS, dtype=np.float32)
        # Determine jaw from one of the FDI labels
        if jaw == "upper":
            mapping = UPPER_FDI_TO_IDX16
            jaw_label = 0.0  # 0.0 upper jaw 
        elif jaw == "lower":
            mapping = LOWER_FDI_TO_IDX16
            jaw_label = 1.0  # 1.0 lower jaw
        else:
            # error case
            return output_vector 

        # label processing(0-15): present -> 1, missing -> 0
        present_teeth = np.zeros(NUM_TEETH_PER_JAW, dtype=np.float32)
        for fdi in vertex_labels_set:
            idx = mapping.get(fdi)
            if idx is not None:
                present_teeth[idx] = 1.0
        
        missing_teeth = 1.0 - present_teeth
        
        # Fill output vector
        output_vector[:NUM_TEETH_PER_JAW] = missing_teeth
        output_vector[NUM_TEETH_PER_JAW] = jaw_label # jaw info

        return output_vector

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img']).convert('RGB')
        
        # Load labels
        labels_set = self.load_labels(sample['json'])
        jaw = sample['jaw']
        
        # Create multi-label vector
        y = self.create_tooth_missing_vector(labels_set, jaw)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(y).float()

# ==================== MODEL (2D) ====================
class ResNetMultiLabel(nn.Module):
    def __init__(self, backbone="resnet18", num_outputs=17):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights="IMAGENET1K_V2")
        else:
            net = resnet18(weights="IMAGENET1K_V1")
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_outputs)
        self.net = net

    def forward(self, x):
        return self.net(x)

# ==================== METRICS ====================

def calculate_classification_metrics(pred_teeth, target_teeth):
# only for teeth (not jaw)
    pred = (pred_teeth > 0.5).cpu().numpy().astype(int)
    target = target_teeth.cpu().numpy().astype(int)
    
    pred_flat, target_flat = pred.flatten(), target.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(target_flat, pred_flat, average='binary', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy_score(target_flat, pred_flat)}

def calculate_jaw_accuracy(pred_jaw, target_jaw):
# only for jaw
    pred = (pred_jaw > 0.5).cpu().numpy().astype(int)
    target = target_jaw.cpu().numpy().astype(int)
    return {'jaw_accuracy': accuracy_score(target, pred)}

# ==================== TRAIN / VAL ====================

# function definition changed to accept two loss functions
def train_epoch(model, dataloader, criterion_teeth, criterion_jaw, optimizer, device):
    model.train(); total_loss = 0
    all_preds_teeth, all_labels_teeth = [], []
    all_preds_jaw, all_labels_jaw = [], []
    
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        
        # seperate teeth and jaw labels
        labels_teeth = labels[:, :NUM_TEETH_PER_JAW]
        labels_jaw = labels[:, NUM_TEETH_PER_JAW]

        optimizer.zero_grad()
        logits = model(imgs) # shape [B, 17]
        
        # seperate teeth and jaw logits
        logits_teeth = logits[:, :NUM_TEETH_PER_JAW]
        logits_jaw = logits[:, NUM_TEETH_PER_JAW] # shape [B]

        # calculate separate losses
        loss_teeth = criterion_teeth(logits_teeth, labels_teeth)
        loss_jaw = criterion_jaw(logits_jaw, labels_jaw)
        loss = loss_teeth + loss_jaw # total loss
        
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        
        # Collect predictions and labels
        all_preds_teeth.append(torch.sigmoid(logits_teeth).detach())
        all_labels_teeth.append(labels_teeth)
        all_preds_jaw.append(torch.sigmoid(logits_jaw).detach())
        all_labels_jaw.append(labels_jaw)
        
    all_preds_teeth = torch.cat(all_preds_teeth, dim=0)
    all_labels_teeth = torch.cat(all_labels_teeth, dim=0)
    all_preds_jaw = torch.cat(all_preds_jaw, dim=0)
    all_labels_jaw = torch.cat(all_labels_jaw, dim=0)

    # Calculate metrics
    avg_metrics = calculate_classification_metrics(all_preds_teeth, all_labels_teeth)
    avg_metrics.update(calculate_jaw_accuracy(all_preds_jaw, all_labels_jaw))
    avg_metrics['loss'] = total_loss / len(dataloader)
    return avg_metrics


# function definition changed to accept two loss functions
def validate(model, dataloader, criterion_teeth, criterion_jaw, device):
    model.eval(); total_loss = 0
    all_preds_teeth, all_labels_teeth = [], []
    all_preds_jaw, all_labels_jaw = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            labels_teeth = labels[:, :NUM_TEETH_PER_JAW]
            labels_jaw = labels[:, NUM_TEETH_PER_JAW]

            logits = model(imgs)
            logits_teeth = logits[:, :NUM_TEETH_PER_JAW]
            logits_jaw = logits[:, NUM_TEETH_PER_JAW]
            
            # calculate separate losses
            loss_teeth = criterion_teeth(logits_teeth, labels_teeth)
            loss_jaw = criterion_jaw(logits_jaw, labels_jaw)
            loss = loss_teeth + loss_jaw
            
            total_loss += loss.item()
            
            all_preds_teeth.append(torch.sigmoid(logits_teeth))
            all_labels_teeth.append(labels_teeth)
            all_preds_jaw.append(torch.sigmoid(logits_jaw))
            all_labels_jaw.append(labels_jaw)
            
    all_preds_teeth = torch.cat(all_preds_teeth, dim=0)
    all_labels_teeth = torch.cat(all_labels_teeth, dim=0)
    all_preds_jaw = torch.cat(all_preds_jaw, dim=0)
    all_labels_jaw = torch.cat(all_labels_jaw, dim=0)

    avg_metrics = calculate_classification_metrics(all_preds_teeth, all_labels_teeth)
    avg_metrics.update(calculate_jaw_accuracy(all_preds_jaw, all_labels_jaw))
    avg_metrics['loss'] = total_loss / len(dataloader)
    
    # Return predictions and labels for further analysis
    return avg_metrics, all_preds_teeth, all_labels_teeth, all_preds_jaw, all_labels_jaw


def plot_training_curves(history, save_dir, filename):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(22, 12)) 
    fig.suptitle('2D ResNet Training Metrics (16 Teeth + 1 Jaw)', fontsize=16, fontweight='bold')

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # F1 Score - Teeth
    axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1 (Teeth)')
    axes[0, 1].plot(epochs, history['val_f1'], 'r-', label='Val F1 (Teeth)')
    axes[0, 1].set_title('F1 Score (Micro - Teeth)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Jaw Accuracy - New Chart
    axes[0, 2].plot(epochs, history['train_jaw_acc'], 'b-', label='Train Jaw Acc')
    axes[0, 2].plot(epochs, history['val_jaw_acc'], 'r-', label='Val Jaw Acc')
    axes[0, 2].set_title('Jaw Classification Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Accuracy - Teeth
    axes[1, 0].plot(epochs, history['train_acc'], 'b-', label='Train Acc (Teeth)')
    axes[1, 0].plot(epochs, history['val_acc'], 'r-', label='Val Acc (Teeth)')
    axes[1, 0].set_title('Accuracy (Teeth)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Precision - Teeth
    axes[1, 1].plot(epochs, history['train_precision'], 'b-', label='Train Precision (Teeth)')
    axes[1, 1].plot(epochs, history['val_precision'], 'r-', label='Val Precision (Teeth)')
    axes[1, 1].set_title('Precision (Teeth)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Recall - Teeth
    axes[1, 2].plot(epochs, history['train_recall'], 'b-', label='Train Recall (Teeth)')
    axes[1, 2].plot(epochs, history['val_recall'], 'r-', label='Val Recall (Teeth)')
    axes[1, 2].set_title('Recall (Teeth)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n Training plots saved to: {save_path}")

# ==================== MAIN ====================

def main():
    global available_gpus, device
    set_seed(SEED)

    print("\n[0/5] Detecting free GPUs...")
    available_gpus = get_free_gpus(); device = torch.device(f"cuda:{available_gpus[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Primary device: {device}")

    print("\n[1/5] Building dataset...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Tooth2DDataset(
        img_roots=[IMG_ROOT_LOWER, IMG_ROOT_UPPER],
        json_roots=[JSON_ROOT_LOWER, JSON_ROOT_UPPER],
        transform=transform
    )
    train_size = int(0.8 * len(dataset)); val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    print("\n[2/5] Initializing 2D model...")
    model = ResNetMultiLabel(backbone=BACKBONE, num_outputs=NUM_OUTPUTS).to(device)
    if len(available_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)
    # Separate loss functions for teeth and jaw
    criterion_teeth = nn.BCEWithLogitsLoss().to(device)
    criterion_jaw = nn.BCEWithLogitsLoss().to(device)
    print(" Using separate BCEWithLogitsLosses for teeth (16) and jaw (1).")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 
               'train_acc': [], 'val_acc': [], 'train_precision': [], 'val_precision': [], 
               'train_recall': [], 'val_recall': [], 'train_jaw_acc': [], 'val_jaw_acc': []} # jaw acc

    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs..."); print("=" * 80)
    best_f1 = 0.0; 
    # best_val_preds = None; best_val_targets = None

    for epoch in range(NUM_EPOCHS):
        # two loss functions passed
        train_metrics = train_epoch(model, train_loader, criterion_teeth, criterion_jaw, optimizer, device)
        val_metrics, val_preds_teeth, val_labels_teeth, val_preds_jaw, val_labels_jaw = validate(model, val_loader, criterion_teeth, criterion_jaw, device)
        scheduler.step()

        for key in ['loss', 'f1', 'acc', 'precision', 'recall']:
            history[f'train_{key}'].append(train_metrics.get(key.replace('acc', 'accuracy'), 0))
            history[f'val_{key}'].append(val_metrics.get(key.replace('acc', 'accuracy'), 0))
        history['train_jaw_acc'].append(train_metrics.get('jaw_accuracy', 0))
        history['val_jaw_acc'].append(val_metrics.get('jaw_accuracy', 0))
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1 (teeth): {val_metrics['f1']:.4f} | Val Jaw Acc: {val_metrics['jaw_accuracy']:.4f}")

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict()}, Path(OUTPUT_DIR) / LAST_MODEL_FILENAME)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']; 
            # best_val_preds = val_preds; best_val_targets = val_targets
            torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict()}, Path(OUTPUT_DIR) / BEST_MODEL_FILENAME)
            print(f"        → Best F1 model saved (F1: {val_metrics['f1']:.4f}) to {BEST_MODEL_FILENAME}")

    print("\n" + "=" * 80 + f"\n[4/5] Training complete!\nBest validation F1 (micro): {best_f1:.4f}")
    
    # (Per-tooth metrics section removed as per original file, but can be added back if needed)

    print(f"\n[5/5] Generating training plots...")
    plot_training_curves(history, PLOT_DIR, PLOT_FILENAME)
    print("\n All done!")

if __name__ == "__main__":
    main()