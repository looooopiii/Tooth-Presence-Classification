import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
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
from typing import Optional
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet50

# ============= CONFIGURATION =============
# Augmented CSV data sources (same as teammate)
DATA_SOURCES_CSV = [
    ("/home/user/lzhou/week10/output/augment_test/train_labels_augmented.csv", 
     Path("/home/user/lzhou/week10/output/augment_test")),
    ("/home/user/lzhou/week10/output/augment_random/train_labels_random.csv", 
     Path("/home/user/lzhou/week10/output/augment_random")),
]

# Render roots for 2D images
RENDER_ROOT_RANDOM = Path("/home/user/lzhou/week15/render_output/render_aug_random")
RENDER_ROOT_TEST = Path("/home/user/lzhou/week15/render_output/render_aug_test")

# Output directories
OUTPUT_DIR = Path("/home/user/lzhou/week15-32/output/Train2D/aug_normal_32teeth")
PLOT_DIR = Path("/home/user/lzhou/week15-32/output/Train2D/aug_normal_32teeth/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILENAME = "aug_bce_best_2d_32teeth.pth"
LAST_MODEL_FILENAME = "aug_bce_last_2d_32teeth.pth"
PLOT_FILENAME = "training_metrics_aug_bce_32teeth.png"
METRICS_FILENAME = "detailed_metrics_aug_bce_32teeth.json"

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
IMG_SIZE = 256
NUM_TEETH = 32
SEED = 41
BACKBONE = "resnet18"
DROPOUT_RATE = 0.5
LABEL_SMOOTHING = 0.1

# FDI Notation
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi: i for i, fdi in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi for fdi, i in FDI_TO_INDEX.items()}

available_gpus = []
device = None


# =========================================
# HELPER FUNCTIONS 
# =========================================
def get_free_gpus(threshold_mb=1000, max_gpus=2):
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'],
                          capture_output=True, text=True)
        gpus = [int(line.split(', ')[0]) for line in r.stdout.strip().split('\n') 
                if int(line.split(', ')[1]) < threshold_mb]
        if len(gpus) > max_gpus:
            gpus = gpus[:max_gpus]
        if not gpus:
            print("Warning: No free GPUs found, using GPU 0")
            return [0]
        print(f"✓ Free GPUs detected: {gpus}")
        return gpus
    except Exception as e:
        print(f"Error detecting GPUs: {e}\nFalling back to GPU 0")
        return [0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================
# LOSS FUNCTION
# =========================================
class BCEWithLogitsLossSmoothed(nn.Module):
    """BCE Loss with label smoothing - same as baseline"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets_smooth)


# =========================================
# DATA TRANSFORMS
# =========================================
def build_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# =========================================
# PATH RESOLUTION
# =========================================
def resolve_png_path(rel_obj_path: str, primary_root: Path) -> Optional[Path]:
    """Resolve PNG path from OBJ filename"""
    if not isinstance(rel_obj_path, str):
        return None
    
    rel_obj = Path(rel_obj_path)
    png_name = rel_obj.with_suffix("").name + "_top.png"
    lower_name = rel_obj_path.lower()
    
    jaw = None
    if "upper" in lower_name:
        jaw = "upper"
    elif "lower" in lower_name:
        jaw = "lower"
    
    case_dir = rel_obj.parent.name if rel_obj.parent.name else None

    candidate_roots = [primary_root] if primary_root else []
    candidate_roots.extend([RENDER_ROOT_RANDOM, RENDER_ROOT_TEST])

    for root in candidate_roots:
        root = Path(root)
        if jaw and case_dir:
            cand = root / jaw / case_dir / png_name
            if cand.exists():
                return cand
        cand = root / png_name
        if cand.exists():
            return cand
    return None


# =========================================
# MODEL
# =========================================
class ResNetMultiLabel(nn.Module):
    def __init__(self, backbone="resnet18", num_teeth=32, dropout_rate=0.5):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights="IMAGENET1K_V2")
        else:
            net = resnet18(weights="IMAGENET1K_V1")
        
        in_feats = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

        # 3-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_teeth)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# =========================================
# DATASET
# =========================================
class CSVToothDataset(Dataset):
    """Dataset for tooth presence from CSV file - follows teammate's 3D logic"""
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        df = pd.read_csv(csv_file)
        missing_count = 0
        
        for _, row in df.iterrows():
            png_path = resolve_png_path(row.get("filename", ""), self.root_dir)
            if png_path is None:
                missing_count += 1
                continue
            
            try:
                # Read labels directly from CSV (0=present, 1=missing)
                labels = row[[str(fdi) for fdi in VALID_FDI_LABELS]].astype(np.float32).values
            except KeyError:
                continue
            
            if labels.shape[0] != NUM_TEETH:
                continue
            
            self.samples.append((png_path, labels))
        
        if missing_count > 0:
            print(f"[WARN] {missing_count} rows in {csv_file} skipped (no matching PNG).")
        print(f"  Loaded {len(self.samples)} samples from {csv_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        png_path, labels = self.samples[idx]
        try:
            img = Image.open(png_path).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.from_numpy(labels).float()


# =========================================
# METRICS
# =========================================
def calculate_classification_metrics(predictions, targets):
    """Calculate micro-averaged metrics - same as teammate"""
    preds_binary = (predictions > 0.5).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    preds_flat = preds_binary.flatten()
    targets_flat = targets_np.flatten()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_flat, preds_flat, average='binary', zero_division=0
    )
    acc = accuracy_score(targets_flat, preds_flat)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'acc': acc
    }


def calculate_per_tooth_metrics(predictions, targets):
    """Calculate per-tooth metrics - same as teammate"""
    preds_binary = (predictions > 0.5).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    metrics = OrderedDict()
    for i in range(NUM_TEETH):
        fdi = INDEX_TO_FDI[i]
        p, r, f1, _ = precision_recall_fscore_support(
            targets_np[:, i], preds_binary[:, i], average='binary', zero_division=0
        )
        acc = accuracy_score(targets_np[:, i], preds_binary[:, i])
        metrics[fdi] = {
            'precision': float(p),
            'recall': float(r),
            'f1': float(f1),
            'accuracy': float(acc),
            'support': int(targets_np[:, i].sum())
        }
    
    macro_metrics = {
        'macro_precision': np.mean([m['precision'] for m in metrics.values()]),
        'macro_recall': np.mean([m['recall'] for m in metrics.values()]),
        'macro_f1': np.mean([m['f1'] for m in metrics.values()]),
        'macro_accuracy': np.mean([m['accuracy'] for m in metrics.values()])
    }
    
    return metrics, macro_metrics


# =========================================
# TRAINING FUNCTIONS
# =========================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits.detach()))
        all_labels.append(labels)
    
    metrics = calculate_classification_metrics(torch.cat(all_preds), torch.cat(all_labels))
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits))
            all_targets.append(labels)
    
    metrics = calculate_classification_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics['loss'] = total_loss / len(dataloader)
    return metrics, torch.cat(all_preds), torch.cat(all_targets)


def plot_training_curves(history, save_dir):
    """Plot training curves - same style as teammate"""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ResNet18 2D Training Metrics (Augmented Dataset - BCE Loss)', fontsize=16, fontweight='bold')
    
    ax[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax[0, 0].set_title('Loss')
    ax[0, 0].legend()
    ax[0, 0].grid(True, alpha=0.3)
    
    ax[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1')
    ax[0, 1].plot(epochs, history['val_f1'], 'r-', label='Val F1')
    ax[0, 1].set_title('F1 Score (Micro)')
    ax[0, 1].legend()
    ax[0, 1].grid(True, alpha=0.3)
    
    ax[1, 0].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax[1, 0].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax[1, 0].set_title('Accuracy')
    ax[1, 0].legend()
    ax[1, 0].grid(True, alpha=0.3)
    
    ax[1, 1].plot(epochs, history['train_precision'], 'b--', label='Train Precision')
    ax[1, 1].plot(epochs, history['val_precision'], 'r--', label='Val Precision')
    ax[1, 1].plot(epochs, history['train_recall'], 'b:', label='Train Recall')
    ax[1, 1].plot(epochs, history['val_recall'], 'r:', label='Val Recall')
    ax[1, 1].set_title('Precision & Recall')
    ax[1, 1].legend()
    ax[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    sp = Path(save_dir) / PLOT_FILENAME
    plt.savefig(sp, dpi=300)
    plt.close()
    print(f"\n✓ Plots saved: {sp}")


# =========================================
# MAIN EXECUTION
# =========================================
def main():
    global available_gpus, device
    set_seed(SEED)
    
    print("\n[0/5] Setting up environment...")
    available_gpus = get_free_gpus()
    device = torch.device(f"cuda:{available_gpus[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Primary device: {device}")
    
    print("\n[1/5] Loading and combining datasets...")
    datasets = []
    
    for csv_path, root_dir in DATA_SOURCES_CSV:
        if Path(csv_path).exists():
            ds = CSVToothDataset(csv_path, root_dir, transform=None)
            datasets.append(ds)
        else:
            print(f"[WARN] CSV not found: {csv_path}")
    
    if not datasets:
        print("[ERROR] No datasets found!")
        return
    
    # Combine all samples
    all_samples = []
    for ds in datasets:
        all_samples.extend(ds.samples)
    
    print(f" Combined dataset loaded with {len(all_samples)} samples.")
    
    # Analyze label distribution 
    all_labels = np.array([s[1] for s in all_samples])
    missing_rate = all_labels.mean()
    print(f"  Overall missing rate: {missing_rate:.2%}")
    
    # Train/Val split
    train_size = int(0.9 * len(all_samples))
    val_size = len(all_samples) - train_size
    
    random.shuffle(all_samples)
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:]
    
    # Create datasets with transforms
    class SampleListDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            png_path, labels = self.samples[idx]
            try:
                img = Image.open(png_path).convert("RGB")
            except:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            if self.transform:
                img = self.transform(img)
            return img, torch.from_numpy(labels).float()
    
    train_dataset = SampleListDataset(train_samples, build_train_transform())
    val_dataset = SampleListDataset(val_samples, build_val_transform())
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True
    )
    
    print("\n[2/5] Initializing model...")
    model = ResNetMultiLabel(backbone=BACKBONE, num_teeth=NUM_TEETH, dropout_rate=DROPOUT_RATE).to(device)
    
    if len(available_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)
        print(f" Model wrapped for {len(available_gpus)} GPUs.")
    
    # BCE Loss with label smoothing
    criterion = BCEWithLogitsLossSmoothed(smoothing=LABEL_SMOOTHING)
    print(f" Using BCE Loss with Label Smoothing ({LABEL_SMOOTHING})")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }
    
    best_f1 = 0.0
    best_val_preds, best_val_targets = None, None
    
    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs...")
    print("=" * 80)
    
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        # Record history
        for key in ['loss', 'f1', 'acc', 'precision', 'recall']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        
        print(f"E {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train L:{train_metrics['loss']:.4f}, F1:{train_metrics['f1']:.4f} | "
              f"Val L:{val_metrics['loss']:.4f}, F1:{val_metrics['f1']:.4f}")
        
        # Save models
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict()
        }, OUTPUT_DIR / LAST_MODEL_FILENAME)
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_val_preds, best_val_targets = val_preds, val_targets
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict()
            }, OUTPUT_DIR / BEST_MODEL_FILENAME)
            print(f"  →  Best F1 model saved (F1: {val_metrics['f1']:.4f})")
    
    print("\n" + "=" * 80)
    print(f"[4/5] Training complete! Best F1: {best_f1:.4f}")
    
    print("\n[4.5/5] Calculating metrics...")
    ptm, mm = calculate_per_tooth_metrics(best_val_preds, best_val_targets)
    
    print("\n MACRO-AVERAGED METRICS:")
    print("-" * 80)
    print(f"  Macro Precision: {mm['macro_precision']:.4f}")
    print(f"  Macro Recall:    {mm['macro_recall']:.4f}")
    print(f"  Macro F1:        {mm['macro_f1']:.4f}")
    print(f"  Macro Accuracy:  {mm['macro_accuracy']:.4f}")
    
    print("\n PER-TOOTH METRICS:")
    print("-" * 80)
    print(f"{'FDI':<10}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Accuracy':<12}{'Support':<10}")
    print("-" * 80)
    
    for fdi, mets in ptm.items():
        print(f"  {fdi:<8} {mets['precision']:>10.4f}  {mets['recall']:>10.4f}  "
              f"{mets['f1']:>10.4f}  {mets['accuracy']:>10.4f}  {mets['support']:>8}")
    
    # Save metrics
    mf = OUTPUT_DIR / METRICS_FILENAME
    with open(mf, 'w') as f:
        json.dump({
            'macro_metrics': mm,
            'per_tooth_metrics': {str(k): v for k, v in ptm.items()}
        }, f, indent=2)
    print(f"\n Metrics saved: {mf}")
    
    print(f"\n[5/5] Generating plots...")
    plot_training_curves(history, PLOT_DIR)
    print("\n All done!")


if __name__ == "__main__":
    main()