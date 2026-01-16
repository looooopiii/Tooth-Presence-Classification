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
from sklearn.model_selection import train_test_split
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
IMG_ROOT_LOWER = "/home/user/lzhou/week15/render_output/train/lowerjaw"
IMG_ROOT_UPPER = "/home/user/lzhou/week15/render_output/train/upperjaw"

# Outputs
OUTPUT_DIR = "/home/user/lzhou/week15-32/output/Train2D/32teeth_dynamit"
PLOT_DIR = "/home/user/lzhou/week15-32/output/Train2D/32teeth_dynamit/plots"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILENAME = "dynamit_best_2d_32teeth.pth"
LAST_MODEL_FILENAME = "dynamit_last_2d_32teeth.pth"
PLOT_FILENAME = "training_metrics_dynamit_32teeth.png"
METRICS_FILENAME = "detailed_metrics_dynamit_32teeth.json"

# ==================== HYPERPARAMETERS ====================
BATCH_SIZE = 16
NUM_EPOCHS = 35
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
IMG_SIZE = 256
NUM_TEETH = 32
SEED = 41
BACKBONE = "resnet18"

# Early stopping
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001

# Dropout rate
DROPOUT_RATE = 0.5

# FDI Notation
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}

# Upper/Lower teeth indices
UPPER_FDI = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
UPPER_IDX = [FDI_TO_INDEX[f] for f in UPPER_FDI]
LOWER_IDX = [FDI_TO_INDEX[f] for f in LOWER_FDI]

# GPU config
available_gpus = []
device = None
# =======================================================


def get_free_gpus(threshold_mb=1000, max_gpus=2):
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        free_gpus = [
            int(line.split(', ')[0]) 
            for line in result.stdout.strip().split('\n') 
            if int(line.split(', ')[1]) < threshold_mb
        ]
        if len(free_gpus) > max_gpus:
            free_gpus = free_gpus[:max_gpus]
        if not free_gpus:
            print("Warning: No free GPUs found, using GPU 0")
            return [0]
        print(f" Free GPUs detected: {free_gpus}")
        return free_gpus
    except Exception as e:
        print(f"Error detecting free GPUs: {e}\nFalling back to GPU 0")
        return [0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== DYNAMIC LOSS ====================
class DynamitLoss(nn.Module):
    """
    Batch-adaptive class-balancing loss.
    Positive class = missing (1), Negative = present (0).
    Weights are computed per-batch from target counts.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def forward(self, logits, targets):
        S_pos = (targets == 1).sum().float()
        S_neg = (targets == 0).sum().float()
        
        pos_coeff_val = min(1.0, (S_neg / S_pos).item()) if S_pos > 0 else 0.0
        neg_coeff_val = min(1.0, (S_pos / S_neg).item()) if S_neg > 0 else 0.0
        
        pos_coeff = torch.tensor(pos_coeff_val, device=self.device)
        neg_coeff = torch.tensor(neg_coeff_val, device=self.device)
        
        weights = torch.where(targets == 1, pos_coeff, neg_coeff)
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)


# ==================== DATASET ====================
class Tooth2DDataset(Dataset):
    def __init__(self, img_roots, json_roots, transform=None, is_training=True):
        self.samples = []
        self.transform = transform
        self.is_training = is_training
        
        lower_img_root, upper_img_root = [Path(p) for p in img_roots]
        lower_json_root, upper_json_root = [Path(p) for p in json_roots]

        def add_samples(img_root, jaw, json_root):
            if not img_root.exists():
                return
            for png in sorted(img_root.glob("*_top.png")):
                name = png.stem
                if name.endswith("_top"):
                    core = name[:-4]
                else:
                    core = name
                
                if core.endswith("_lower"):
                    case_id = core[:-6]
                    cur_jaw = "lower"
                elif core.endswith("_upper"):
                    case_id = core[:-6]
                    cur_jaw = "upper"
                else:
                    parts = core.split("_")
                    if len(parts) >= 2:
                        case_id, cur_jaw = parts[0], parts[1]
                    else:
                        continue
                
                if cur_jaw != jaw:
                    continue
                
                json_path = json_root / case_id / f"{case_id}_{jaw}.json"
                if json_path.exists():
                    self.samples.append({
                        'img': str(png),
                        'json': str(json_path),
                        'case_id': case_id,
                        'jaw': jaw,
                        'case_key': case_id
                    })
        
        add_samples(lower_img_root, "lower", lower_json_root)
        add_samples(upper_img_root, "upper", upper_json_root)
        print(f"[Info] Loaded {len(self.samples)} 2D samples")

    def __len__(self):
        return len(self.samples)

    def load_labels(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return set(data.get('labels', []))

    def create_tooth_missing_vector(self, vertex_labels_set, jaw_type):
        """
        Create a binary vector indicating missing teeth (1 = missing, 0 = present).
        Adjusts for upper/lower jaw teeth only.
        """
        tooth_presence = np.zeros(NUM_TEETH, dtype=np.float32)
        
        for fdi_label in vertex_labels_set:
            index = FDI_TO_INDEX.get(fdi_label)
            if index is not None:
                tooth_presence[index] = 1.0
        
        tooth_missing = 1.0 - tooth_presence
        return tooth_missing

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img']).convert('RGB')
        vertex_labels_set = self.load_labels(sample['json'])
        jaw_type = sample['jaw']
        
        tooth_labels = self.create_tooth_missing_vector(vertex_labels_set, jaw_type)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, torch.from_numpy(tooth_labels).float(), jaw_type


# ==================== MODEL====================
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
        
        # Classification head
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


# ==================== METRICS ====================
def calculate_micro_metrics(pred, target):
    """Calculate micro-averaged metrics"""
    pred_np = (pred > 0.5).cpu().numpy().astype(int)
    target_np = target.cpu().numpy().astype(int)
    
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average='micro', zero_division=0
    )
    accuracy = accuracy_score(target_flat, pred_flat)
    
    return {
        'micro_precision': precision,
        'micro_recall': recall,
        'micro_f1': f1,
        'micro_accuracy': accuracy
    }


def calculate_per_tooth_metrics(pred, target, num_teeth=32):
    """Calculate per-tooth metrics"""
    pred = (pred > 0.5).cpu().numpy().astype(int)
    target = target.cpu().numpy().astype(int)
    
    per_tooth_metrics = OrderedDict()
    for tooth_idx in range(num_teeth):
        fdi_label = INDEX_TO_FDI[tooth_idx]
        tooth_pred = pred[:, tooth_idx]
        tooth_target = target[:, tooth_idx]
        precision, recall, f1, _ = precision_recall_fscore_support(
            tooth_target, tooth_pred, average='binary', zero_division=0
        )
        accuracy = accuracy_score(tooth_target, tooth_pred)
        support = int(np.sum(tooth_target == 1))
        per_tooth_metrics[fdi_label] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'support': support
        }
    
    # Macro average
    macro_precision = np.mean([m['precision'] for m in per_tooth_metrics.values()])
    macro_recall = np.mean([m['recall'] for m in per_tooth_metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in per_tooth_metrics.values()])
    macro_accuracy = np.mean([m['accuracy'] for m in per_tooth_metrics.values()])
    
    macro_metrics = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_accuracy': macro_accuracy
    }
    
    return per_tooth_metrics, macro_metrics


# ==================== TRAINING ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for imgs, labels, jaw_types in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits.detach()))
        all_labels.append(labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    micro_metrics = calculate_micro_metrics(all_preds, all_labels)
    
    return {
        'loss': total_loss / len(dataloader),
        'micro_f1': micro_metrics['micro_f1'],
        'micro_precision': micro_metrics['micro_precision'],
        'micro_recall': micro_metrics['micro_recall'],
        'micro_accuracy': micro_metrics['micro_accuracy']
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels, jaw_types in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits))
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    micro_metrics = calculate_micro_metrics(all_preds, all_labels)
    
    return {
        'loss': total_loss / len(dataloader),
        'micro_f1': micro_metrics['micro_f1'],
        'micro_precision': micro_metrics['micro_precision'],
        'micro_recall': micro_metrics['micro_recall'],
        'micro_accuracy': micro_metrics['micro_accuracy']
    }, all_preds, all_labels


# ==================== PLOTTING ====================
def plot_training_curves(history, save_dir, filename):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ResNet18 2D Training with Dynamic Loss - Micro Metrics', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 (Micro)
    axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1 (micro)')
    axes[0, 1].plot(epochs, history['val_f1'], 'r-', label='Val F1 (micro)')
    axes[0, 1].set_title('F1 Score (Micro)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs, history['train_acc'], 'b-', label='Train Acc (micro)')
    axes[1, 0].plot(epochs, history['val_acc'], 'r-', label='Val Acc (micro)')
    axes[1, 0].set_title('Accuracy (Micro)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision/Recall
    axes[1, 1].plot(epochs, history['train_precision'], 'b--', label='Train Precision')
    axes[1, 1].plot(epochs, history['val_precision'], 'r--', label='Val Precision')
    axes[1, 1].plot(epochs, history['train_recall'], 'b:', label='Train Recall')
    axes[1, 1].plot(epochs, history['val_recall'], 'r:', label='Val Recall')
    axes[1, 1].set_title('Precision and Recall (Micro)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n✓ Training plots saved to: {save_path}")


# ==================== CUSTOM COLLATE ====================
def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    jaw_types = [item[2] for item in batch]
    return imgs, labels, jaw_types


# ==================== MAIN ====================
def main():
    global available_gpus, device
    set_seed(SEED)
    
    print("\n" + "="*80)
    print(" "*10 + "2D TRAINING (32 Teeth - ResNet18 + Dynamic Loss)")
    print("="*80)
    
    print("\n[0/5] Detecting free GPUs...")
    available_gpus = get_free_gpus()
    device = torch.device(f"cuda:{available_gpus[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Primary device: {device}")

    # Data augmentations (与 baseline 一致)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n[1/5] Building dataset...")
    full_dataset = Tooth2DDataset(
        img_roots=[IMG_ROOT_LOWER, IMG_ROOT_UPPER],
        json_roots=[JSON_ROOT_LOWER, JSON_ROOT_UPPER],
        transform=None,
        is_training=True
    )
    
    # Train/Val split
    print("\n[1.5/5] Performing train/val split...")
    case_to_indices = {}
    for idx, sample in enumerate(full_dataset.samples):
        case_key = sample.get('case_key') or sample.get('case_id')
        if case_key is None:
            continue
        case_to_indices.setdefault(case_key, []).append(idx)

    case_ids = sorted(case_to_indices.keys())
    train_cases, val_cases = train_test_split(
        case_ids, test_size=0.2, random_state=SEED
    )
    train_indices = [i for cid in train_cases for i in case_to_indices[cid]]
    val_indices = [i for cid in val_cases for i in case_to_indices[cid]]
    
    # Subset datasets with transforms
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            sample = self.dataset.samples[self.indices[idx]]
            img = Image.open(sample['img']).convert('RGB')
            vertex_labels_set = self.dataset.load_labels(sample['json'])
            jaw_type = sample['jaw']
            tooth_labels = self.dataset.create_tooth_missing_vector(vertex_labels_set, jaw_type)
            
            if self.transform is not None:
                img = self.transform(img)
            
            return img, torch.from_numpy(tooth_labels).float(), jaw_type
    
    train_dataset = TransformedSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformedSubset(full_dataset, val_indices, val_transform)
    
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )

    print("\n[2/5] Initializing 2D model...")
    model = ResNetMultiLabel(
        backbone=BACKBONE, 
        num_teeth=NUM_TEETH, 
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    if len(available_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)

    # ========== Dynamic Loss ==========
    criterion = DynamitLoss(device)
    print(" Using Dynamic Loss for class imbalance handling")
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )

    # Use ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }

    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs...")
    
    best_f1 = 0.0
    best_val_preds = None
    best_val_labels = None
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['micro_f1'])

        # Store metrics
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_f1'].append(train_metrics['micro_f1'])
        history['val_f1'].append(val_metrics['micro_f1'])
        history['train_acc'].append(train_metrics['micro_accuracy'])
        history['val_acc'].append(val_metrics['micro_accuracy'])
        history['train_precision'].append(train_metrics['micro_precision'])
        history['val_precision'].append(val_metrics['micro_precision'])
        history['train_recall'].append(train_metrics['micro_recall'])
        history['val_recall'].append(val_metrics['micro_recall'])
        
        val_f1 = val_metrics['micro_f1']
        
        # Print epoch summary
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val F1: {val_f1:.4f}")

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict()
        }, Path(OUTPUT_DIR) / LAST_MODEL_FILENAME)
        
        # Save best model
        if val_f1 > best_f1 + MIN_DELTA:
            best_f1 = val_f1
            best_val_preds = val_preds
            best_val_labels = val_labels
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict()
            }, Path(OUTPUT_DIR) / BEST_MODEL_FILENAME)
            print(f"        → Best F1 model saved (F1: {val_f1:.4f}) to {BEST_MODEL_FILENAME}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n Early stopping at epoch {epoch+1}")
                break

    print("\n" + "="*80)
    print(f"[4/5] Training complete!")
    print(f"Best validation F1 (micro): {best_f1:.4f}")
    print("="*80)

    # Per-tooth metrics
    print("\n[4.5/5] Calculating per-tooth metrics on best model...")
    per_tooth_metrics, macro_metrics = calculate_per_tooth_metrics(
        best_val_preds, best_val_labels, num_teeth=NUM_TEETH
    )
    
    micro_metrics = calculate_micro_metrics(best_val_preds, best_val_labels)

    print("\n MACRO-AVERAGED METRICS (across all 32 teeth):")
    print("-"*80)
    print(f"  Macro Precision: {macro_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {macro_metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {macro_metrics['macro_f1']:.4f}")
    print(f"  Macro Accuracy:  {macro_metrics['macro_accuracy']:.4f}")

    print("\n PER-TOOTH METRICS (FDI Notation):")
    print("-"*90)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-"*90)
    for fdi_label, metrics in per_tooth_metrics.items():
        print(f"Tooth {fdi_label:<5} {metrics['precision']:>10.4f}   {metrics['recall']:>10.4f}   "
              f"{metrics['f1']:>10.4f}   {metrics['accuracy']:>10.4f}   {metrics['support']:>8}")

    # Save metrics
    metrics_file = Path(OUTPUT_DIR) / METRICS_FILENAME
    with open(metrics_file, 'w') as f:
        serializable = {str(k): v for k, v in per_tooth_metrics.items()}
        json.dump({
            'micro_metrics': micro_metrics,
            'macro_metrics': macro_metrics, 
            'per_tooth_metrics': serializable,
            'training_config': {
                'loss_function': 'DynamitLoss',
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'dropout_rate': DROPOUT_RATE,
                'batch_size': BATCH_SIZE,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                'min_delta': MIN_DELTA
            }
        }, f, indent=2)
    print(f"\n Metrics saved to: {metrics_file}")

    print(f"\n[5/5] Generating training plots...")
    plot_training_curves(history, PLOT_DIR, PLOT_FILENAME)
    print("\n All done!")


if __name__ == "__main__":
    main()
