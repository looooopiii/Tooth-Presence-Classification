import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
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
import json

# ==================== Dataset Training Configuration ====================
USE_ORIGINAL = True   # if use original dataset
USE_AUGMENTED = True  # if use augmented dataset (2400 samples)
USE_RANDOM = False     # if use random augmented (3600 samples)

# Original dataset roots
# 3D JSON label roots
JSON_ROOT_LOWER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower"
JSON_ROOT_UPPER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"

# 2D rendered image roots
IMG_ROOT_LOWER = "/home/user/lzhou/week15/render_output/train/lowerjaw"
IMG_ROOT_UPPER = "/home/user/lzhou/week15/render_output/train/upperjaw"

# CSV
AUGMENTED_CSV = "/home/user/lzhou/week10/output/augment_test/train_labels_augmented.csv"
RANDOM_CSV = "/home/user/lzhou/week10/output/augment_random/train_labels_random.csv"

# Rendered image directories
RENDER_ROOT = "/home/user/lzhou/week15/render_output/render_aug_random"
RENDER_ROOT_TEST = "/home/user/lzhou/week15/render_output/render_aug_test"

# Output directories
OUTPUT_DIR = "/home/user/lzhou/week15-32/output/Train2DAugTest/Augmented_32teeth_dynamit"
PLOT_DIR = f"{OUTPUT_DIR}/plots"

# Model filenames
BEST_MODEL_FILENAME = "augmented_dynamit_best_2d_32teeth.pth"
LAST_MODEL_FILENAME = "augmented_dynamit_last_2d_32teeth.pth"
PLOT_FILENAME = "training_metrics_dynamit.png"
METRICS_FILENAME = "detailed_metrics_dynamit.json"

# ==================== Training ====================
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
IMG_SIZE = 256
NUM_TEETH = 32
SEED = 41
BACKBONE = "resnet18"  # or "resnet50"

# Early stopping
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001

# Dropout
DROPOUT_RATE = 0.5

# FDI Notation
UPPER_TEETH = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_TEETH = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
ALL_TEETH = sorted(UPPER_TEETH + LOWER_TEETH)
FDI_TO_INDEX = {fdi: i for i, fdi in enumerate(ALL_TEETH)}
INDEX_TO_FDI = {i: fdi for fdi, i in FDI_TO_INDEX.items()}

# GPU config
available_gpus = []
device = None
# =======================================================


def get_free_gpus(threshold_mb=1000, max_gpus=2):
    """Get free GPU indices"""
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
            print("  Warning: No free GPUs found, using GPU 0")
            return [0]
        print(f" Free GPUs detected: {free_gpus}")
        return free_gpus
    except Exception as e:
        print(f" Error detecting free GPUs: {e}\n   Falling back to GPU 0")
        return [0]


def set_seed(seed):
    """set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_case_and_jaw(png_path):
    name = png_path.stem
    if name.endswith("_top"):
        core = name[:-4]
    else:
        core = name
    if core.endswith("_lower"):
        return core[:-6], "lower"
    if core.endswith("_upper"):
        return core[:-6], "upper"
    parts = core.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def load_json_labels(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return set(data.get('labels', []))
    except Exception as e:
        print(f"  Warning: failed to read {json_path}: {e}")
        return set()


def create_tooth_missing_vector(vertex_labels_set):
    tooth_presence = np.zeros(NUM_TEETH, dtype=np.float32)
    for fdi_label in vertex_labels_set:
        index = FDI_TO_INDEX.get(fdi_label)
        if index is not None:
            tooth_presence[index] = 1.0
    return 1.0 - tooth_presence


def build_original_samples(img_roots, json_roots):
    samples = []
    lower_img_root, upper_img_root = [Path(p) for p in img_roots]
    lower_json_root, upper_json_root = [Path(p) for p in json_roots]

    def add_samples(img_root, jaw, json_root):
        if not img_root.exists():
            print(f"  Warning: image root not found: {img_root}")
            return 0
        if not json_root.exists():
            print(f"  Warning: JSON root not found: {json_root}")
            return 0
        count = 0
        for png in sorted(img_root.glob("*_top.png")):
            case_id, cur_jaw = parse_case_and_jaw(png)
            if cur_jaw != jaw or case_id is None:
                continue
            json_path = json_root / case_id / f"{case_id}_{jaw}.json"
            if not json_path.exists():
                continue
            vertex_labels_set = load_json_labels(json_path)
            tooth_labels = create_tooth_missing_vector(vertex_labels_set)
            samples.append({
                'png_path': str(png),
                'labels': tooth_labels,
                'jaw_type': jaw,
                'case_id': case_id,
                'case_key': case_id
            })
            count += 1
        return count

    add_samples(lower_img_root, "lower", lower_json_root)
    add_samples(upper_img_root, "upper", upper_json_root)
    return samples


# ==================== Dataset Class ====================
class AugmentedToothDataset(Dataset):
    """
    reads tooth presence labels from CSV and loads rendered PNG images.
    CSV:
    - filename: OBJ file name
    - new_id: case ID
    - 11, 12, ..., 48: each tooth's label (1=missing, 0=present)

    PNG file naming conventions:
    - {render_root}/{jaw_type}/{case_id}_top.png
    """
    def __init__(self, csv_paths, render_root, transform=None):
        """
        Args:
            csv_paths: List of CSV file paths
            render_root: Root directory (or list of roots) for rendered PNG images
            transform: torchvision transforms
        """
        self.samples = []
        self.transform = transform
        if isinstance(render_root, (list, tuple, set)):
            self.render_roots = [Path(r) for r in render_root]
        else:
            self.render_roots = [Path(render_root)]
        
        # load data from all CSVs
        for csv_path in csv_paths:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            
            for idx, row in df.iterrows():
                # note jaw type from filename
                obj_filename = row['filename']
                obj_path = Path(obj_filename)
                case_dir = None
                if obj_path.parts and obj_path.parts[0].lower() in ('upper', 'lower'):
                    jaw_type = obj_path.parts[0].lower()
                    if len(obj_path.parts) > 1:
                        case_dir = obj_path.parts[1]
                elif 'lower' in obj_filename.lower():
                    jaw_type = 'lower'
                elif 'upper' in obj_filename.lower():
                    jaw_type = 'upper'
                else:
                    print(f"  Cannot determine jaw type for: {obj_filename}")
                    continue
                
                # generate PNG path
                case_id = row.get('new_id', Path(obj_filename).stem)
                png_filename = f"{case_id}_top.png"
                png_path = None
                for root in self.render_roots:
                    if case_dir:
                        candidate = root / jaw_type / case_dir / png_filename
                        if candidate.exists():
                            png_path = candidate
                            break
                    candidate = root / jaw_type / png_filename
                    if candidate.exists():
                        png_path = candidate
                        break
                
                # check if PNG exists, else try alternatives
                if png_path is None:
                    # try alternative naming conventions
                    alt_names = [
                        f"{case_id}.png",
                        f"{Path(obj_filename).stem}_top.png",
                        f"{Path(obj_filename).stem}.png"
                    ]
                    found = False
                    for alt_name in alt_names:
                        for root in self.render_roots:
                            if case_dir:
                                alt_path = root / jaw_type / case_dir / alt_name
                                if alt_path.exists():
                                    png_path = alt_path
                                    found = True
                                    break
                            alt_path = root / jaw_type / alt_name
                            if alt_path.exists():
                                png_path = alt_path
                                found = True
                                break
                        if found:
                            break
                    
                    if not found:
                        continue  # Skip if image not found
                
                # read tooth labels
                tooth_labels = []
                for tooth_fdi in ALL_TEETH:
                    col_name = str(tooth_fdi)
                    if col_name in row:
                        # CSV: 1 = missing, 0 = present
                        tooth_labels.append(float(row[col_name]))
                    else:
                        # if the column is missing
                        tooth_labels.append(1.0)
                
                tooth_labels = np.array(tooth_labels, dtype=np.float32)
                
                case_key = case_dir if case_dir else case_id
                self.samples.append({
                    'png_path': str(png_path),
                    'labels': tooth_labels,
                    'jaw_type': jaw_type,
                    'case_id': case_id,
                    'case_key': case_key
                })
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # load image
        try:
            img = Image.open(sample['png_path']).convert('RGB')
        except Exception as e:
            print(f"  Error loading image {sample['png_path']}: {e}")
            # use a black image as fallback
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
        
        labels = torch.from_numpy(sample['labels']).float()
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, labels, sample['jaw_type']


# ==================== Model ====================
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


# ==================== Loss Function (DYNAMIT) ====================
class Dynamit_Loss(nn.Module):
    """
    Dynamit Loss: Dynamic weighting based on class imbalance.
    Numerically stable version that avoids UserWarning.
    """
    def __init__(self):
        super(Dynamit_Loss, self).__init__()

    def forward(self, predictions, targets):
        S_pos = (targets == 1).sum().float()
        S_neg = (targets == 0).sum().float()
        epsilon = 1e-8
        
        # Keep calculations as tensors and detach at the end
        pos_coeff_raw = S_neg / (S_pos + epsilon)
        neg_coeff_raw = S_pos / (S_neg + epsilon)
        
        pos_coeff = torch.clamp(pos_coeff_raw, max=1.0).detach()
        neg_coeff = torch.clamp(neg_coeff_raw, max=1.0).detach()
        
        weights = torch.where(targets == 1, pos_coeff, neg_coeff)
        return F.binary_cross_entropy_with_logits(predictions, targets, weight=weights)


# ==================== Metrics ====================
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
    
    return per_tooth_metrics, {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_accuracy': macro_accuracy
    }


# ==================== Training ====================
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


# ==================== Plotting ====================
def plot_training_curves(history, save_dir, filename):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ResNet18 2D Training (Dynamit Loss) - Micro Metrics', fontsize=16, fontweight='bold')
    
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


# ==================== Custom Collate ====================
def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    jaw_types = [item[2] for item in batch]
    return imgs, labels, jaw_types


# ==================== MAIN ====================
def main():
    global available_gpus, device
    set_seed(SEED)
    
    # create output directories
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(" "*10 + "2D TRAINING (32 Teeth - ResNet18 + Dynamit Loss)")
    print("="*80)
    
    print("\n[0/5] Detecting free GPUs...")
    available_gpus = get_free_gpus()
    device = torch.device(f"cuda:{available_gpus[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Primary device: {device}")

    # load datasets
    csv_paths = []
    if USE_AUGMENTED and Path(AUGMENTED_CSV).exists():
        csv_paths.append(AUGMENTED_CSV)
    elif USE_AUGMENTED:
        print(f"  Augmented CSV not found: {AUGMENTED_CSV}")
    
    if USE_RANDOM and Path(RANDOM_CSV).exists():
        csv_paths.append(RANDOM_CSV)
    elif USE_RANDOM:
        print(f"  Random CSV not found: {RANDOM_CSV}")
    
    if not csv_paths and not USE_ORIGINAL:
        print(" ERROR: No datasets enabled!")
        print("   Please check USE_ORIGINAL/USE_AUGMENTED/USE_RANDOM")
        return

    # Data transforms
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
    full_dataset = AugmentedToothDataset(
        csv_paths=csv_paths,
        render_root=[RENDER_ROOT, RENDER_ROOT_TEST],
        transform=None
    )
    original_samples = []
    if USE_ORIGINAL:
        original_samples = build_original_samples(
            img_roots=[IMG_ROOT_LOWER, IMG_ROOT_UPPER],
            json_roots=[JSON_ROOT_LOWER, JSON_ROOT_UPPER]
        )
        if original_samples:
            full_dataset.samples.extend(original_samples)
    print(f"[Info] Loaded {len(full_dataset.samples)} 2D samples")
    
    if len(full_dataset) == 0:
        print(" ERROR: Dataset is empty! Please check:")
        print(f"   1. Augmented PNG images in: {RENDER_ROOT}/upper/, {RENDER_ROOT}/lower/, "
              f"{RENDER_ROOT_TEST}/upper/, {RENDER_ROOT_TEST}/lower/")
        print(f"   2. Original PNG images in: {IMG_ROOT_LOWER}, {IMG_ROOT_UPPER}")
        print(f"   3. JSON labels in: {JSON_ROOT_LOWER}, {JSON_ROOT_UPPER}")
        print(f"   4. Image naming: case_id_top.png")
        return

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
    
    # Create subset datasets
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            sample = self.dataset.samples[self.indices[idx]]
            try:
                img = Image.open(sample['png_path']).convert('RGB')
            except:
                img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
            
            labels = torch.from_numpy(sample['labels']).float()
            if self.transform is not None:
                img = self.transform(img)
            return img, labels, sample['jaw_type']
    
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

    # ========== KEY DIFFERENCE: Dynamit Loss instead of BCE ==========
    criterion = Dynamit_Loss()
    # =================================================================
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }

    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs...")
    print(f"       Loss Function: Dynamit Loss (Dynamic Weighting)")
    
    best_f1 = 0.0
    best_val_preds = None
    best_val_labels = None
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_preds, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['micro_f1'])

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
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val F1: {val_f1:.4f}")

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict()
        }, Path(OUTPUT_DIR) / LAST_MODEL_FILENAME)
        
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
                'loss_function': 'Dynamit_Loss',
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
