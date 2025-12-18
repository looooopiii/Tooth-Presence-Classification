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
OUTPUT_DIR = "/home/user/lzhou/week10/output/Train2D/normal"
PLOT_DIR = "/home/user/lzhou/week10/output/Train2D/normal/plots"
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
NUM_TEETH = 32
SEED = 41
BACKBONE = "resnet18"  # or "resnet50"

EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 1e-4

# FDI Notation
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}

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

    def create_tooth_missing_vector(self, vertex_labels_set):
        # 1 for missing, 0 for present
        present = np.zeros(NUM_TEETH, dtype=np.float32)
        for fdi in vertex_labels_set:
            idx = FDI_TO_INDEX.get(fdi)
            if idx is not None:
                present[idx] = 1.0
        missing = 1.0 - present
        return missing

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img']).convert('RGB')
        labels_set = self.load_labels(sample['json'])
        y = self.create_tooth_missing_vector(labels_set)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(y).float()

# ==================== MODEL (2D) ====================
class ResNetMultiLabel(nn.Module):
    def __init__(self, backbone="resnet18", num_teeth=32):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights="IMAGENET1K_V2")
        else:
            net = resnet18(weights="IMAGENET1K_V1")
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_teeth)
        self.net = net
    def forward(self, x):
        return self.net(x)

# ==================== METRICS ====================

def calculate_classification_metrics(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(int); target = target.cpu().numpy().astype(int)
    pred_flat, target_flat = pred.flatten(), target.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(target_flat, pred_flat, average='binary', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy_score(target_flat, pred_flat)}


def calculate_per_tooth_metrics(pred, target, num_teeth=32):
    pred = (pred > 0.5).cpu().numpy().astype(int)
    target = target.cpu().numpy().astype(int)
    per_tooth_metrics = OrderedDict()
    for tooth_idx in range(num_teeth):
        fdi_label = INDEX_TO_FDI[tooth_idx]
        tooth_pred = pred[:, tooth_idx]
        tooth_target = target[:, tooth_idx]
        precision, recall, f1, _ = precision_recall_fscore_support(tooth_target, tooth_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(tooth_target, tooth_pred)
        support = int(np.sum(tooth_target == 1))
        per_tooth_metrics[fdi_label] = {
            'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy, 'support': support
        }
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

# ==================== TRAIN / VAL ====================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train(); total_loss = 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).detach()); all_labels.append(labels)
    all_preds = torch.cat(all_preds, dim=0); all_labels = torch.cat(all_labels, dim=0)
    avg_metrics = calculate_classification_metrics(all_preds, all_labels)
    avg_metrics['loss'] = total_loss / len(dataloader)
    return avg_metrics


def validate(model, dataloader, criterion, device):
    model.eval(); total_loss = 0; all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs); loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits)); all_targets.append(labels)
    all_preds = torch.cat(all_preds, dim=0); all_targets = torch.cat(all_targets, dim=0)
    avg_metrics = calculate_classification_metrics(all_preds, all_targets)
    avg_metrics['loss'] = total_loss / len(dataloader)
    return avg_metrics, all_preds, all_targets


def plot_training_curves(history, save_dir, filename):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12)); fig.suptitle('2D ResNet Training Metrics', fontsize=16, fontweight='bold')
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss'); axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1'); axes[0, 1].plot(epochs, history['val_f1'], 'r-', label='Val F1')
    axes[0, 1].set_title('F1 Score (Micro)'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(epochs, history['train_acc'], 'b-', label='Train Acc'); axes[1, 0].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1, 0].set_title('Accuracy'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(epochs, history['train_precision'], 'b--', label='Train Precision'); axes[1, 1].plot(epochs, history['val_precision'], 'r--', label='Val Precision')
    axes[1, 1].plot(epochs, history['train_recall'], 'b:', label='Train Recall'); axes[1, 1].plot(epochs, history['val_recall'], 'r:', label='Val Recall')
    axes[1, 1].set_title('Precision and Recall'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout(); save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=300); plt.close(); print(f"\n✓ Training plots saved to: {save_path}")

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
    model = ResNetMultiLabel(backbone=BACKBONE, num_teeth=NUM_TEETH).to(device)
    if len(available_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 'train_acc': [], 'val_acc': [], 'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': []}

    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs..."); print("=" * 80)
    best_f1 = 0.0; best_val_preds = None; best_val_targets = None

    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_preds, val_targets = validate(model, val_loader, criterion, device)
        scheduler.step()

        for key in ['loss', 'f1', 'acc', 'precision', 'recall']:
            history[f'train_{key}'].append(train_metrics.get(key.replace('acc', 'accuracy'), 0))
            history[f'val_{key}'].append(val_metrics.get(key.replace('acc', 'accuracy'), 0))
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f}")

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict()}, Path(OUTPUT_DIR) / LAST_MODEL_FILENAME)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']; best_val_preds = val_preds; best_val_targets = val_targets
            torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict()}, Path(OUTPUT_DIR) / BEST_MODEL_FILENAME)
            print(f"        → Best F1 model saved (F1: {val_metrics['f1']:.4f}) to {BEST_MODEL_FILENAME}")

    print("\n" + "=" * 80 + f"\n[4/5] Training complete!\nBest validation F1 (micro): {best_f1:.4f}")

    print("\n" + "=" * 80 + "\n[4.5/5] Calculating per-tooth metrics on best model...")
    per_tooth_metrics, macro_metrics = calculate_per_tooth_metrics(best_val_preds, best_val_targets, num_teeth=NUM_TEETH)

    print("\n MACRO-AVERAGED METRICS (across all 32 teeth):"); print("-" * 80)
    print(f"  Macro Precision: {macro_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {macro_metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {macro_metrics['macro_f1']:.4f}")
    print(f"  Macro Accuracy:  {macro_metrics['macro_accuracy']:.4f}")

    print("\n PER-TOOTH METRICS (FDI Notation):"); print("-" * 80)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    for fdi_label, metrics in per_tooth_metrics.items():
        print(f"Tooth {fdi_label:<5} {metrics['precision']:>10.4f}   {metrics['recall']:>10.4f}   {metrics['f1']:>10.4f}   {metrics['accuracy']:>10.4f}   {metrics['support']:>8}")

    metrics_file = Path(OUTPUT_DIR) / METRICS_FILENAME
    with open(metrics_file, 'w') as f:
        serializable_per_tooth = {str(k): v for k, v in per_tooth_metrics.items()}
        json.dump({'macro_metrics': macro_metrics, 'per_tooth_metrics': serializable_per_tooth}, f, indent=2)
    print(f"\n✓ Detailed metrics saved to: {metrics_file}")

    print(f"\n[5/5] Generating training plots...")
    plot_training_curves(history, PLOT_DIR, PLOT_FILENAME)
    print("\n✓ All done!")

if __name__ == "__main__":
    main()