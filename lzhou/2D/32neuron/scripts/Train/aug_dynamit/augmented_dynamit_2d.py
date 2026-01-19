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
from sklearn.model_selection import train_test_split
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
DATA_SOURCES_CSV = [
    ("/home/user/lzhou/week10/output/augment_test/train_labels_augmented.csv", Path("/home/user/lzhou/week10/output/augment_test")),
    ("/home/user/lzhou/week10/output/augment_random/train_labels_random.csv", Path("/home/user/lzhou/week10/output/augment_random")),
]
ORIGINAL_DATA_PATHS = []

# Rendered 2D roots used to resolve *_top.png from CSV 'filename' (.obj relative paths)
RENDER_ROOT_RANDOM = Path("/home/user/lzhou/week15/render_output/render_aug_random")
RENDER_ROOT_TEST = Path("/home/user/lzhou/week15/render_output/render_aug_test")

OUTPUT_DIR = Path("/home/user/lzhou/week15-32/output/Train2D/aug_dynamit")
PLOT_DIR = Path("/home/user/lzhou/week15-32/output/Train2D/aug_dynamit/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILENAME = "dynamit_loss_full_dataset_best.pth"
LAST_MODEL_FILENAME = "dynamit_loss_full_dataset_last.pth"
PLOT_FILENAME = "dynamit_loss_full_dataset_metrics.png"
METRICS_FILENAME = "dynamit_loss_full_dataset_detailed_metrics.json"

BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 8
LEARNING_RATE = 3e-4
NUM_TEETH = 32
SEED = 41
IMG_SIZE = 320

# Early Stopping - match Aug Baseline
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 1e-4

VALID_FDI_LABELS = sorted([18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28, 38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48])
FDI_TO_INDEX = {fdi: i for i, fdi in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi for fdi, i in FDI_TO_INDEX.items()}

available_gpus = []
device = None


def build_image_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def resolve_png_path(rel_obj_path: str, primary_root: Path) -> Optional[Path]:
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

    candidate_roots = []
    if primary_root is not None:
        candidate_roots.append(primary_root)
    candidate_roots.extend([RENDER_ROOT_RANDOM, RENDER_ROOT_TEST])

    candidates = []
    for root in candidate_roots:
        root = Path(root)
        if jaw and case_dir:
            candidates.append(root / jaw / case_dir / png_name)
        candidates.append(root / png_name)

    for cand in candidates:
        if cand.exists():
            return cand
    return None


# =========================================
# MODELS AND LOSS FUNCTION
# =========================================
class Dynamit_Loss(nn.Module):
    """Numerically stable Dynamit Loss mirroring the 3D training setup."""

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, predictions, targets):
        S_pos = (targets == 1).sum().float()
        S_neg = (targets == 0).sum().float()
        epsilon = 1e-8

        pos_coeff_raw = S_neg / (S_pos + epsilon)
        neg_coeff_raw = S_pos / (S_neg + epsilon)

        pos_coeff = torch.clamp(pos_coeff_raw, max=1.0).detach()
        neg_coeff = torch.clamp(neg_coeff_raw, max=1.0).detach()

        weights = torch.where(targets == 1, pos_coeff, neg_coeff)
        return F.binary_cross_entropy_with_logits(predictions, targets, weight=weights)


class ToothClassificationModel(nn.Module):
    """2D classifier that mirrors the 3D PointNet head with ResNet features."""

    def __init__(self, num_teeth=32, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights=None)
        else:
            net = resnet18(weights=None)

        feat_dim = net.fc.in_features
        self.feature_extractor = nn.Sequential(*list(net.children())[:-1])
        self.fc1 = nn.Linear(feat_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_teeth)

    def forward(self, x):
        features = self.feature_extractor(x).flatten(1)
        x = self.dropout1(F.relu(self.bn1(self.fc1(features))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


# =========================================
# DATASET CLASSES
# =========================================
class CSVToothDataset(Dataset):
    def __init__(self, csv_file, root_dir, image_size=IMG_SIZE):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = build_image_transform()
        self.samples = []
        df = pd.read_csv(csv_file)

        missing_count = 0
        for _, row in df.iterrows():
            png_path = resolve_png_path(row.get("filename", ""), self.root_dir)
            if png_path is None:
                missing_count += 1
                continue

            try:
                labels = row[[str(fdi) for fdi in VALID_FDI_LABELS]].astype(np.float32).values
            except KeyError:
                continue

            if labels.shape[0] != NUM_TEETH:
                continue
            self.samples.append((png_path, labels))

        if missing_count > 0:
            print(f"[WARN] {missing_count} rows in {csv_file} skipped (no matching PNG).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        png_path, labels = self.samples[idx]
        try:
            with Image.open(png_path) as img:
                image = img.convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (self.image_size, self.image_size))
        image = self.transform(image)
        return image, torch.from_numpy(labels).float()


class OriginalToothDataset(Dataset):
    def __init__(self, data_paths, image_size=IMG_SIZE):
        self.image_size = image_size
        self.transform = build_image_transform()
        self.samples = []
        for data_path_str in data_paths:
            path = Path(data_path_str)
            if not path.exists():
                continue
            for png_path in sorted(path.rglob("*_top.png")):
                json_path = png_path.with_suffix(".json")
                if json_path.exists():
                    self.samples.append((png_path, json_path))

    def __len__(self):
        return len(self.samples)

    def _load_labels_from_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        present_teeth_fdi = {label for label in data.get("labels", []) if label != 0}
        labels = np.ones(NUM_TEETH, dtype=np.float32)
        for fdi in present_teeth_fdi:
            if fdi in FDI_TO_INDEX:
                labels[FDI_TO_INDEX[fdi]] = 0.0
        return labels

    def __getitem__(self, idx):
        png_path, json_path = self.samples[idx]
        try:
            with Image.open(png_path) as img:
                image = img.convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (self.image_size, self.image_size))
        image = self.transform(image)
        labels = self._load_labels_from_json(json_path)
        return image, torch.from_numpy(labels).float()


# =========================================
# HELPER AND TRAINING FUNCTIONS
# =========================================
def get_free_gpus(threshold_mb=1000, max_gpus=2):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        gpus = []
        for line in lines:
            idx_str, mem_str = line.split(", ")
            if int(mem_str) < threshold_mb:
                gpus.append(int(idx_str))
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


def filter_visible_gpus(candidate_ids):
    if not torch.cuda.is_available():
        print("CUDA not available; switching to CPU.")
        return []
    visible = torch.cuda.device_count()
    if visible == 0:
        print("CUDA reports zero visible devices; switching to CPU.")
        return []
    filtered = [idx for idx in candidate_ids if 0 <= idx < visible]
    if not filtered:
        print(f"[WARN] Requested GPU indices {candidate_ids} not visible (visible count: {visible}). Using GPU 0.")
        filtered = [0]
    return filtered


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_classification_metrics(predictions, targets):
    """Calculate both micro and macro metrics"""
    preds_binary = (predictions > 0.5).cpu().numpy()
    targets_np = targets.cpu().numpy()
    preds_flat, targets_flat = preds_binary.flatten(), targets_np.flatten()
    
    # Micro-averaged metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        targets_flat, preds_flat, average="binary", zero_division=0
    )
    acc_micro = accuracy_score(targets_flat, preds_flat)
    
    # Macro-averaged metrics (per-tooth then average)
    precisions, recalls, f1s, accs = [], [], [], []
    for i in range(NUM_TEETH):
        p, r, f1, _ = precision_recall_fscore_support(
            targets_np[:, i], preds_binary[:, i], average="binary", zero_division=0
        )
        acc = accuracy_score(targets_np[:, i], preds_binary[:, i])
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        accs.append(acc)
    
    return {
        "precision": precision_micro,
        "recall": recall_micro,
        "f1": f1_micro,
        "acc": acc_micro,
        "macro_precision": np.mean(precisions),
        "macro_recall": np.mean(recalls),
        "macro_f1": np.mean(f1s),
        "macro_acc": np.mean(accs),
    }


def calculate_per_tooth_metrics(predictions, targets):
    preds_binary = (predictions > 0.5).cpu().numpy()
    targets_np = targets.cpu().numpy()
    metrics = OrderedDict()
    for i in range(NUM_TEETH):
        fdi = INDEX_TO_FDI[i]
        p, r, f1, _ = precision_recall_fscore_support(targets_np[:, i], preds_binary[:, i], average="binary", zero_division=0)
        acc = accuracy_score(targets_np[:, i], preds_binary[:, i])
        metrics[fdi] = {"precision": p, "recall": r, "f1": f1, "accuracy": acc, "support": int(targets_np[:, i].sum())}
    macro_metrics = {
        "macro_precision": np.mean([m["precision"] for m in metrics.values()]),
        "macro_recall": np.mean([m["recall"] for m in metrics.values()]),
        "macro_f1": np.mean([m["f1"] for m in metrics.values()]),
        "macro_accuracy": np.mean([m["accuracy"] for m in metrics.values()]),
    }
    return metrics, macro_metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits.detach()))
        all_labels.append(labels)
    metrics = calculate_classification_metrics(torch.cat(all_preds), torch.cat(all_labels))
    metrics["loss"] = total_loss / max(1, len(dataloader))
    return metrics


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits))
            all_targets.append(labels)
    metrics = calculate_classification_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["loss"] = total_loss / max(1, len(dataloader))
    return metrics, torch.cat(all_preds), torch.cat(all_targets)


def plot_training_curves(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle("2D Training Metrics (Full Dataset - Aug Dynamit)", fontsize=16, fontweight="bold")

    # Loss
    ax[0, 0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax[0, 0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax[0, 0].set_title("Loss", fontsize=14, fontweight="bold")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].legend()
    ax[0, 0].grid(True, alpha=0.3)

    # Micro F1
    ax[0, 1].plot(epochs, history["train_f1"], "b-", label="Train Micro F1", linewidth=2)
    ax[0, 1].plot(epochs, history["val_f1"], "r-", label="Val Micro F1", linewidth=2)
    ax[0, 1].set_title("Micro F1 Score", fontsize=14, fontweight="bold")
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel("F1")
    ax[0, 1].legend()
    ax[0, 1].grid(True, alpha=0.3)

    # Macro F1
    ax[1, 0].plot(epochs, history["train_macro_f1"], "b-", label="Train Macro F1", linewidth=2)
    ax[1, 0].plot(epochs, history["val_macro_f1"], "r-", label="Val Macro F1", linewidth=2)
    ax[1, 0].set_title("Macro F1 Score", fontsize=14, fontweight="bold")
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].set_ylabel("Macro F1")
    ax[1, 0].legend()
    ax[1, 0].grid(True, alpha=0.3)

    # Accuracy (Micro & Macro)
    ax[1, 1].plot(epochs, history["train_acc"], "b-", label="Train Micro Acc", linewidth=2)
    ax[1, 1].plot(epochs, history["val_acc"], "r-", label="Val Micro Acc", linewidth=2)
    ax[1, 1].plot(epochs, history["train_macro_acc"], "b--", label="Train Macro Acc", linewidth=2, alpha=0.7)
    ax[1, 1].plot(epochs, history["val_macro_acc"], "r--", label="Val Macro Acc", linewidth=2, alpha=0.7)
    ax[1, 1].set_title("Accuracy", fontsize=14, fontweight="bold")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel("Accuracy")
    ax[1, 1].legend()
    ax[1, 1].grid(True, alpha=0.3)

    # Precision (Micro & Macro)
    ax[2, 0].plot(epochs, history["train_precision"], "b-", label="Train Micro Prec", linewidth=2)
    ax[2, 0].plot(epochs, history["val_precision"], "r-", label="Val Micro Prec", linewidth=2)
    ax[2, 0].plot(epochs, history["train_macro_precision"], "b--", label="Train Macro Prec", linewidth=2, alpha=0.7)
    ax[2, 0].plot(epochs, history["val_macro_precision"], "r--", label="Val Macro Prec", linewidth=2, alpha=0.7)
    ax[2, 0].set_title("Precision", fontsize=14, fontweight="bold")
    ax[2, 0].set_xlabel("Epoch")
    ax[2, 0].set_ylabel("Precision")
    ax[2, 0].legend()
    ax[2, 0].grid(True, alpha=0.3)

    # Recall (Micro & Macro)
    ax[2, 1].plot(epochs, history["train_recall"], "b-", label="Train Micro Rec", linewidth=2)
    ax[2, 1].plot(epochs, history["val_recall"], "r-", label="Val Micro Rec", linewidth=2)
    ax[2, 1].plot(epochs, history["train_macro_recall"], "b--", label="Train Macro Rec", linewidth=2, alpha=0.7)
    ax[2, 1].plot(epochs, history["val_macro_recall"], "r--", label="Val Macro Rec", linewidth=2, alpha=0.7)
    ax[2, 1].set_title("Recall", fontsize=14, fontweight="bold")
    ax[2, 1].set_xlabel("Epoch")
    ax[2, 1].set_ylabel("Recall")
    ax[2, 1].legend()
    ax[2, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = Path(save_dir) / PLOT_FILENAME
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n✓ Plots saved: {save_path}")


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


def stratified_split_dataset(dataset, train_ratio=0.9, seed=42):
    """Stratified split to maintain tooth distribution - matches Aug Baseline"""
    # Extract all labels
    all_labels = []
    all_indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label.numpy())
        all_indices.append(i)
    all_labels = np.array(all_labels)
    
    # Calculate missing teeth count for each sample (for stratification)
    missing_counts = all_labels.sum(axis=1).astype(int)
    
    # Bin the counts to avoid too many strata with few samples
    def bin_missing_count(count):
        if count == 0:
            return 0
        elif count <= 2:
            return 1
        elif count <= 5:
            return 2
        else:
            return 3
    
    stratify_bins = np.array([bin_missing_count(c) for c in missing_counts])
    
    # Print stratification info
    unique_bins, bin_counts = np.unique(stratify_bins, return_counts=True)
    bin_names = {0: "0 missing", 1: "1-2 missing", 2: "3-5 missing", 3: "6+ missing"}
    print(f"  Stratification distribution:")
    for bin_id, count in zip(unique_bins, bin_counts):
        print(f"    {bin_names[bin_id]}: {count} samples ({count/len(dataset)*100:.1f}%)")
    
    # Perform stratified split
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=1.0 - train_ratio,
        stratify=stratify_bins,
        random_state=seed
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Verify stratification worked
    train_bins = stratify_bins[train_indices]
    val_bins = stratify_bins[val_indices]
    print(f"\n  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"  Train distribution:")
    unique_bins, bin_counts = np.unique(train_bins, return_counts=True)
    for bin_id, count in zip(unique_bins, bin_counts):
        print(f"    {bin_names[bin_id]}: {count} samples ({count/len(train_dataset)*100:.1f}%)")
    print(f"  Val distribution:")
    unique_bins, bin_counts = np.unique(val_bins, return_counts=True)
    for bin_id, count in zip(unique_bins, bin_counts):
        print(f"    {bin_names[bin_id]}: {count} samples ({count/len(val_dataset)*100:.1f}%)")
    
    return train_dataset, val_dataset


# =========================================
# MAIN EXECUTION
# =========================================
def main():
    global available_gpus, device
    set_seed(SEED)

    print("\n" + "="*80)
    print("AUG DYNAMIT TRAINING - FIXED VERSION")
    print("="*80)
    
    print("\n[0/5] Setting up environment...")
    detected_gpus = get_free_gpus()
    available_gpus = filter_visible_gpus(detected_gpus)
    if torch.cuda.is_available() and available_gpus:
        device = torch.device(f"cuda:{available_gpus[0]}")
    else:
        device = torch.device("cpu")
    print(f"✓ Primary device: {device}")

    print("\n[1/5] Loading and combining datasets...")
    datasets = []
    original_dataset = OriginalToothDataset(ORIGINAL_DATA_PATHS)
    if len(original_dataset) > 0:
        datasets.append(original_dataset)
        print(f"✓ Loaded {len(original_dataset)} original samples")
    
    for csv_path, root_dir in DATA_SOURCES_CSV:
        if Path(csv_path).exists():
            ds = CSVToothDataset(csv_path, root_dir)
            datasets.append(ds)
            print(f"✓ Loaded {len(ds)} samples from {Path(csv_path).name}")
        else:
            print(f"[WARN] CSV not found, skipped: {csv_path}")

    if len(datasets) == 0:
        print("[ERROR] No datasets found. Check CSV paths and renders.")
        return

    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    if len(full_dataset) < 2:
        print(f"[ERROR] Need at least 2 samples, found {len(full_dataset)}.")
        return
    print(f"✓ Combined dataset: {len(full_dataset)} samples")

    print("\n[1.5/5] Performing stratified train/val split...")
    train_dataset, val_dataset = stratified_split_dataset(full_dataset, train_ratio=0.9, seed=SEED)
    print(f"✓ Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    print("\n[2/5] Initializing model...")
    model = ToothClassificationModel(num_teeth=NUM_TEETH).to(device)
    if len(available_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)
        print(f"✓ Model wrapped for {len(available_gpus)} GPUs")

    criterion = Dynamit_Loss(device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "train_macro_f1": [], "val_macro_f1": [],
        "train_acc": [], "val_acc": [],
        "train_macro_acc": [], "val_macro_acc": [],
        "train_precision": [], "val_precision": [],
        "train_macro_precision": [], "val_macro_precision": [],
        "train_recall": [], "val_recall": [],
        "train_macro_recall": [], "val_macro_recall": [],
    }
    best_macro_f1 = 0.0
    best_preds = None
    best_targets = None
    patience_counter = 0

    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs (Early Stopping: patience={EARLY_STOPPING_PATIENCE})...")
    print("="*80)
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # Store all metrics
        for key in ["loss", "f1", "macro_f1", "acc", "macro_acc", "precision", "macro_precision", "recall", "macro_recall"]:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])

        print(
            f"E {epoch + 1:2d}/{NUM_EPOCHS} | "
            f"Train L:{train_metrics['loss']:.4f}, F1:{train_metrics['f1']:.4f} | "
            f"Val L:{val_metrics['loss']:.4f}, F1 (Micro):{val_metrics['f1']:.4f}, F1 (Macro):{val_metrics['macro_f1']:.4f}"
        )

        # Save last model
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save({"epoch": epoch, "model_state_dict": model_to_save.state_dict()}, OUTPUT_DIR / LAST_MODEL_FILENAME)

        # Save best model based on MACRO F1 (better for balanced per-tooth performance)
        if val_metrics["macro_f1"] > best_macro_f1 + EARLY_STOPPING_MIN_DELTA:
            best_macro_f1 = val_metrics["macro_f1"]
            best_preds, best_targets = val_preds, val_targets
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model_to_save.state_dict()}, OUTPUT_DIR / BEST_MODEL_FILENAME)
            print(f"  → Best model saved (Micro F1: {val_metrics['f1']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n✓ Early stopping triggered after {epoch+1} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    print("\n" + "="*80 + f"\n[4/5] Training complete!")
    print(f"Best validation Macro F1: {best_macro_f1:.4f}")
    print(f"Training stopped at epoch: {len(history['train_loss'])}/{NUM_EPOCHS}")
    print("="*80)

    if best_preds is None or best_targets is None:
        print("[WARN] No validation predictions captured; skipping metric export.")
    else:
        print("\n[4.5/5] Calculating detailed metrics...")
        per_tooth_metrics, macro_metrics = calculate_per_tooth_metrics(best_preds, best_targets)
        
        print("\n" + "="*80)
        print("MACRO-AVERAGED METRICS (Per-Tooth then Averaged)")
        print("="*80)
        print(f"  Macro Precision: {macro_metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:    {macro_metrics['macro_recall']:.4f}")
        print(f"  Macro F1:        {macro_metrics['macro_f1']:.4f}")
        print(f"  Macro Accuracy:  {macro_metrics['macro_accuracy']:.4f}")
        
        print("\n" + "="*80)
        print("PER-TOOTH METRICS")
        print("="*80)
        print(f"{'FDI':<6} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10} {'Support':<8}")
        print("-"*80)
        for fdi, mets in per_tooth_metrics.items():
            print(
                f"{fdi:<6} {mets['precision']:>10.4f} {mets['recall']:>10.4f} "
                f"{mets['f1']:>10.4f} {mets['accuracy']:>10.4f} {mets['support']:>8}"
            )
        
        metrics_path = OUTPUT_DIR / METRICS_FILENAME
        with open(metrics_path, "w") as f:
            json.dump({
                "macro_metrics": macro_metrics,
                "per_tooth_metrics": {str(k): v for k, v in per_tooth_metrics.items()}
            }, f, indent=2)
        print(f"\n✓ Metrics saved: {metrics_path}")

    print(f"\n[5/5] Generating plots...")
    plot_training_curves(history, PLOT_DIR)
    print("\n" + "="*80)
    print("✓ ALL DONE!")
    print("="*80)


if __name__ == "__main__":
    main()