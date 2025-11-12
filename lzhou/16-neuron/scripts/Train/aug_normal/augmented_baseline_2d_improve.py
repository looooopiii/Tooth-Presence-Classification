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
from typing import Optional, Tuple
from PIL import Image
from torchvision import transforms
from torchvision import models
from torchvision.models import resnet18, resnet50

# ============= CONFIGURATION =============
DATA_SOURCES_CSV = [
    ("/home/user/lzhou/week10/output/augment_test/train_labels_augmented.csv", Path("/home/user/lzhou/week10/output/augment_test")),
    ("/home/user/lzhou/week10/output/augment_random/train_labels_random.csv", Path("/home/user/lzhou/week10/output/augment_random")),
]
ORIGINAL_DATA_PATHS = []

# Rendered 2D roots used to resolve *_top.png from CSV 'filename' (.obj relative paths)
RENDER_ROOT_RANDOM = Path("/home/user/lzhou/week10/output/render_aug_random")
RENDER_ROOT_TEST = Path("/home/user/lzhou/week10/output/render_aug_test")
OUTPUT_DIR = Path("/home/user/lzhou/week11/output/Train2D/aug_normal_17out")
PLOT_DIR = Path("/home/user/lzhou/week11/output/Train2D/aug_normal_17out/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- BCE Loss ---
BEST_MODEL_FILENAME = "bce_loss_full_dataset_best_17out.pth"
LAST_MODEL_FILENAME = "bce_loss_full_dataset_last_17out.pth"
PLOT_FILENAME = "bce_loss_full_dataset_metrics_17out.png"
METRICS_FILENAME = "bce_loss_full_dataset_detailed_metrics_17out.json"
# -----------------------------------

BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 8
LEARNING_RATE = 3e-4
# NUM_TEETH = 32
NUM_OUTPUTS = 17
NUM_TEETH_PER_JAW = 16
SEED = 41
IMG_SIZE = 320

# old 32-FDI mapping （used to read CSV labels）
VALID_FDI_LABELS_32 = sorted([18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28, 38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48])
FDI_TO_INDEX_32 = {fdi: i for i, fdi in enumerate(VALID_FDI_LABELS_32)}
INDEX_TO_FDI_32 = {i: fdi for fdi, i in FDI_TO_INDEX_32.items()}

# new 16-FDI to 16-idx mapping (for 17-dim output)
UPPER_FDI_TO_IDX16 = {
    18: 0, 17: 1, 16: 2, 15: 3, 14: 4, 13: 5, 12: 6, 11: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15
}
LOWER_FDI_TO_IDX16 = {
    48: 0, 47: 1, 46: 2, 45: 3, 44: 4, 43: 5, 42: 6, 41: 7,
    31: 8, 32: 9, 33: 10, 34: 11, 35: 12, 36: 13, 37: 14, 38: 15
}

available_gpus = []
device = None


def build_image_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

# for jaw detection and png path resolution
def resolve_png_path(rel_obj_path: str, primary_root: Path) -> Tuple[Optional[Path], Optional[str]]:
    if not isinstance(rel_obj_path, str):
        return None, None
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
            return cand, jaw
    return None, jaw 


# =========================================
# MODELS AND LOSS FUNCTION
# =========================================
class ToothClassificationModel(nn.Module):
    """2D classifier that mirrors the 3D PointNet head with ResNet features."""

    # --- MODIFIED: 更改为 num_outputs ---
    def __init__(self, num_outputs=17, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet50":
            net = models.resnet50(weights=None)
        else:
            net = models.resnet18(weights=None)

        feat_dim = net.fc.in_features
        self.feature_extractor = nn.Sequential(*list(net.children())[:-1])
        self.fc1 = nn.Linear(feat_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_outputs)

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
            # get png path and jaw
            png_path, jaw = resolve_png_path(row.get("filename", ""), self.root_dir)
            
            if png_path is None:
                missing_count += 1
                continue
            
            if jaw is None:
                continue

            try:
                # load 32-dim labels from CSV
                labels_32 = row[[str(fdi) for fdi in VALID_FDI_LABELS_32]].astype(np.float32).values
            except KeyError:
                continue
            
            if labels_32.shape[0] != len(VALID_FDI_LABELS_32):
                continue

            # transform to 17-dim labels
            labels_17 = self._convert_32_to_17(labels_32, jaw)

            # store 17-dim labels
            self.samples.append((png_path, labels_17))

        if missing_count > 0:
            print(f"[WARN] {missing_count} rows in {csv_file} skipped (no matching PNG).")

    # helper to convert 32-dim labels to 17-dim
    def _convert_32_to_17(self, labels_32: np.ndarray, jaw: str) -> np.ndarray:
        output_vector = np.zeros(NUM_OUTPUTS, dtype=np.float32)
        mapping = None
        jaw_label = 0.0

        if jaw == "upper":
            mapping = UPPER_FDI_TO_IDX16
            jaw_label = 0.0
        elif jaw == "lower":
            mapping = LOWER_FDI_TO_IDX16
            jaw_label = 1.0
        else:
            return output_vector

        # iterate over 32-dim labels and map to 16-dim
        for idx_32, label_val in enumerate(labels_32):
            fdi = INDEX_TO_FDI_32[idx_32]
            
            # map if in upper/lower mapping
            if fdi in mapping:
                idx_16 = mapping[fdi]
                output_vector[idx_16] = label_val

        output_vector[NUM_TEETH_PER_JAW] = jaw_label
        return output_vector

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
                    # determine jaw from filename
                    jaw = "lower" if "lower" in png_path.name.lower() else "upper"
                    self.samples.append((png_path, json_path, jaw))

    def __len__(self):
        return len(self.samples)

    # helper to load labels from JSON and convert to 17-dim
    def _load_labels_from_json(self, json_path, jaw):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # extract present teeth FDI labels
        present_teeth_fdi = {label for label in data.get("labels", []) if label != 0}
        
        # create 32-dim labels with missing teeth as 1.0
        labels_32_missing = np.ones(len(VALID_FDI_LABELS_32), dtype=np.float32)
        for fdi in present_teeth_fdi:
            if fdi in FDI_TO_INDEX_32:
                labels_32_missing[FDI_TO_INDEX_32[fdi]] = 0.0
        
        # use helper to convert to 17-dim
        output_vector = np.zeros(NUM_OUTPUTS, dtype=np.float32)
        mapping = UPPER_FDI_TO_IDX16 if jaw == "upper" else LOWER_FDI_TO_IDX16
        jaw_label = 0.0 if jaw == "upper" else 1.0

        for idx_32, label_val in enumerate(labels_32_missing):
            fdi = INDEX_TO_FDI_32[idx_32]
            if fdi in mapping:
                idx_16 = mapping[fdi]
                output_vector[idx_16] = label_val
        
        output_vector[NUM_TEETH_PER_JAW] = jaw_label
        return output_vector

    def __getitem__(self, idx):
        # get png path, json path, and jaw
        png_path, json_path, jaw = self.samples[idx]
        try:
            with Image.open(png_path) as img:
                image = img.convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (self.image_size, self.image_size))
        image = self.transform(image)
        # load and convert labels to 17-dim
        labels_17 = self._load_labels_from_json(json_path, jaw)
        return image, torch.from_numpy(labels_17).float()


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
        print(f" Free GPUs detected: {gpus}")
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
    # Binary classification metrics
    preds_binary = (predictions > 0.5).cpu().numpy()
    targets_np = targets.cpu().numpy()
    preds_flat, targets_flat = preds_binary.flatten(), targets_np.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(targets_flat, preds_flat, average="binary", zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1, "acc": accuracy_score(targets_flat, preds_flat)}

# jaw accuracy calculation
def calculate_jaw_accuracy(pred_jaw, target_jaw):
    pred = (pred_jaw > 0.5).cpu().numpy().astype(int)
    target = target_jaw.cpu().numpy().astype(int)
    return {'jaw_accuracy': accuracy_score(target, pred)}

# for each epoch training
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds_teeth, all_labels_teeth = [], []
    all_preds_jaw, all_labels_jaw = [], []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        labels_teeth = labels[:, :NUM_TEETH_PER_JAW]
        labels_jaw = labels[:, NUM_TEETH_PER_JAW]

        optimizer.zero_grad()
        logits = model(images) # Shape [B, 17]
        
        # calculate loss for all outputs
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # separate teeth and jaw logits
        logits_teeth = logits[:, :NUM_TEETH_PER_JAW]
        logits_jaw = logits[:, NUM_TEETH_PER_JAW]
        
        all_preds_teeth.append(torch.sigmoid(logits_teeth.detach()))
        all_labels_teeth.append(labels_teeth)
        all_preds_jaw.append(torch.sigmoid(logits_jaw.detach()))
        all_labels_jaw.append(labels_jaw)

    # calculate metrics separately
    metrics_teeth = calculate_classification_metrics(torch.cat(all_preds_teeth), torch.cat(all_labels_teeth))
    metrics_jaw = calculate_jaw_accuracy(torch.cat(all_preds_jaw), torch.cat(all_labels_jaw))
    
    metrics = {**metrics_teeth, **metrics_jaw} # merge metrics
    metrics["loss"] = total_loss / max(1, len(dataloader))
    return metrics

# for each epoch validation
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds_teeth, all_labels_teeth = [], []
    all_preds_jaw, all_labels_jaw = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            labels_teeth = labels[:, :NUM_TEETH_PER_JAW]
            labels_jaw = labels[:, NUM_TEETH_PER_JAW]

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            logits_teeth = logits[:, :NUM_TEETH_PER_JAW]
            logits_jaw = logits[:, NUM_TEETH_PER_JAW]

            all_preds_teeth.append(torch.sigmoid(logits_teeth))
            all_labels_teeth.append(labels_teeth)
            all_preds_jaw.append(torch.sigmoid(logits_jaw))
            all_labels_jaw.append(labels_jaw)

    all_preds_teeth_cat = torch.cat(all_preds_teeth)
    all_labels_teeth_cat = torch.cat(all_labels_teeth)
    all_preds_jaw_cat = torch.cat(all_preds_jaw)
    all_labels_jaw_cat = torch.cat(all_labels_jaw)

    metrics_teeth = calculate_classification_metrics(all_preds_teeth_cat, all_labels_teeth_cat)
    metrics_jaw = calculate_jaw_accuracy(all_preds_jaw_cat, all_labels_jaw_cat)

    metrics = {**metrics_teeth, **metrics_jaw}
    metrics["loss"] = total_loss / max(1, len(dataloader))

    # return all predictions and labels
    return metrics, all_preds_teeth_cat, all_labels_teeth_cat, all_preds_jaw_cat, all_labels_jaw_cat


# new plotting function
def plot_training_curves(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(2, 3, figsize=(22, 12)) # 2x3 grid
    fig.suptitle("2D Training Metrics (Baseline BCE Loss, 17 Out)", fontsize=16, fontweight="bold")

    # Loss
    ax[0, 0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax[0, 0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax[0, 0].set_title("Loss")
    ax[0, 0].legend(); ax[0, 0].grid(True, alpha=0.3)

    # F1 (Teeth)
    ax[0, 1].plot(epochs, history["train_f1"], "b-", label="Train F1 (Teeth)")
    ax[0, 1].plot(epochs, history["val_f1"], "r-", label="Val F1 (Teeth)")
    ax[0, 1].set_title("F1 Score (Micro - Teeth)")
    ax[0, 1].legend(); ax[0, 1].grid(True, alpha=0.3)

    # Jaw Accuracy (New)
    ax[0, 2].plot(epochs, history.get("train_jaw_accuracy", []), "b-", label="Train Jaw Acc")
    ax[0, 2].plot(epochs, history.get("val_jaw_accuracy", []), "r-", label="Val Jaw Acc")
    ax[0, 2].set_title("Jaw Classification Accuracy")
    ax[0, 2].legend(); ax[0, 2].grid(True, alpha=0.3)

    # Accuracy (Teeth)
    ax[1, 0].plot(epochs, history["train_acc"], "b-", label="Train Acc (Teeth)")
    ax[1, 0].plot(epochs, history["val_acc"], "r-", label="Val Acc (Teeth)")
    ax[1, 0].set_title("Accuracy (Teeth)")
    ax[1, 0].legend(); ax[1, 0].grid(True, alpha=0.3)

    # Precision (Teeth)
    ax[1, 1].plot(epochs, history["train_precision"], "b--", label="Train Precision")
    ax[1, 1].plot(epochs, history["val_precision"], "r--", label="Val Precision")
    ax[1, 1].set_title("Precision (Teeth)")
    ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)

    # Recall (Teeth)
    ax[1, 2].plot(epochs, history["train_recall"], "b:", label="Train Recall")
    ax[1, 2].plot(epochs, history["val_recall"], "r:", label="Val Recall")
    ax[1, 2].set_title("Recall (Teeth)")
    ax[1, 2].legend(); ax[1, 2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = Path(save_dir) / PLOT_FILENAME
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n✓ Plots saved: {save_path}")


# =========================================
# MAIN EXECUTION
# =========================================
def main():
    global available_gpus, device
    set_seed(SEED)

    print("\n[0/5] Setting up environment...")
    detected_gpus = get_free_gpus()
    available_gpus = filter_visible_gpus(detected_gpus)
    if torch.cuda.is_available() and available_gpus:
        device = torch.device(f"cuda:{available_gpus[0]}")
    else:
        device = torch.device("cpu")
    print(f"Primary device: {device}")

    print("\n[1/5] Loading and combining datasets...")
    # 17-dim label datasets
    datasets = []
    original_dataset = OriginalToothDataset(ORIGINAL_DATA_PATHS)
    if len(original_dataset) > 0:
        datasets.append(original_dataset)
    for csv_path, root_dir in DATA_SOURCES_CSV:
        if Path(csv_path).exists():
            datasets.append(CSVToothDataset(csv_path, root_dir))
        else:
            print(f"[WARN] CSV not found, skipped: {csv_path}")

    if len(datasets) == 0:
        print("[ERROR] No datasets found. Check CSV paths and renders.")
        return

    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    if len(full_dataset) < 2:
        print(f"[ERROR] Need at least 2 samples, found {len(full_dataset)}.")
        return
    print(f" Combined dataset loaded with {len(full_dataset)} samples (using 17-dim labels).")

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    print("\n[2/5] Initializing model...")
    # num_outputs changed to NUM_OUTPUTS
    model = ToothClassificationModel(num_outputs=NUM_OUTPUTS).to(device)
    if len(available_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)
        print(f" Model wrapped for {len(available_gpus)} GPUs.")

    # --- use BCEWithLogitsLoss ---
    criterion = nn.BCEWithLogitsLoss().to(device)
    print(" Using nn.BCEWithLogitsLoss for all 17 outputs.")
    # ----------------------------------------------
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # history to track metrics
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "train_acc": [], "val_acc": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": [],
        "train_jaw_accuracy": [], "val_jaw_accuracy": [],
    }
    best_f1 = 0.0

    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs...")
    print("=" * 80)
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_preds_teeth, val_targets_teeth, val_preds_jaw, val_labels_jaw = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # new history logging
        for key in ["loss", "f1", "acc", "precision", "recall"]:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])
        # add jaw accuracy
        history["train_jaw_accuracy"].append(train_metrics.get("jaw_accuracy", 0))
        history["val_jaw_accuracy"].append(val_metrics.get("jaw_accuracy", 0))

        # print training and validation metrics
        print(
            f"E {epoch + 1:2d}/{NUM_EPOCHS}|"
            f"Train L:{train_metrics['loss']:.4f}, F1:{train_metrics['f1']:.4f}, Jaw:{train_metrics.get('jaw_accuracy', 0):.4f}|"
            f"Val L:{val_metrics['loss']:.4f}, F1:{val_metrics['f1']:.4f}, Jaw:{val_metrics.get('jaw_accuracy', 0):.4f}"
        )

        model_to_save = model.module if hasattr(model, "module") else model
        torch.save({"epoch": epoch, "model_state_dict": model_to_save.state_dict()}, OUTPUT_DIR / LAST_MODEL_FILENAME)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            # best_preds, best_targets = val_preds_teeth, val_targets_teeth
            torch.save({"epoch": epoch, "model_state_dict": model_to_save.state_dict()}, OUTPUT_DIR / BEST_MODEL_FILENAME)
            print(f"  →  Best F1 model saved (F1: {val_metrics['f1']:.4f})")

    print("\n" + "=" * 80 + f"\n[4/5] Training complete! Best F1: {best_f1:.4f}")

    print(f"\n[5/5] Generating plots...")
    plot_training_curves(history, PLOT_DIR)
    print("\n All done!")


if __name__ == "__main__":
    main()