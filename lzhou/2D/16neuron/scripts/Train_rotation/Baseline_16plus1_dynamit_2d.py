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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet50

# ============= CONFIGURATION =============
DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]
IMG_ROOTS = {
    "lower": "/home/user/lzhou/week15/render_output/train/lowerjaw",
    "upper": "/home/user/lzhou/week15/render_output/train/upperjaw"
}
IMAGE_SUFFIX = "_top.png"
CACHE_IMAGES = True

OUTPUT_DIR = "/home/user/lzhou/week16-17/output/Train_rotation/16plus1teeth_dynamit"
PLOT_DIR = "/home/user/lzhou/week16-17/output/Train_rotation/16plus1teeth_dynamit/plots"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILENAME = "dynamit_best_2d_16plus1teeth.pth"
METRICS_FILENAME = "detailed_metrics_dynamit_16plus1teeth.json"
PLOT_FILENAME = "training_metrics_dynamit_16plus1teeth.png"

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
NUM_EPOCHS = 35
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
IMG_SIZE = 256
SEED = 41
BACKBONE = "resnet18"
DROPOUT_RATE = 0.5
NUM_WORKERS = 4

# 16+1 Config
NUM_TEETH_PER_JAW = 16
NUM_JAW_CLASSES = 1
TOTAL_OUTPUTS = NUM_TEETH_PER_JAW + NUM_JAW_CLASSES

# FDI mappings
UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

UPPER_TO_IDX = {fdi: i for i, fdi in enumerate(UPPER_FDI)}
LOWER_TO_IDX = {fdi: i for i, fdi in enumerate(LOWER_FDI)}

# =========================================
#  LOSS FUNCTION
# =========================================

class Dynamit_Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, predictions, targets):
        teeth_targets = targets[:, :NUM_TEETH_PER_JAW]
        S_pos = (teeth_targets == 1).sum().float()
        S_neg = (teeth_targets == 0).sum().float()

        if S_pos > 0 and S_neg > 0:
            pos_coeff_val = min(1.0, (S_neg / S_pos).item())
            neg_coeff_val = min(1.0, (S_pos / S_neg).item())
        elif S_pos == 0:
            pos_coeff_val = 0.1
            neg_coeff_val = 1.0
        elif S_neg == 0:
            pos_coeff_val = 1.0
            neg_coeff_val = 0.1
        else:
            pos_coeff_val = 1.0
            neg_coeff_val = 1.0

        pos_coeff = torch.tensor(pos_coeff_val, device=self.device)
        neg_coeff = torch.tensor(neg_coeff_val, device=self.device)

        weights = torch.where(teeth_targets == 1, pos_coeff, neg_coeff)  # [B,16]
        weights = torch.cat([weights, torch.ones_like(targets[:,16:17])], dim=1)
        return F.binary_cross_entropy_with_logits(predictions, targets, weight=weights)

# =========================================
#  UTILS
# =========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_by_case_id(dataset, train_ratio, seed):
    case_to_indices = {}
    for idx, sample in enumerate(dataset.data_cache):
        case_id = sample.get("case_id")
        if case_id is None:
            raise ValueError("Missing case_id in dataset cache for group split.")
        case_to_indices.setdefault(case_id, []).append(idx)

    case_ids = sorted(case_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(case_ids)

    train_case_count = int(len(case_ids) * train_ratio)
    train_cases = set(case_ids[:train_case_count])

    train_indices, val_indices = [], []
    for case_id in case_ids:
        if case_id in train_cases:
            train_indices.extend(case_to_indices[case_id])
        else:
            val_indices.extend(case_to_indices[case_id])

    return train_indices, val_indices

# =========================================
#  DATASET
# =========================================

class Tooth2DDataset(Dataset):
    def __init__(self, data_paths, img_roots, image_suffix="_top.png", transform=None, cache_images=False):
        self.transform = transform
        self.image_suffix = image_suffix
        self.cache_images = cache_images
        self.data_cache = []

        img_root_map = {jaw: Path(path) for jaw, path in img_roots.items()}

        print("[Info] Scanning and caching labels...")
        for data_path_str in data_paths:
            data_path = Path(data_path_str)
            if not data_path.exists():
                continue

            case_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
            for case_dir in tqdm(case_dirs, desc=f"Loading {data_path.name}"):
                case_id = case_dir.name
                for jaw_type in ["upper", "lower"]:
                    json_file = case_dir / f"{case_id}_{jaw_type}.json"
                    img_root = img_root_map.get(jaw_type)
                    if img_root is None or not img_root.exists():
                        continue
                    img_file = img_root / f"{case_id}_{jaw_type}{self.image_suffix}"

                    if json_file.exists() and img_file.exists():
                        try:
                            labels = self.load_labels(str(json_file))
                            targets = self.create_16plus1_targets(labels, jaw_type)
                            img_data = None
                            if self.cache_images:
                                with Image.open(img_file) as img:
                                    img_data = img.convert("RGB").copy()
                            self.data_cache.append({
                                "img": img_data if self.cache_images else str(img_file),
                                "targets": targets,
                                "case_id": case_id,
                                "jaw_type": jaw_type
                            })
                        except Exception as e:
                            print(f"Skipping {case_id} ({jaw_type}): {e}")

        print(f"[Info] Cached {len(self.data_cache)} samples.")

    def __len__(self):
        return len(self.data_cache)

    def load_labels(self, json_path):
        with open(json_path, "r") as f:
            return set(json.load(f).get("labels", []))

    def create_16plus1_targets(self, vertex_labels_set, jaw_type):
        is_lower = 1.0 if jaw_type == "lower" else 0.0
        tooth_presence = np.zeros(NUM_TEETH_PER_JAW, dtype=np.float32)
        mapping = LOWER_TO_IDX if jaw_type == "lower" else UPPER_TO_IDX

        for fdi_label in vertex_labels_set:
            if fdi_label in mapping:
                tooth_presence[mapping[fdi_label]] = 1.0

        tooth_missing = 1.0 - tooth_presence
        return np.concatenate([tooth_missing, [is_lower]]).astype(np.float32)

    def __getitem__(self, idx):
        sample = self.data_cache[idx]
        if self.cache_images:
            img = sample["img"].copy()
        else:
            img = Image.open(sample["img"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        targets = torch.from_numpy(sample["targets"]).float()
        return img, targets


class TransformWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, targets = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, targets


def plot_training_curves(history, save_dir, filename):
    if not history["train_loss"]:
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("ResNet 2D Training - 16+1 Neurons (Dynamic Loss)", fontsize=14, fontweight="bold")

    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["train_f1"], "b-", label="Train F1")
    axes[0, 1].plot(epochs, history["val_f1"], "r-", label="Val F1")
    axes[0, 1].set_title("Teeth F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("F1")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["train_jaw_acc"], "b-", label="Train Jaw Acc")
    axes[1, 0].plot(epochs, history["val_jaw_acc"], "r-", label="Val Jaw Acc")
    axes[1, 0].set_title("Jaw Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["train_f1"], "b--", label="Train F1")
    axes[1, 1].plot(epochs, history["val_f1"], "r--", label="Val F1")
    axes[1, 1].plot(epochs, history["train_jaw_acc"], "b:", label="Train Jaw Acc")
    axes[1, 1].plot(epochs, history["val_jaw_acc"], "r:", label="Val Jaw Acc")
    axes[1, 1].set_title("F1 & Jaw Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training plots saved to: {save_path}")

# =========================================
#  MODEL
# =========================================

class ResNet16Plus1(nn.Module):
    def __init__(self, backbone="resnet18", dropout_rate=0.5):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights="IMAGENET1K_V2")
        else:
            net = resnet18(weights="IMAGENET1K_V1")

        in_feats = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

        self.fc_shared = nn.Linear(in_feats, 512)
        self.bn_shared = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(dropout_rate)

        self.fc_teeth = nn.Linear(512, 256)
        self.bn_teeth = nn.BatchNorm1d(256)
        self.out_teeth = nn.Linear(256, NUM_TEETH_PER_JAW)

        self.fc_jaw = nn.Linear(512, 128)
        self.bn_jaw = nn.BatchNorm1d(128)
        self.out_jaw = nn.Linear(128, 1)

    def forward(self, x):
        features = self.backbone(x)
        shared = self.drop(F.relu(self.bn_shared(self.fc_shared(features))))
        teeth = self.out_teeth(self.drop(F.relu(self.bn_teeth(self.fc_teeth(shared)))))
        jaw = self.out_jaw(self.drop(F.relu(self.bn_jaw(self.fc_jaw(shared)))))
        return torch.cat([teeth, jaw], dim=1)

# =========================================
#  METRICS & LOOP
# =========================================

def calculate_metrics(logits, targets):
    pred = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy().astype(int)
    tgt = targets.float().cpu().numpy().astype(int)
    _, _, f1, _ = precision_recall_fscore_support(
        tgt[:, :NUM_TEETH_PER_JAW].flatten(),
        pred[:, :NUM_TEETH_PER_JAW].flatten(),
        average="binary",
        zero_division=0
    )
    jaw_acc = accuracy_score(tgt[:, NUM_TEETH_PER_JAW], pred[:, NUM_TEETH_PER_JAW])
    return {"f1": f1, "jaw_acc": jaw_acc}

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, all_logits, all_targets = 0.0, [], []

    if is_train:
        optimizer.zero_grad()

    amp_enabled = device.type == "cuda"

    with torch.set_grad_enabled(is_train):
        for imgs, labels in tqdm(loader, leave=False, desc="Train" if is_train else "Val"):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled and is_train):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if is_train:
                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(labels.detach().float().cpu())

    if not all_logits:
        return {"loss": 0.0, "f1": 0.0, "jaw_acc": 0.0}

    full_logits = torch.cat(all_logits)
    full_targets = torch.cat(all_targets)
    metrics = calculate_metrics(full_logits, full_targets)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def print_final_report(model, loader, device):
    print("\n[5/5] Calculating jaw-aware detailed metrics on best model...")
    model.eval()
    all_preds_list, all_targets_list = [], []
    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Final Eval", leave=False):
            imgs = imgs.to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(imgs)
            all_preds_list.append((torch.sigmoid(logits.float()) > 0.5).cpu().numpy().astype(int))
            all_targets_list.append(labels.cpu().numpy().astype(int))

    all_preds = np.concatenate(all_preds_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)

    valid_precs, valid_recs, valid_f1s, valid_accs = [], [], [], []

    print("\n" + "=" * 90)
    print("PER-TOOTH METRICS (Support > 0)")
    print("-" * 90)
    print(f"{'FDI':<6} {'Prec':<10} {'Recall':<10} {'F1':<10} {'Acc':<10} {'Supp':<6} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5}")
    print("-" * 90)

    for section_name, fdi_list, jaw_label in [("UPPER", UPPER_FDI, 0.0), ("LOWER", LOWER_FDI, 1.0)]:
        print(f"\n{section_name} JAW:")
        jaw_mask = (all_targets[:, NUM_TEETH_PER_JAW] == jaw_label)
        if jaw_mask.sum() == 0:
            continue

        for local_idx, fdi in enumerate(fdi_list):
            t_p = all_preds[jaw_mask, local_idx]
            t_t = all_targets[jaw_mask, local_idx]
            support = int(np.sum(t_t == 1))
            tn, fp, fn, tp = confusion_matrix(t_t, t_p, labels=[0, 1]).ravel()
            acc = accuracy_score(t_t, t_p)

            if support > 0:
                p, r, f, _ = precision_recall_fscore_support(
                    t_t, t_p, average="binary", zero_division=0
                )
                valid_precs.append(p)
                valid_recs.append(r)
                valid_f1s.append(f)
                valid_accs.append(acc)
                p_s, r_s, f_s = f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"
            else:
                p_s, r_s, f_s = "N/A", "N/A", "N/A"

            print(f"{fdi:<6} {p_s:<10} {r_s:<10} {f_s:<10} {acc:<10.4f} {support:<6} {tp:<5} {fp:<5} {fn:<5} {tn:<5}")

    m_prec = float(np.mean(valid_precs)) if valid_precs else 0.0
    m_rec = float(np.mean(valid_recs)) if valid_recs else 0.0
    m_f1 = float(np.mean(valid_f1s)) if valid_f1s else 0.0
    m_acc = float(np.mean(valid_accs)) if valid_accs else 0.0
    jaw_acc = accuracy_score(all_targets[:, NUM_TEETH_PER_JAW], all_preds[:, NUM_TEETH_PER_JAW])

    print("\n" + "=" * 90)
    print("OVERALL SUMMARY (Support > 0)")
    print("=" * 90)
    print(f"Overall Precision:     {m_prec:.4f}")
    print(f"Overall Recall:        {m_rec:.4f}")
    print(f"Overall F1 Score:      {m_f1:.4f}")
    print(f"Overall Accuracy:      {m_acc:.4f}")
    print("-" * 90)
    print(f"Jaw Classification Accuracy: {jaw_acc:.4f}")
    print("=" * 90 + "\n")

    metrics_file = Path(OUTPUT_DIR) / METRICS_FILENAME
    with open(metrics_file, "w") as f:
        json.dump({
            "overall_precision": m_prec,
            "overall_recall": m_rec,
            "overall_f1": m_f1,
            "overall_accuracy": m_acc,
            "jaw_accuracy": float(jaw_acc)
        }, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")

# =========================================
#  MAIN
# =========================================

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    dataset = Tooth2DDataset(
        DATA_PATHS,
        IMG_ROOTS,
        image_suffix=IMAGE_SUFFIX,
        transform=None,
        cache_images=CACHE_IMAGES
    )
    if len(dataset) == 0:
        return

    train_indices, val_indices = split_by_case_id(dataset, train_ratio=0.8, seed=SEED)
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    train_dataset = TransformWrapper(train_set, train_transform)
    val_dataset = TransformWrapper(val_set, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )

    model = ResNet16Plus1(backbone=BACKBONE, dropout_rate=DROPOUT_RATE).to(device)
    criterion = Dynamit_Loss(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"\n[3/5] Training for {NUM_EPOCHS} epochs...")
    best_f1 = -1.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_jaw_acc": [],
        "val_jaw_acc": []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        t_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, is_train=True)
        v_metrics = run_epoch(model, val_loader, criterion, device, is_train=False)
        scheduler.step()

        history["train_loss"].append(t_metrics["loss"])
        history["val_loss"].append(v_metrics["loss"])
        history["train_f1"].append(t_metrics["f1"])
        history["val_f1"].append(v_metrics["f1"])
        history["train_jaw_acc"].append(t_metrics["jaw_acc"])
        history["val_jaw_acc"].append(v_metrics["jaw_acc"])

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"Loss: {t_metrics['loss']:.4f}/{v_metrics['loss']:.4f} | "
            f"F1: {t_metrics['f1']:.4f}/{v_metrics['f1']:.4f} | "
            f"Jaw: {t_metrics['jaw_acc']:.4f}/{v_metrics['jaw_acc']:.4f}"
        )

        if v_metrics["f1"] > best_f1:
            best_f1 = v_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": best_f1
            }, Path(OUTPUT_DIR) / BEST_MODEL_FILENAME)

    print("\nLoading best model for final report...")
    checkpoint = torch.load(Path(OUTPUT_DIR) / BEST_MODEL_FILENAME, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print_final_report(model, val_loader, device)
    plot_training_curves(history, PLOT_DIR, PLOT_FILENAME)


if __name__ == "__main__":
    main()
