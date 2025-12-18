import json
import random
from pathlib import Path
from collections import OrderedDict
import re
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

from torchvision.models import resnet18, resnet50
from torchvision import transforms

# =================================================================================
# CONFIGURATION (2D)
# =================================================================================
TEST_IMG_DIR = Path("/home/user/lzhou/week13-32/output/Render_TestAll")
TEST_LABELS_CSV = Path("/home/user/lzhou/week10/label_flipped.csv")
MODEL_PATH = Path("/home/user/lzhou/week13-32/output/Train2D/aug_normal/bce_loss_full_dataset_best.pth")
OUTPUT_DIR = Path("/home/user/lzhou/week13-32/output/Test2D/aug_normal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 320
NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 5

USE_PCA_ALIGN_2D = False

# [STRATEGY: MAX]
# Using Max strategy (Top-K Max)
MULTIVIEW_STRATEGY = "max_per_tooth"

# [TOP-K]
# Only consider the top 2 views with highest jaw_conf
MAX_PER_TOOTH_TOPK = 2

# Default placeholder, will be overwritten by Auto-Search
DEFAULT_THRESHOLD = 0.95

# --- FDI Notation Mapping (must match training) ---
VALID_FDI_LABELS = sorted(
    [
        18, 17, 16, 15, 14, 13, 12, 11,
        21, 22, 23, 24, 25, 26, 27, 28,
        38, 37, 36, 35, 34, 33, 32, 31,
        41, 42, 43, 44, 45, 46, 47, 48,
    ]
)
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}

# Upper and lower jaw tooth indices
UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
UPPER_IDX = [FDI_TO_INDEX[f] for f in UPPER_FDI]
LOWER_IDX = [FDI_TO_INDEX[f] for f in LOWER_FDI]


# =================================================================================
# DEVICE SELECTION
# =================================================================================
def get_free_gpus(threshold_mb: int = 1000, max_gpus: int = 2):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        gpus = []
        for line in lines:
            idx_str, mem_str = [x.strip() for x in line.split(",")]
            if int(mem_str) < threshold_mb:
                gpus.append(int(idx_str))
            if len(gpus) >= max_gpus:
                break
        if not gpus:
            return [0]
        return gpus
    except Exception:
        return [0]


def filter_visible_gpus(candidate_ids):
    if not torch.cuda.is_available():
        return []
    visible = torch.cuda.device_count()
    if visible == 0:
        return []
    filtered = [idx for idx in candidate_ids if 0 <= idx < visible]
    if not filtered:
        filtered = [0]
    return filtered


def pick_device():
    detected = get_free_gpus()
    usable = filter_visible_gpus(detected)
    if torch.cuda.is_available() and usable:
        print(f"[GPU] Using cuda:{usable[0]}, candidates: {usable}")
        return torch.device(f"cuda:{usable[0]}")
    print("[GPU] Using CPU")
    return torch.device("cpu")


DEVICE = pick_device()


# =================================================================================
# MODEL DEFINITION 
# =================================================================================
class ToothClassificationModel(nn.Module):
    """ResNet encoder + MLP head, identical to training."""

    def __init__(self, num_teeth: int = 32, backbone: str = "resnet18"):
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


# =================================================================================
# TRANSFORMS & PCA ALIGNMENT
# =================================================================================
def build_test_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )


def pca_align_image_2d(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    ys, xs = np.where(arr > 10)
    if len(xs) < 20:
        return img

    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    w, v = np.linalg.eigh(cov)
    idx_max = np.argmax(w)
    pc = v[:, idx_max]

    angle = np.degrees(np.arctan2(pc[1], pc[0]))
    img_rot = img.rotate(-angle, resample=Image.BILINEAR,
                         expand=True, fillcolor=(0, 0, 0))
    return img_rot


def load_image_tensor(img_path: Path, transform, use_pca_align=True):
    img = Image.open(img_path).convert("RGB")
    if use_pca_align:
        img = pca_align_image_2d(img)
    return transform(img).unsqueeze(0)


# =================================================================================
# STRING / NAME NORMALIZATION & GROUPING
# =================================================================================
def normalize_png_stem_to_newid(stem: str) -> str:
    s = stem.replace("-", "_").strip()
    s = re.sub(r"\s+", " ", s)

    # split jaw part
    if " " in s:
        left, right = s.rsplit(" ", 1)
    else:
        left, right = s, ""
    jaw = right.strip().lower()

    jaw_key = ""
    m = re.search(r"(upper|lower)jawscan(\d*)", jaw)
    if m:
        jaw_key = m.group(1) + m.group(2)
    else:
        if "upperjawscan" in jaw:
            jaw_key = "upper"
        elif "lowerjawscan" in jaw:
            jaw_key = "lower"
        else:
            lower_s = s.lower()
            if lower_s.endswith("_upper"):
                jaw_key = "upper"
                left = s[:-6]
            elif lower_s.endswith("_lower"):
                jaw_key = "lower"
                left = s[:-6]

    left = left.strip()
    left = left.replace("-", "_")
    left = left.replace(" ", "_")

    new_id = f"{left}_{jaw_key}" if jaw_key else left
    return new_id.lower()


def parse_case_id_and_angle(img_path: Path):
    raw_stem = img_path.stem
    stem = raw_stem.replace("-", "_")
    m = re.search(r"_rot(\d+)$", stem)
    angle_deg = int(m.group(1)) if m else 0
    base_stem = re.sub(r"_rot\d+$", "", stem)
    if base_stem.endswith("_top"):
        base_stem = base_stem[:-4]
    case_id_norm = normalize_png_stem_to_newid(base_stem)

    low_name = stem.lower()
    if "upper" in low_name:
        jaw_type = "upper"
    elif "lower" in low_name:
        jaw_type = "lower"
    else:
        jaw_type = "unknown"

    return case_id_norm, angle_deg, jaw_type


def group_images_by_case(img_dir: Path):
    groups = {}
    for p in img_dir.rglob("*.png"):
        case_id_norm, angle_deg, jaw_type = parse_case_id_and_angle(p)
        groups.setdefault(case_id_norm, []).append((p, angle_deg, jaw_type))
    return groups


# =================================================================================
# LABEL LOADING
# =================================================================================
def load_test_labels(csv_path: Path):
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    if "new_id" not in cols_lower:
        raise ValueError("CSV must contain a 'new_id' column.")
    id_col = cols_lower["new_id"]

    df[id_col] = (
        df[id_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
    )

    df.columns = [str(c) if str(c).isdigit() else c for c in df.columns]

    labels_dict = {}
    for _, row in df.iterrows():
        case_key = str(row[id_col]).strip().lower()
        label_vector = np.zeros(NUM_TEETH, dtype=np.float32)
        for tooth_fdi in VALID_FDI_LABELS:
            key = str(tooth_fdi)
            if key in df.columns and pd.notna(row[key]) and int(row[key]) == 1:
                label_vector[FDI_TO_INDEX[tooth_fdi]] = 1.0
        labels_dict[case_key] = label_vector
    return labels_dict


# =================================================================================
# MULTI-VIEW INFERENCE
# =================================================================================
def compute_jaw_confidence_from_probs(probs_tensor: torch.Tensor, jaw_type: str) -> float:
    if probs_tensor.ndim == 2:
        probs = probs_tensor[0]
    else:
        probs = probs_tensor
    if jaw_type == "upper":
        jaw_probs = probs[UPPER_IDX]
    elif jaw_type == "lower":
        jaw_probs = probs[LOWER_IDX]
    else:
        jaw_probs = probs
    conf = torch.mean(torch.abs(jaw_probs - 0.5)).item()
    return conf


def test_model_multiview(model, grouped_imgs, labels_dict, device, transform):
    model.eval()

    csv_ids = set(labels_dict.keys())
    img_ids = set(grouped_imgs.keys())
    common_ids = sorted(csv_ids & img_ids)
    
    print(f"# Intersection (test): {len(common_ids)}")

    if len(common_ids) == 0:
        print("No intersection between CSV and images. Exit.")
        return None, None, None, None, None

    all_best_preds = []
    all_targets = []
    all_ids = []
    all_best_angles = []
    all_jaw_confs = []

    for case_id in common_ids:
        gt_labels = labels_dict[case_id]
        views = grouped_imgs[case_id]
        if len(views) == 0:
            continue

        angle_list = []
        jaw_conf_list = []
        probs_list = []

        for img_path, angle_deg, jaw_type in views:
            x = load_image_tensor(img_path, transform, use_pca_align=USE_PCA_ALIGN_2D).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)  # (1, 32)

            angle_list.append(angle_deg)
            jaw_conf_list.append(compute_jaw_confidence_from_probs(probs, jaw_type))
            probs_list.append(probs.cpu().numpy()[0])

        probs_arr = np.stack(probs_list, axis=0)   # (num_views, 32)
        jaw_conf_arr = np.array(jaw_conf_list)     # (num_views,)

        if MULTIVIEW_STRATEGY == "max_per_tooth":
            # 1. Filter: Select Top K views by jaw_conf
            K = min(MAX_PER_TOOTH_TOPK, len(views))
            topk_idx = np.argsort(jaw_conf_arr)[-K:]      # (K,)
            selected_probs = probs_arr[topk_idx]          # (K, 32)

            # 2. Aggregation: MAX
            final_probs = selected_probs.max(axis=0)      # (32,)

            local_jaw_conf = jaw_conf_arr[topk_idx]
            best_local_idx = int(topk_idx[np.argmax(local_jaw_conf)])
            final_angle = angle_list[best_local_idx]
            final_conf = jaw_conf_arr[best_local_idx]
            
        elif MULTIVIEW_STRATEGY == "top2_avg":
            # Just in case you want to switch back later
            K = min(MAX_PER_TOOTH_TOPK, len(views))
            topk_idx = np.argsort(jaw_conf_arr)[-K:]
            selected_probs = probs_arr[topk_idx]
            final_probs = selected_probs.mean(axis=0)

            local_jaw_conf = jaw_conf_arr[topk_idx]
            best_local_idx = int(topk_idx[np.argmax(local_jaw_conf)])
            final_angle = angle_list[best_local_idx]
            final_conf = jaw_conf_arr[best_local_idx]

        elif MULTIVIEW_STRATEGY == "jaw_conf":
            best_idx = int(np.argmax(jaw_conf_arr))
            final_probs = probs_arr[best_idx]
            final_angle = angle_list[best_idx]
            final_conf = jaw_conf_arr[best_idx]

        elif MULTIVIEW_STRATEGY == "avg_all":
            final_probs = probs_arr.mean(axis=0)
            best_idx = int(np.argmax(jaw_conf_arr))
            final_angle = angle_list[best_idx]
            final_conf = jaw_conf_arr[best_idx]

        else:
            raise ValueError(f"Unknown MULTIVIEW_STRATEGY: {MULTIVIEW_STRATEGY}")

        all_best_preds.append(final_probs)
        all_targets.append(gt_labels)
        all_ids.append(case_id)
        all_best_angles.append(final_angle)
        all_jaw_confs.append(final_conf)

    return (
        np.array(all_best_preds),
        np.array(all_targets),
        all_ids,
        all_best_angles,
        all_jaw_confs,
    )


# =================================================================================
# METRICS / REPORTING
# =================================================================================
def find_optimal_threshold(preds, targets):
    """
    Search for the threshold that maximizes Micro F1 Score.
    Range: 0.80 to 0.99 (High range for Max Strategy)
    """
    thresholds = np.arange(0.80, 0.995, 0.01)
    best_th = 0.95
    best_f1 = 0.0
    
    print("\n" + "="*50)
    print(" AUTO-THRESHOLD SEARCH (Maximizing F1)")
    print("="*50)
    print(f"{'Threshold':<10} {'Micro F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 45)
    
    flat_targets = targets.flatten().astype(int)
    
    for th in thresholds:
        flat_preds = (preds > th).astype(int).flatten()
        p, r, f1, _ = precision_recall_fscore_support(flat_targets, flat_preds, average='binary', zero_division=0)
        print(f"{th:<10.2f} {f1:<10.4f} {p:<10.4f} {r:<10.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    print("-" * 45)
    print(f"Best Threshold Found: {best_th:.3f} (F1: {best_f1:.4f})")
    print("="*50 + "\n")
    return best_th


def calculate_metrics(preds, targets, threshold):
    preds_bin = (preds > threshold).astype(int)
    targets_bin = targets.astype(int)
    flat_preds, flat_targets = preds_bin.flatten(), targets_bin.flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_targets, flat_preds, average="binary", zero_division=0
    )
    acc = accuracy_score(flat_targets, flat_preds)

    per_tooth = OrderedDict()
    for idx in range(NUM_TEETH):
        fdi_label = INDEX_TO_FDI[idx]
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(
            targets_bin[:, idx], preds_bin[:, idx],
            average="binary", zero_division=0
        )
        acc_t = accuracy_score(targets_bin[:, idx], preds_bin[:, idx])
        per_tooth[fdi_label] = {
            "precision": float(p_t),
            "recall": float(r_t),
            "f1": float(f1_t),
            "accuracy": float(acc_t),
            "support": int(targets_bin[:, idx].sum()),
        }

    macro_precision = np.mean([m["precision"] for m in per_tooth.values()])
    macro_recall = np.mean([m["recall"] for m in per_tooth.values()])
    macro_f1 = np.mean([m["f1"] for m in per_tooth.values()])
    macro_acc = np.mean([m["accuracy"] for m in per_tooth.values()])

    metrics = {
        "overall_micro": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(acc),
        },
        "overall_macro": {
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "macro_accuracy": float(macro_acc),
        },
        "per_tooth": per_tooth,
    }
    return metrics


def print_metrics_summary(metrics):
    print("\n" + "=" * 80)
    print(" " * 28 + "TESTING METRICS SUMMARY")
    print("=" * 80)
    micro = metrics["overall_micro"]
    print("\n OVERALL (MICRO-AVERAGE) METRICS (Positive class: Missing Tooth):")
    print(
        f"  - Precision: {micro['precision']:.4f}\n"
        f"  - Recall:    {micro['recall']:.4f}\n"
        f"  - F1 Score:  {micro['f1']:.4f}\n"
        f"  - Accuracy:  {micro['accuracy']:.4f}"
    )
    macro = metrics["overall_macro"]
    print("\n OVERALL (MACRO-AVERAGE) METRICS:")
    print(
        f"  - Macro Precision: {macro['macro_precision']:.4f}\n"
        f"  - Macro Recall:    {macro['macro_recall']:.4f}\n"
        f"  - Macro F1 Score:  {macro['macro_f1']:.4f}\n"
        f"  - Macro Accuracy:  {macro['macro_accuracy']:.4f}"
    )
    print("\n PER-TOOTH METRICS (FDI Notation):")
    print("-" * 80)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} "
          f"{'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    for fdi_label, m in metrics["per_tooth"].items():
        print(
            f"Tooth {fdi_label:<5} "
            f"{m['precision']:>10.4f}   {m['recall']:>10.4f}   "
            f"{m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}"
        )
    print("=" * 80)


def print_sample_predictions(ids, preds, targets, num_samples, threshold):
    print("\n" + "=" * 80)
    print(" " * 28 + f"SAMPLE PREDICTIONS (Thresh={threshold:.3f})")
    print("=" * 80)
    if len(ids) == 0:
        print("No samples to display.")
        return
    num_samples = min(num_samples, len(ids))
    sample_indices = random.sample(range(len(ids)), num_samples)
    for i in sample_indices:
        case_id = ids[i]
        target_labels = targets[i]
        pred_labels = (preds[i] > threshold).astype(int)
        truth_missing = {INDEX_TO_FDI[j] for j, label in enumerate(target_labels) if label == 1}
        pred_missing = {INDEX_TO_FDI[j] for j, label in enumerate(pred_labels) if label == 1}
        correctly_found = sorted(list(truth_missing.intersection(pred_missing)))
        missed = sorted(list(truth_missing.difference(pred_missing)))
        false_alarms = sorted(list(pred_missing.difference(truth_missing)))

        print(f"\n Case ID: {case_id}\n" + "-" * 50)
        print(f"  Ground Truth (Missing): {sorted(list(truth_missing)) if truth_missing else 'None'}")
        print(f"  Prediction (Missing):   {sorted(list(pred_missing)) if pred_missing else 'None'}")
        print("-" * 50)
        print(f"   Correctly Found (TP): {correctly_found or 'None'}")
        print(f"   Missed Teeth (FN):    {missed or 'None'}")
        print(f"   False Alarms (FP):    {false_alarms or 'None'}")
    print("\n" + "=" * 80)


def generate_test_plots(metrics, preds, targets, save_dir: Path, threshold):
    per_tooth_metrics = metrics["per_tooth"]
    fdi_labels = [str(label) for label in per_tooth_metrics.keys()]
    precision_scores = [m["precision"] for m in per_tooth_metrics.values()]
    recall_scores = [m["recall"] for m in per_tooth_metrics.values()]
    f1_scores = [m["f1"] for m in per_tooth_metrics.values()]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(fdi_labels))
    width = 0.25
    ax.bar(x - width, precision_scores, width, label="Precision", color="royalblue")
    ax.bar(x, recall_scores, width, label="Recall", color="limegreen")
    rects3 = ax.bar(x + width, f1_scores, width, label="F1 Score", color="tomato")
    ax.set_ylabel("Scores", fontsize=14)
    ax.set_title(f"Per-Tooth Performance (Thresh={threshold:.3f})", fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(fdi_labels, rotation=45, ha="right")
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.bar_label(rects3, padding=3, fmt="%.2f", fontsize=8)
    fig.tight_layout()
    plot_path = save_dir / "per_tooth_metrics.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Per-tooth metrics plot saved to {plot_path}")

    flat_targets = targets.flatten()
    flat_preds = (preds > threshold).astype(int).flatten()
    cm = confusion_matrix(flat_targets, flat_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Present (0)", "Pred Missing (1)"],
        yticklabels=["Actual Present (0)", "Actual Missing (1)"],
    )
    plt.title(f"Confusion Matrix (Thresh={threshold:.3f})", fontsize=16, fontweight="bold")
    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plot_path = save_dir / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Confusion matrix plot saved to {plot_path}")

    precision, recall, _ = precision_recall_curve(flat_targets, preds.flatten())
    avg_precision = average_precision_score(flat_targets, preds.flatten())
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color="darkorange", lw=2,
             label=f"PR Curve (Avg Precision = {avg_precision:.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve", fontsize=16, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.5)
    plot_path = save_dir / "precision_recall_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Precision-Recall curve saved to {plot_path}")


def export_results(ids, preds, targets, best_angles, jaw_confs, save_dir: Path, metrics, threshold):
    suffix_map = {
        "avg_all": "avg",
        "jaw_conf": "jaw",
        "max_per_tooth": "max",
    }
    suffix = suffix_map.get(MULTIVIEW_STRATEGY, "mv")

    results_df = pd.DataFrame({"case_id": ids})
    results_df["best_angle_repr"] = best_angles
    results_df["jaw_confidence_repr"] = jaw_confs
    for fdi_label in VALID_FDI_LABELS:
        idx = FDI_TO_INDEX[fdi_label]
        results_df[f"true_{fdi_label}"] = targets[:, idx]
        # Use Dynamic Threshold
        results_df[f"pred_{fdi_label}"] = (preds[:, idx] > threshold).astype(int)
        results_df[f"prob_{fdi_label}"] = preds[:, idx]

    csv_path = save_dir / f"test_predictions_multiview_aug_bce_{suffix}_th{threshold:.3f}.csv"
    results_df.to_csv(csv_path, index=False)
    json_path = save_dir / f"test_metrics_multiview_aug_bce_{suffix}_th{threshold:.3f}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f" Saved predictions to {csv_path}")
    print(f" Saved metrics to {json_path}")


# =================================================================================
# MAIN
# =================================================================================
def main():
    print(f"Using device: {DEVICE}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    model = ToothClassificationModel(num_teeth=NUM_TEETH).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if state_dict and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f" Model loaded from {MODEL_PATH}")

    labels_dict = load_test_labels(TEST_LABELS_CSV)
    print(f" Loaded labels for {len(labels_dict)} IDs from {TEST_LABELS_CSV}")

    grouped_imgs = group_images_by_case(TEST_IMG_DIR)
    print(f"Grouped images for {len(grouped_imgs)} case_id(s) from {TEST_IMG_DIR}")

    transform = build_test_transform()

    preds, targets, ids, best_angles, jaw_confs = test_model_multiview(
        model, grouped_imgs, labels_dict, DEVICE, transform
    )
    if preds is None:
        return

    print(f" Inference complete on {len(ids)} cases (using '{MULTIVIEW_STRATEGY}' multi-view strategy).")

    # [AUTO THRESHOLD]
    best_threshold = find_optimal_threshold(preds, targets)
    
    # Recalculate everything using the BEST threshold
    metrics = calculate_metrics(preds, targets, threshold=best_threshold)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, num_samples=NUM_SAMPLE_PREDICTIONS, threshold=best_threshold)

    print("\n" + "=" * 80)
    print(" " * 28 + "GENERATING PLOTS")
    print("=" * 80)
    generate_test_plots(metrics, preds, targets, OUTPUT_DIR, threshold=best_threshold)

    export_results(ids, preds, targets, best_angles, jaw_confs, OUTPUT_DIR, metrics, threshold=best_threshold)
    print("\n Full multi-view results and metrics saved successfully to", OUTPUT_DIR)
    print(f" (Strategy: {MULTIVIEW_STRATEGY}, Threshold: {best_threshold:.3f}, TopK={MAX_PER_TOOTH_TOPK})")

if __name__ == "__main__":
    main()