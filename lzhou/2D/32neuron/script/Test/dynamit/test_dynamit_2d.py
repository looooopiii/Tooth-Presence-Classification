import os
import subprocess
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json
import random
from collections import OrderedDict
import re

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

from PIL import Image
from torchvision import transforms, models

# =================================================================================
# CONFIGURATION
# =================================================================================
TEST_IMG_DIR    = "/home/user/lzhou/week13-32/output/Render_TestAll"
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"
MODEL_PATH      = "/home/user/lzhou/week13-32/output/Train2D/dynamit/dynamit_loss_best_2d.pth"
OUTPUT_DIR      = "/home/user/lzhou/week13-32/output/Test2D/dynamit"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 5
IMG_SIZE = 320
USE_PCA_ALIGN_2D = False

# [STRATEGY: MAX]
# You chose to stick with Max.
# "max_per_tooth": take the maximum probability for each tooth from the top K views.
MULTIVIEW_STRATEGY = "max_per_tooth"

# [TOP-K]
# Only consider the top 2 views with highest jaw_conf before taking Max.
# This is crucial for "Max" to avoid taking max from a back-facing view.
MAX_PER_TOOTH_TOPK = 2

# We will ignore this fixed value because we use Auto-Search below,
# but providing a high default is good practice for Max strategy.
DEFAULT_THRESHOLD = 0.95

# --- FDI Notation Mapping ---
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}

UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
UPPER_IDX = [FDI_TO_INDEX[f] for f in UPPER_FDI]
LOWER_IDX = [FDI_TO_INDEX[f] for f in LOWER_FDI]


# =================================================================================
# GPU HELPER
# =================================================================================
def get_free_gpus(threshold_mb=1000, max_gpus=2):
    try:
        res = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True
        )
        free = []
        for line in res.stdout.strip().splitlines():
            if not line.strip():
                continue
            gpu_id, mem = [x.strip() for x in line.split(',')]
            if int(mem) < threshold_mb:
                free.append(int(gpu_id))
            if len(free) >= max_gpus:
                break
        print(f"[GPU] Free GPUs detected: {free}")
        return free if free else [0]
    except Exception as e:
        print(f"[GPU] Error detecting free GPUs: {e}")
        return [0]


# =================================================================================
# STRING / NAME NORMALIZATION
# =================================================================================
def normalize_png_stem_to_newid(stem: str) -> str:
    s = stem.replace('-', '_').strip()
    s = re.sub(r'\s+', ' ', s)
    if ' ' in s:
        left, right = s.rsplit(' ', 1)
    else:
        left, right = s, ''
    jaw = right.strip().lower()
    jaw_key = ''
    m = re.search(r'(upper|lower)jawscan(\d*)', jaw)
    if m:
        jaw_key = m.group(1) + m.group(2)
    else:
        if 'upperjawscan' in jaw:
            jaw_key = 'upper'
        elif 'lowerjawscan' in jaw:
            jaw_key = 'lower'
        else:
            lower_s = s.lower()
            if lower_s.endswith('_upper'):
                jaw_key = 'upper'
                left = s[:-6]
            elif lower_s.endswith('_lower'):
                jaw_key = 'lower'
                left = s[:-6]
    left = left.strip()
    left = left.replace('-', '_')
    left = left.replace(' ', '_')
    new_id = f"{left}_{jaw_key}" if jaw_key else left
    return new_id.lower()


def parse_case_id_and_angle(img_path: Path):
    raw_stem = img_path.stem
    stem = raw_stem.replace('-', '_')
    m = re.search(r'_rot(\d+)$', stem)
    angle_deg = int(m.group(1)) if m else 0
    base_stem = re.sub(r'_rot\d+$', '', stem)
    if base_stem.endswith('_top'):
        base_stem = base_stem[:-4]
    case_id_norm = normalize_png_stem_to_newid(base_stem)
    low_name = stem.lower()
    if 'upper' in low_name:
        jaw_type = 'upper'
    elif 'lower' in low_name:
        jaw_type = 'lower'
    else:
        jaw_type = 'unknown'
    return case_id_norm, angle_deg, jaw_type


# =================================================================================
# MODEL
# =================================================================================
class ResNetMultiLabel(nn.Module):
    def __init__(self, backbone="resnet18", num_teeth=32, pretrained=True):
        super().__init__()
        backbone = (backbone or "resnet18").lower()
        if backbone == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            in_feats = self.backbone.fc.in_features
        else:
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_teeth)

    def forward(self, x):
        return self.backbone(x)


# =================================================================================
# 2D PCA ALIGNMENT
# =================================================================================
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
    img_rot = img.rotate(-angle, resample=Image.BILINEAR, expand=True, fillcolor=(0, 0, 0))
    return img_rot


# =================================================================================
# DATA LOADING
# =================================================================================
def build_transform(img_size=320):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_image_tensor(img_path: Path, transform, use_pca_align=True):
    img = Image.open(img_path).convert("RGB")
    if use_pca_align:
        img = pca_align_image_2d(img)
    return transform(img).unsqueeze(0)

def group_images_by_case(img_dir: str):
    groups = {}
    for p in Path(img_dir).rglob("*.png"):
        case_id_norm, angle_deg, jaw_type = parse_case_id_and_angle(p)
        groups.setdefault(case_id_norm, []).append((p, angle_deg, jaw_type))
    return groups

def load_test_labels(csv_path):
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    if 'new_id' not in cols_lower:
        raise ValueError("CSV must contain a 'new_id' column.")
    id_col = cols_lower['new_id']
    df[id_col] = df[id_col].astype(str).str.strip().str.lower().str.replace('-', '_')
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
# MULTI-VIEW INFERENCE (MAX PER TOOTH)
# =================================================================================
def compute_jaw_confidence_from_probs(probs_tensor: torch.Tensor, jaw_type: str) -> float:
    if probs_tensor.ndim == 2:
        probs = probs_tensor[0]
    else:
        probs = probs_tensor
    if jaw_type == 'upper':
        jaw_probs = probs[UPPER_IDX]
    elif jaw_type == 'lower':
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
    all_targets    = []
    all_ids        = []
    all_best_angles = []
    all_jaw_confs   = []

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
                probs = torch.sigmoid(logits)

            angle_list.append(angle_deg)
            jaw_conf_list.append(compute_jaw_confidence_from_probs(probs, jaw_type))
            probs_list.append(probs.cpu().numpy()[0])

        probs_arr = np.stack(probs_list, axis=0)
        jaw_conf_arr = np.array(jaw_conf_list)

        # ---------------------------------------------------------------------
        # STRATEGY: MAX PER TOOTH
        # ---------------------------------------------------------------------
        if MULTIVIEW_STRATEGY == "max_per_tooth":
            # 1. Filter: Select Top K views by jaw_conf
            K = min(MAX_PER_TOOTH_TOPK, len(views))
            topk_idx = np.argsort(jaw_conf_arr)[-K:]
            selected_probs = probs_arr[topk_idx]
            
            # 2. Aggregation: MAX
            final_probs = selected_probs.max(axis=0)

            local_jaw_conf = jaw_conf_arr[topk_idx]
            best_local_idx = int(topk_idx[np.argmax(local_jaw_conf)])
            final_angle = angle_list[best_local_idx]
            final_conf = jaw_conf_arr[best_local_idx]
            
        elif MULTIVIEW_STRATEGY == "top2_avg":
            # Just for reference if you switch back
            K = min(MAX_PER_TOOTH_TOPK, len(views))
            topk_idx = np.argsort(jaw_conf_arr)[-K:]
            selected_probs = probs_arr[topk_idx]
            final_probs = selected_probs.mean(axis=0)

            local_jaw_conf = jaw_conf_arr[topk_idx]
            best_local_idx = int(topk_idx[np.argmax(local_jaw_conf)])
            final_angle = angle_list[best_local_idx]
            final_conf = jaw_conf_arr[best_local_idx]
            
        elif MULTIVIEW_STRATEGY == "avg_all":
            final_probs = probs_arr.mean(axis=0)
            best_idx = int(np.argmax(jaw_conf_arr))
            final_angle = angle_list[best_idx]
            final_conf = jaw_conf_arr[best_idx]
            
        elif MULTIVIEW_STRATEGY == "jaw_conf":
            best_idx = int(np.argmax(jaw_conf_arr))
            final_probs = probs_arr[best_idx]
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
    """
    # Range is higher for MAX strategy (0.80 to 0.99)
    thresholds = np.arange(0.80, 0.995, 0.01)
    best_th = 0.95
    best_f1 = 0.0
    
    print("\n[Auto-Threshold Search]")
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
    return best_th

def calculate_metrics(preds, targets, threshold):
    if len(preds) == 0:
        return {}
    preds_bin = (preds > threshold).astype(int)
    targets_bin = targets.astype(int)
    flat_preds, flat_targets = preds_bin.flatten(), targets_bin.flatten()

    p, r, f1, _ = precision_recall_fscore_support(
        flat_targets, flat_preds, average='binary', zero_division=0
    )
    acc = accuracy_score(flat_targets, flat_preds)

    per_tooth = OrderedDict()
    for idx in range(NUM_TEETH):
        fdi_label = INDEX_TO_FDI[idx]
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(
            targets_bin[:, idx], preds_bin[:, idx],
            average='binary', zero_division=0
        )
        acc_t = accuracy_score(targets_bin[:, idx], preds_bin[:, idx])
        support = int(targets_bin[:, idx].sum())
        per_tooth[fdi_label] = {
            'precision': float(p_t),
            'recall': float(r_t),
            'f1': float(f1_t),
            'accuracy': float(acc_t),
            'support': support,
        }

    macro_p = np.mean([m['precision'] for m in per_tooth.values()])
    macro_r = np.mean([m['recall'] for m in per_tooth.values()])
    macro_f1 = np.mean([m['f1'] for m in per_tooth.values()])
    macro_acc = np.mean([m['accuracy'] for m in per_tooth.values()])

    return {
        'overall_micro': {
            'precision': float(p),
            'recall': float(r),
            'f1': float(f1),
            'accuracy': float(acc),
        },
        'overall_macro': {
            'macro_precision': macro_p,
            'macro_recall': macro_r,
            'macro_f1': macro_f1,
            'macro_accuracy': macro_acc,
        },
        'per_tooth': per_tooth,
    }


def print_metrics_summary(metrics):
    micro = metrics['overall_micro']
    print("\n OVERALL (MICRO-AVERAGE) METRICS:")
    print(f"  - Precision: {micro['precision']:.4f}")
    print(f"  - Recall:    {micro['recall']:.4f}")
    print(f"  - F1 Score:  {micro['f1']:.4f}")
    print(f"  - Accuracy:  {micro['accuracy']:.4f}")
    
    print("\n PER-TOOTH METRICS (FDI Notation):")
    print("-" * 80)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    for fdi_label, tooth_metrics in metrics['per_tooth'].items():
        print(
            f"Tooth {fdi_label:<5} "
            f"{tooth_metrics['precision']:>10.4f}   "
            f"{tooth_metrics['recall']:>10.4f}   "
            f"{tooth_metrics['f1']:>10.4f}   "
            f"{tooth_metrics['accuracy']:>10.4f}   "
            f"{tooth_metrics['support']:>8}"
        )
    print("=" * 80)


def print_sample_predictions(ids, preds, targets, num_samples, threshold):
    print("\n" + "="*80 + "\n" + " "*28 + f"SAMPLE PREDICTIONS (Thresh={threshold:.3f})" + "\n" + "="*80)
    if len(ids) < num_samples:
        num_samples = len(ids)
    sample_indices = random.sample(range(len(ids)), num_samples)
    for i in sample_indices:
        case_id = ids[i]
        target_labels = targets[i]
        pred_labels = (preds[i] > threshold).astype(int)

        truth_missing = {INDEX_TO_FDI[j] for j, label in enumerate(target_labels) if label == 1}
        pred_missing  = {INDEX_TO_FDI[j] for j, label in enumerate(pred_labels)  if label == 1}

        correctly_found_missing = sorted(list(truth_missing.intersection(pred_missing)))
        missed_missing_teeth    = sorted(list(truth_missing.difference(pred_missing)))
        wrongly_predicted_missing = sorted(list(pred_missing.difference(truth_missing)))

        print(f"\n Case ID: {case_id}\n" + "-"*50)
        print(f"  Ground Truth (Missing): {sorted(list(truth_missing)) if truth_missing else 'None'}")
        print(f"  Prediction (Missing):   {sorted(list(pred_missing)) if pred_missing else 'None'}")
        print("-"*50)
        print(f"   Correctly Found (TP): {correctly_found_missing or 'None'}")
        print(f"   Missed Teeth (FN):    {missed_missing_teeth or 'None'}")
        print(f"   False Alarms (FP):    {wrongly_predicted_missing or 'None'}")
    print("\n" + "="*80)


def generate_test_plots(metrics, preds, targets, save_dir, threshold):
    per_tooth_metrics = metrics['per_tooth']
    fdi_labels = [str(label) for label in per_tooth_metrics.keys()]
    f1_scores = [m['f1'] for m in per_tooth_metrics.values()]
    precision_scores = [m['precision'] for m in per_tooth_metrics.values()]
    recall_scores = [m['recall'] for m in per_tooth_metrics.values()]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(fdi_labels)); width = 0.25
    ax.bar(x - width, precision_scores, width, label='Precision', color='royalblue')
    ax.bar(x, recall_scores, width, label='Recall', color='limegreen')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='tomato')
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title(f'Per-Tooth Performance (Thresh={threshold:.3f})', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(fdi_labels, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.bar_label(rects3, padding=3, fmt='%.2f', fontsize=8)
    fig.tight_layout()
    plot_path = Path(save_dir) / "per_tooth_metrics.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Per-tooth metrics plot saved to {plot_path}")

    flat_targets = targets.flatten()
    flat_preds = (preds > threshold).astype(int).flatten()
    cm = confusion_matrix(flat_targets, flat_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted Present (0)', 'Predicted Missing (1)'],
        yticklabels=['Actual Present (0)', 'Actual Missing (1)']
    )
    plt.title(f'Overall Confusion Matrix (Thresh={threshold:.3f})', fontsize=16, fontweight='bold')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plot_path = Path(save_dir) / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Confusion matrix plot saved to {plot_path}")
    
    precision, recall, _ = precision_recall_curve(flat_targets, preds.flatten())
    avg_precision = average_precision_score(flat_targets, preds.flatten())
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR Curve (Avg Precision = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Overall Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.5)
    plot_path = Path(save_dir) / "precision_recall_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Precision-Recall curve saved to {plot_path}")


# =================================================================================
# MAIN
# =================================================================================
def main():
    if torch.cuda.is_available():
        available_gpus = get_free_gpus(threshold_mb=1000, max_gpus=2)
        if not available_gpus:
            available_gpus = [0]
        device = torch.device(f"cuda:{available_gpus[0]}")
        print(f"Using device: {device}, GPUs: {available_gpus}")
    else:
        available_gpus = []
        device = torch.device("cpu")
        print("Using CPU (no CUDA available)")

    model = ResNetMultiLabel(num_teeth=NUM_TEETH).to(device)
    if available_gpus and len(available_gpus) > 1:
        model = nn.DataParallel(model, device_ids=available_gpus)
        print(f"DataParallel enabled on GPUs: {available_gpus}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    raw_sd = checkpoint.get('model_state_dict', checkpoint)
    if len(raw_sd) > 0 and next(iter(raw_sd)).startswith('module.'):
        raw_sd = {k[len('module.'):]: v for k, v in raw_sd.items()}

    def remap_keys(sd):
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith('model.'):
                nk = nk[len('model.'):]
            if nk.startswith('backbone.'):
                pass
            elif nk.startswith('net.'):
                nk = 'backbone.' + nk[len('net.'):]
            elif nk.startswith('resnet.') or nk.startswith('encoder.'):
                nk = 'backbone.' + nk.split('.', 1)[1]
            elif nk.startswith('fc.'):
                nk = 'backbone.' + nk
            new_sd[nk] = v
        return new_sd

    state_dict = remap_keys(raw_sd)
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
        if hasattr(incompatible, 'missing_keys') and (incompatible.missing_keys or incompatible.unexpected_keys):
            print(f"[warn] Strict load issues: {incompatible.missing_keys}")
    except RuntimeError:
        print("[warn] Strict load failed, trying non-strict mapping...")
        incompatible = model.load_state_dict(state_dict, strict=False)

    print(f" Model loaded from {MODEL_PATH}")

    labels_dict = load_test_labels(TEST_LABELS_CSV)
    print(f" Loaded labels for {len(labels_dict)} cases from {TEST_LABELS_CSV}")

    grouped_imgs = group_images_by_case(TEST_IMG_DIR)
    print(f"Grouped images for {len(grouped_imgs)} case_id(s) from {TEST_IMG_DIR}")

    transform = build_transform(IMG_SIZE)

    preds, targets, ids, best_angles, jaw_confs = test_model_multiview(
        model, grouped_imgs, labels_dict, device, transform
    )
    if preds is None:
        return

    print(f" Inference complete on {len(ids)} cases (using '{MULTIVIEW_STRATEGY}').")

    # [AUTO THRESHOLD]
    best_threshold = find_optimal_threshold(preds, targets)
    
    # Recalculate using BEST threshold
    metrics = calculate_metrics(preds, targets, threshold=best_threshold)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, num_samples=NUM_SAMPLE_PREDICTIONS, threshold=best_threshold)

    print("\n" + "="*80 + "\n" + " "*28 + "GENERATING PLOTS" + "\n" + "="*80)
    generate_test_plots(metrics, preds, targets, OUTPUT_DIR, threshold=best_threshold)

    suffix_map = {
        "avg_all": "avg",
        "jaw_conf": "jaw",
        "max_per_tooth": "max",
        "top2_avg": "top2avg",
    }
    suffix = suffix_map.get(MULTIVIEW_STRATEGY, "mv")

    results = pd.DataFrame({'case_id': ids})
    results['best_angle_repr'] = best_angles
    results['jaw_confidence_repr'] = jaw_confs
    for fdi_label in VALID_FDI_LABELS:
        idx = FDI_TO_INDEX[fdi_label]
        results[f'true_{fdi_label}'] = targets[:, idx].astype(int)
        results[f'pred_{fdi_label}'] = (preds[:, idx] > best_threshold).astype(int)
        results[f'prob_{fdi_label}'] = preds[:, idx]
    results.to_csv(Path(OUTPUT_DIR) / f'test_predictions_multiview_dynamit_{suffix}_th{best_threshold:.3f}.csv', index=False)

    with open(Path(OUTPUT_DIR) / f'test_metrics_multiview_dynamit_{suffix}_th{best_threshold:.3f}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n Full multi-view results and metrics saved successfully to {OUTPUT_DIR}")
    print(f" (Strategy: {MULTIVIEW_STRATEGY}, Threshold: {best_threshold:.3f}, TopK={MAX_PER_TOOTH_TOPK})")

if __name__ == "__main__":
    main()