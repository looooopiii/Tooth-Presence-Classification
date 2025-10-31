import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import random
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for servers
import seaborn as sns

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

import PIL
from PIL import Image
from torchvision import transforms, models


# NOTE: This script now tests a 2D image model (ResNet) using *_top.png files.
# =================================================================================
# CONFIGURATION
# =================================================================================
TEST_IMG_DIR = "/home/user/lzhou/week9/Clean_Test_Data"
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"
MODEL_PATH = "/home/user/lzhou/week11/output/Train2D/normal/resnet18_bce_best_2d.pth"
OUTPUT_DIR = "/home/user/lzhou/week11/output/Test2D/normal"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

NUM_TEETH = 16  # per jaw
NUM_OUTPUTS = 17  # 16 teeth + 1 jaw
NUM_SAMPLE_PREDICTIONS = 5  # How many random samples to print in the terminal
IMG_SIZE = 320

# --- FDI Notation Mapping (Must be identical to the training script) ---
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}

INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}

# Per-jaw fixed FDI order used in training
UPPER_FDI_ORDER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI_ORDER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
JAW_UPPER_LABEL = 0
JAW_LOWER_LABEL = 1


# =================================================================================
# STRING / NAME NORMALIZATION
# =================================================================================
import re
def _norm(s: str) -> str:
    return str(s).strip().lower()

def normalize_png_stem_to_newid(stem: str) -> str:
    """
    Convert stems like:
      '12345_2022-09-14 LowerJawScan' -> '12345_2022_09_14_lower'
      '12345_2022-09-14 UpperJawScan' -> '12345_2022_09_14_upper'
    """
    s = stem.strip()
    s = re.sub(r'\s+', ' ', s)
    if ' ' in s:
        left, right = s.rsplit(' ', 1)
    else:
        left, right = s, ''
    jaw = right.strip().lower()
    if 'upperjawscan' in jaw:
        jaw_key = 'upper'
    elif 'lowerjawscan' in jaw:
        jaw_key = 'lower'
    else:
        if s.lower().endswith('_upper'):
            jaw_key = 'upper'; left = s[:-6]
        elif s.lower().endswith('_lower'):
            jaw_key = 'lower'; left = s[:-6]
        else:
            jaw_key = ''
    left = left.strip()
    left = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1_\2_\3', left)
    left = left.replace(' ', '_')
    new_id = f"{left}_{jaw_key}" if jaw_key else left
    return new_id.lower()



# =================================================================================
# MODEL DEFINITION (Must be identical to the training script)
# =================================================================================

class ResNetMultiLabel(nn.Module):
    def __init__(self, backbone="resnet18", num_teeth=32, pretrained=True):
        super().__init__()
        backbone = (backbone or "resnet18").lower()
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            in_feats = self.backbone.fc.in_features
        else:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, NUM_OUTPUTS)

    def forward(self, x):
        return self.backbone(x)


# =================================================================================
# DATA PREPROCESSING AND LOADING
# =================================================================================

def build_transform(img_size=320):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def list_test_images(img_dir):
    """
    Returns a list of (png_path, case_id_norm, original_stem).
    case_id_norm maps to the CSV 'new_id' format.
    """
    items = []
    for p in Path(img_dir).rglob("*.png"):
        stem = p.stem
        if stem.endswith("_top"):
            stem = stem[:-4]
        case_id_norm = normalize_png_stem_to_newid(stem)
        items.append((p, case_id_norm, stem))
    return sorted(items, key=lambda x: str(x[0]))

def load_image_tensor(img_path, transform):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)  # add batch dim

def load_test_labels(csv_path):
    """
    Loads ground truth labels into a dict keyed by normalized case_id.
    Accepts either 'new_id' or 'filename' as the identifier column.
    If 'filename' is used (e.g., '142595_2023-12-07 UpperJawScan'),
    it will be normalized to the 'new_id' style via normalize_png_stem_to_newid().
    Returns a dict: key -> 32-dim vector (FDI order).
    """
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    id_col = None
    if 'new_id' in cols_lower:
        id_col = cols_lower['new_id']
        df[id_col] = df[id_col].astype(str).str.strip()
        df[id_col] = df[id_col].str.lower()
    elif 'filename' in cols_lower:
        id_col = cols_lower['filename']
        df[id_col] = df[id_col].astype(str).str.strip()
    else:
        raise ValueError("Identifier column not found. Expecting 'new_id' or 'filename' in CSV.")
    # make sure FDI cols are strings
    df.columns = [str(c) if str(c).isdigit() else c for c in df.columns]
    labels_dict = {}
    for _, row in df.iterrows():
        if id_col.lower() == 'new_id':
            case_key = _norm(row[id_col])
        else:
            # filename -> normalize to new_id style
            raw_stem = str(row[id_col]).replace('.png', '').replace('.ply', '')
            case_key = normalize_png_stem_to_newid(raw_stem)
        label_vector = np.zeros(32, dtype=np.float32)
        for tooth_fdi in VALID_FDI_LABELS:
            key = str(tooth_fdi)
            if key in df.columns and pd.notna(row[key]) and int(row[key]) == 1:
                label_vector[FDI_TO_INDEX[tooth_fdi]] = 1.0
        labels_dict[case_key] = label_vector
    return labels_dict

# ===================== Per-jaw helpers for 16+1 model =======================

def infer_jaw_from_case_id(case_key: str) -> int:
    if case_key.endswith('_upper'):
        return JAW_UPPER_LABEL
    if case_key.endswith('_lower'):
        return JAW_LOWER_LABEL
    # Fallback: try to guess from substrings
    k = case_key.lower()
    if 'upper' in k: return JAW_UPPER_LABEL
    if 'lower' in k: return JAW_LOWER_LABEL
    return JAW_UPPER_LABEL  # default


def to_perjaw_16vec(labels32: np.ndarray, jaw_label: int) -> np.ndarray:
    order = UPPER_FDI_ORDER if jaw_label == JAW_UPPER_LABEL else LOWER_FDI_ORDER
    v = np.zeros(NUM_TEETH, dtype=np.float32)
    for i, fdi in enumerate(order):
        idx32 = FDI_TO_INDEX[fdi]
        v[i] = float(labels32[idx32])
    return v


def calculate_per_fdi_metrics(preds_teeth, targets_teeth, targets_jaw):
    """
    Aggregates per-position predictions back into FDI notation so the report matches training.
    """
    if preds_teeth.size == 0:
        return OrderedDict(), {'macro_precision': 0.0, 'macro_recall': 0.0, 'macro_f1': 0.0, 'macro_accuracy': 0.0}

    aggregator = defaultdict(lambda: {'pred': [], 'target': []})
    for sample_idx in range(len(preds_teeth)):
        jaw_label = int(targets_jaw[sample_idx])
        order = UPPER_FDI_ORDER if jaw_label == JAW_UPPER_LABEL else LOWER_FDI_ORDER
        for pos, fdi in enumerate(order):
            aggregator[fdi]['pred'].append(preds_teeth[sample_idx, pos])
            aggregator[fdi]['target'].append(targets_teeth[sample_idx, pos])

    per_fdi = OrderedDict()
    macro_bag = []
    for fdi in sorted(aggregator.keys()):
        preds = np.array(aggregator[fdi]['pred'])
        targets = np.array(aggregator[fdi]['target']).astype(int)
        if targets.size == 0:
            continue
        preds_bin = (preds > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds_bin, average='binary', zero_division=0)
        accuracy = accuracy_score(targets, preds_bin)
        support = int(targets.sum())
        per_fdi[fdi] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'support': support,
        }
        macro_bag.append((precision, recall, f1, accuracy))

    if macro_bag:
        macro_precision = float(np.mean([m[0] for m in macro_bag]))
        macro_recall = float(np.mean([m[1] for m in macro_bag]))
        macro_f1 = float(np.mean([m[2] for m in macro_bag]))
        macro_accuracy = float(np.mean([m[3] for m in macro_bag]))
    else:
        macro_precision = macro_recall = macro_f1 = macro_accuracy = 0.0

    return per_fdi, {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_accuracy': macro_accuracy
    }


def calculate_jaw_metrics(preds_jaw, targets_jaw):
    preds_bin = (preds_jaw > 0.5).astype(int)
    targets_bin = targets_jaw.astype(int)
    accuracy = accuracy_score(targets_bin, preds_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(targets_bin, preds_bin, average='binary', zero_division=0)
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }

# =================================================================================
# INFERENCE AND METRICS
# =================================================================================


def test_model(model, test_items, device, transform):
    model.eval()
    all_preds_teeth, all_targets_teeth, all_ids, all_targets_jaw, all_preds_jaw = [], [], [], [], []
    with torch.no_grad():
        for img_path, case_id, labels32 in tqdm(test_items, desc="Testing"):
            x = load_image_tensor(img_path, transform).to(device)
            logits = model(x)  # [1, 17]
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            teeth_probs = probs[:NUM_TEETH]
            jaw_prob = probs[NUM_TEETH]
            jaw_t = infer_jaw_from_case_id(case_id)
            target_teeth16 = to_perjaw_16vec(labels32, jaw_t)
            all_preds_teeth.append(teeth_probs)
            all_preds_jaw.append(jaw_prob)
            all_targets_teeth.append(target_teeth16)
            all_targets_jaw.append(jaw_t)
            all_ids.append(case_id)
    return (np.array(all_preds_teeth), np.array(all_targets_teeth),
            np.array(all_preds_jaw), np.array(all_targets_jaw), all_ids)


def calculate_metrics(preds_teeth, targets_teeth, preds_jaw=None, targets_jaw=None):
    """Calculates overall (micro), macro-averaged, and per-position (1..16) metrics for teeth;
    and optional jaw accuracy if provided.
    """
    if len(preds_teeth) == 0: return {}
    preds_bin = (preds_teeth > 0.5).astype(int)
    targets_bin = targets_teeth.astype(int)
    flat_preds, flat_targets = preds_bin.flatten(), targets_bin.flatten()

    p, r, f1, _ = precision_recall_fscore_support(flat_targets, flat_preds, average='binary', zero_division=0)
    acc = accuracy_score(flat_targets, flat_preds)

    per_pos = OrderedDict()
    for idx in range(NUM_TEETH):
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(targets_bin[:, idx], preds_bin[:, idx], average='binary', zero_division=0)
        acc_t = accuracy_score(targets_bin[:, idx], preds_bin[:, idx])
        support = int(targets_bin[:, idx].sum())
        per_pos[idx+1] = {'precision': float(p_t), 'recall': float(r_t), 'f1': float(f1_t), 'accuracy': float(acc_t), 'support': support}

    macro_p = np.mean([m['precision'] for m in per_pos.values()]); macro_r = np.mean([m['recall'] for m in per_pos.values()])
    macro_f1 = np.mean([m['f1'] for m in per_pos.values()]); macro_acc = np.mean([m['accuracy'] for m in per_pos.values()])

    out = {
        'overall_micro': {'precision': float(p), 'recall': float(r), 'f1': float(f1), 'accuracy': float(acc)},
        'overall_macro': {'macro_precision': macro_p, 'macro_recall': macro_r, 'macro_f1': macro_f1, 'macro_accuracy': macro_acc},
        'per_position': per_pos
    }
    if preds_jaw is not None and targets_jaw is not None:
        out['jaw_metrics'] = calculate_jaw_metrics(preds_jaw, targets_jaw)
        per_fdi, macro_fdi = calculate_per_fdi_metrics(preds_teeth, targets_teeth, targets_jaw)
        out['per_fdi'] = per_fdi
        out['macro_fdi'] = macro_fdi
        out['missing_counts_per_fdi'] = {str(fdi): stats['support'] for fdi, stats in per_fdi.items()}

    out['missing_counts_per_position'] = [int(val) for val in targets_bin.sum(axis=0)]
    return out

# =================================================================================
# REPORTING AND VISUALIZATION
# =================================================================================

def print_metrics_summary(metrics):
    """Prints a formatted summary of all test metrics."""
    print("\n" + "="*80 + "\n" + " "*25 + "TESTING METRICS SUMMARY" + "\n" + "="*80)
    micro = metrics['overall_micro']
    print("\n OVERALL (MICRO-AVERAGE) METRICS (Positive class: Missing Tooth):")
    print(f"  - Precision: {micro['precision']:.4f}\n  - Recall:    {micro['recall']:.4f}\n  - F1 Score:  {micro['f1']:.4f}\n  - Accuracy:  {micro['accuracy']:.4f}")
    
    macro = metrics['overall_macro']
    print("\n OVERALL (MACRO-AVERAGE) METRICS:")
    print(f"  - Macro Precision: {macro['macro_precision']:.4f}\n  - Macro Recall:    {macro['macro_recall']:.4f}\n  - Macro F1 Score:  {macro['macro_f1']:.4f}\n  - Macro Accuracy:  {macro['macro_accuracy']:.4f}")
    
    print("\n PER-POSITION METRICS (Positions 1..16):")
    print("-" * 80)
    print(f"{'Pos':<6} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    for pos, m in metrics['per_position'].items():
        print(f"{pos:<6} {m['precision']:>10.4f}   {m['recall']:>10.4f}   {m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}")
    print("=" * 80)

    if 'macro_fdi' in metrics and 'per_fdi' in metrics:
        macro_fdi = metrics['macro_fdi']
        print("\n AGGREGATED PER-FDI METRICS:")
        print("-" * 80)
        print(f"  Macro Precision: {macro_fdi['macro_precision']:.4f}")
        print(f"  Macro Recall:    {macro_fdi['macro_recall']:.4f}")
        print(f"  Macro F1:        {macro_fdi['macro_f1']:.4f}")
        print(f"  Macro Accuracy:  {macro_fdi['macro_accuracy']:.4f}")

        print("\n PER-TOOTH (FDI) METRICS:")
        print("-" * 80)
        print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
        print("-" * 80)
        zero_support_teeth = []
        for fdi, stats in metrics['per_fdi'].items():
            support = stats['support']
            if support == 0:
                zero_support_teeth.append(fdi)
            print(
                f"Tooth {fdi:<5} {stats['precision']:>10.4f}   "
                f"{stats['recall']:>10.4f}   {stats['f1']:>10.4f}   "
                f"{stats['accuracy']:>10.4f}   {support:>8}"
            )
        if zero_support_teeth:
            print(f"\n [warn] No positives in ground truth for FDI teeth: {sorted(zero_support_teeth)}")
    print("=" * 80)


def print_sample_predictions(ids, preds, targets, num_samples):
    """Prints a clear, readable comparison for random samples.
    NOTE: Maps 16 positions back to FDI using the *per-jaw* order inferred from case_id.
    """
    print("\n" + "="*80 + "\n" + " "*28 + "SAMPLE PREDICTIONS" + "\n" + "="*80)
    if len(ids) < num_samples: num_samples = len(ids)

    sample_indices = random.sample(range(len(ids)), num_samples)
    for i in sample_indices:
        case_id = ids[i]
        jaw = infer_jaw_from_case_id(case_id)
        fdi_order = UPPER_FDI_ORDER if jaw == JAW_UPPER_LABEL else LOWER_FDI_ORDER
        target_labels = targets[i]
        pred_labels = (preds[i] > 0.5).astype(int)

        # Build FDI sets using the correct per-jaw order
        truth_missing = [fdi_order[j] for j, label in enumerate(target_labels) if label == 1]
        pred_missing  = [fdi_order[j] for j, label in enumerate(pred_labels) if label == 1]

        correctly_found_missing = sorted(list(set(truth_missing).intersection(pred_missing)))
        missed_missing_teeth = sorted(list(set(truth_missing).difference(pred_missing)))  # FN
        wrongly_predicted_missing = sorted(list(set(pred_missing).difference(truth_missing))) # FP

        print(f"\n Case ID: {case_id}\n" + "-"*50)
        print(f"  Ground Truth (Missing): {truth_missing if truth_missing else 'None'}")
        print(f"  Prediction (Missing):   {pred_missing  if pred_missing  else 'None'}")
        print("-"*50)
        print(f"   Correctly Found (TP): {correctly_found_missing or 'None'}")
        print(f"   Missed Teeth (FN):      {missed_missing_teeth or 'None'}")
        print(f"   False Alarms (FP):      {wrongly_predicted_missing or 'None'}")
    print("\n" + "="*80)

def generate_test_plots(metrics, preds_teeth, targets_teeth, save_dir, targets_jaw=None):
    """Generates and saves a collection of plots summarizing test performance."""
    per_pos_metrics = metrics['per_position']
    pos_labels = [str(k) for k in per_pos_metrics.keys()]  # 1..16
    f1_scores = [m['f1'] for m in per_pos_metrics.values()]
    precision_scores = [m['precision'] for m in per_pos_metrics.values()]
    recall_scores = [m['recall'] for m in per_pos_metrics.values()]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(len(pos_labels)); width = 0.25
    ax.bar(x - width, precision_scores, width, label='Precision', color='royalblue')
    ax.bar(x, recall_scores, width, label='Recall', color='limegreen')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='tomato')
    ax.set_ylabel('Scores', fontsize=14); ax.set_title('Per-Position Performance (1..16, Positive: Missing)', fontsize=18, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(pos_labels, rotation=0, ha='center'); ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05); ax.bar_label(rects3, padding=3, fmt='%.2f', fontsize=8); fig.tight_layout()
    plot_path = Path(save_dir) / "per_position_metrics.png"; plt.savefig(plot_path, dpi=300); plt.close()
    print(f"✓ Per-position metrics plot saved to {plot_path}")

    # --- UPDATED CM LABELS ---
    flat_targets = targets_teeth.flatten(); flat_preds = (preds_teeth > 0.5).astype(int).flatten()
    cm = confusion_matrix(flat_targets, flat_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Present (0)', 'Predicted Missing (1)'], 
                yticklabels=['Actual Present (0)', 'Actual Missing (1)'])
    plt.title('Overall Confusion Matrix (All Teeth)', fontsize=16, fontweight='bold'); plt.ylabel('Ground Truth'); plt.xlabel('Prediction')
    plot_path = Path(save_dir) / "confusion_matrix.png"; plt.savefig(plot_path, dpi=300); plt.close()
    print(f"✓ Confusion matrix plot saved to {plot_path}")
    
    precision, recall, _ = precision_recall_curve(flat_targets, preds_teeth.flatten())
    avg_precision = average_precision_score(flat_targets, preds_teeth.flatten())
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (Avg Precision = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Overall Precision-Recall Curve', fontsize=16, fontweight='bold'); plt.legend(loc="lower left"); plt.grid(True, alpha=0.5)
    plot_path = Path(save_dir) / "precision_recall_curve.png"; plt.savefig(plot_path, dpi=300); plt.close()
    print(f"✓ Precision-Recall curve saved to {plot_path}")

    if 'per_fdi' in metrics:
        fdi_labels = [str(label) for label in metrics['per_fdi'].keys()]
        f1_scores = [m['f1'] for m in metrics['per_fdi'].values()]
        plt.figure(figsize=(20, 8))
        sns.barplot(x=fdi_labels, y=f1_scores, color='royalblue')
        plt.ylim(0, 1.05)
        plt.xlabel('FDI Tooth')
        plt.ylabel('F1 Score')
        plt.title('Per-FDI F1 Scores (Positive: Missing)', fontsize=16, fontweight='bold')
        for idx, f1 in enumerate(f1_scores):
            plt.text(idx, f1 + 0.02, f"{f1:.2f}", ha='center', va='bottom', fontsize=8)
        plot_path = Path(save_dir) / "per_fdi_f1_scores.png"; plt.tight_layout(); plt.savefig(plot_path, dpi=300); plt.close()
        print(f"✓ Per-FDI F1 plot saved to {plot_path}")


# =================================================================================
# MAIN EXECUTION
# =================================================================================



def main():
    """Orchestrates the entire testing pipeline."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNetMultiLabel(num_teeth=NUM_TEETH).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Support both { 'model_state_dict': sd } and raw sd
    raw_sd = checkpoint.get('model_state_dict', checkpoint)
    # Strip DistributedDataParallel 'module.' if present
    if len(raw_sd) > 0 and next(iter(raw_sd)).startswith('module.'):
        raw_sd = {k[len('module.'):]: v for k, v in raw_sd.items()}

    def remap_keys(sd):
        """
        Make training-time keys compatible with this script's model definition.
        e.g., 'net.layer1.0.conv1.weight' -> 'backbone.layer1.0.conv1.weight'
        """
        new_sd = {}
        for k, v in sd.items():
            nk = k
            # common wrappers from training scripts
            if nk.startswith('model.'):
                nk = nk[len('model.'):]
            if nk.startswith('backbone.'):  # already correct
                pass
            elif nk.startswith('net.'):
                nk = 'backbone.' + nk[len('net.'):]
            elif nk.startswith('resnet.') or nk.startswith('encoder.'):
                nk = 'backbone.' + nk.split('.', 1)[1]
            elif nk.startswith('fc.'):
                nk = 'backbone.' + nk
            # else: leave as-is; most torchvision weights are under backbone.*
            new_sd[nk] = v
        return new_sd

    state_dict = remap_keys(raw_sd)

    # Try strict load first; if it fails, fall back to strict=False and report missing/unexpected keys
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        # torch>=2.0 returns IncompatibleKeys
        if (hasattr(missing, 'missing_keys') and (missing.missing_keys or missing.unexpected_keys)):
            print(f"[warn] Strict load reported issues. Missing: {getattr(missing, 'missing_keys', [])}, Unexpected: {getattr(missing, 'unexpected_keys', [])}")
    except RuntimeError as e:
        print("[warn] Strict load failed, trying non-strict mapping...")
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(f"[warn] Non-strict load. Missing keys: {incompatible.missing_keys}; Unexpected keys: {incompatible.unexpected_keys}")

    print(f" Model loaded from {MODEL_PATH}")

    labels_dict = load_test_labels(TEST_LABELS_CSV)
    print(f" Loaded labels for {len(labels_dict)} cases from {TEST_LABELS_CSV}")
    if len(labels_dict) > 0:
        print(" Example label IDs:", list(labels_dict.keys())[:3])

    transform = build_transform(IMG_SIZE)

    img_items = list_test_images(TEST_IMG_DIR)
    print(f"Found {len(img_items)} PNG files in {TEST_IMG_DIR}")

    test_items = []
    mismatched = 0
    unmatched_samples = []
    for img_path, case_id_norm, orig_stem in img_items:
        key = _norm(case_id_norm)
        if key in labels_dict:
            labels32 = labels_dict[key]
            test_items.append((img_path, key, labels32))
        else:
            mismatched += 1
            if len(unmatched_samples) < 5:
                unmatched_samples.append((orig_stem, case_id_norm))
    if mismatched:
        print(f" [debug] Unmatched samples (png stem -> normalized) examples: {unmatched_samples}")
    print(f" Prepared {len(test_items)} matching samples for testing. Skipped {mismatched} without labels.")
    if not test_items:
        print(" No matching samples found. Exiting."); return

    preds_teeth, targets_teeth, preds_jaw, targets_jaw, ids = test_model(model, test_items, device, transform)
    print(f" Inference complete on {len(ids)} samples.")

    metrics = calculate_metrics(preds_teeth, targets_teeth, preds_jaw, targets_jaw)
    print_metrics_summary(metrics)
    if 'jaw_metrics' in metrics:
        jm = metrics['jaw_metrics']
        print(f"\n JAW CLASSIFICATION METRICS:\n  Accuracy: {jm['accuracy']:.4f}\n  Precision: {jm['precision']:.4f}\n  Recall: {jm['recall']:.4f}\n  F1 Score: {jm['f1']:.4f}")

    print(f"\n Missing counts per position (1..16): {metrics.get('missing_counts_per_position', [])}")
    if 'missing_counts_per_fdi' in metrics:
        print(" Missing counts per FDI tooth:", metrics['missing_counts_per_fdi'])

    print_sample_predictions(ids, preds_teeth, targets_teeth, num_samples=NUM_SAMPLE_PREDICTIONS)

    print("\n" + "="*80 + "\n" + " "*28 + "GENERATING PLOTS" + "\n" + "="*80)
    generate_test_plots(metrics, preds_teeth, targets_teeth, OUTPUT_DIR, targets_jaw)

    results = pd.DataFrame({'case_id': ids, 'jaw_target': targets_jaw, 'jaw_prob': preds_jaw, 'jaw_pred': (preds_jaw>0.5).astype(int)})
    for pos in range(NUM_TEETH):
        results[f'true_pos{pos+1}'] = targets_teeth[:, pos].astype(int)
        results[f'pred_pos{pos+1}'] = (preds_teeth[:, pos] > 0.5).astype(int)
        results[f'prob_pos{pos+1}'] = preds_teeth[:, pos]
    results.to_csv(Path(OUTPUT_DIR) / 'test_predictions_detailed.csv', index=False)

    # ---- Extra: per-jaw per-position CSVs with correct FDI labels ----
    # Build per-jaw views
    ids_arr = np.array(ids)
    jaw_vec = np.array([infer_jaw_from_case_id(cid) for cid in ids])
    for jaw_label, tag, fdi_order in [
        (JAW_UPPER_LABEL, 'upper_test', UPPER_FDI_ORDER),
        (JAW_LOWER_LABEL, 'lower_test', LOWER_FDI_ORDER),
    ]:
        mask = (jaw_vec == jaw_label)
        if mask.sum() == 0:
            print(f"[info] No samples for jaw={tag} in test set; skipping CSV export.")
            continue
        sub_ids = ids_arr[mask]
        sub_targets = targets_teeth[mask]
        sub_preds = preds_teeth[mask]
        rows = []
        for i in range(sub_targets.shape[0]):
            row = {'case_id': sub_ids[i]}
            # Map back to FDI labels
            for pos in range(NUM_TEETH):
                row[f'true_{fdi_order[pos]}'] = int(sub_targets[i, pos])
                row[f'pred_{fdi_order[pos]}'] = int(sub_preds[i, pos] > 0.5)
                row[f'prob_{fdi_order[pos]}'] = float(sub_preds[i, pos])
            rows.append(row)
        df_jaw = pd.DataFrame(rows)
        out_fp = Path(OUTPUT_DIR) / f'per_tooth_{tag}.csv'
        df_jaw.to_csv(out_fp, index=False)
        print(f"✓ Saved per-jaw per-tooth CSV -> {out_fp}")

    with open(Path(OUTPUT_DIR) / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n Full results and metrics saved successfully to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
