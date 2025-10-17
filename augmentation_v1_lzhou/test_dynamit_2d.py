import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import random
from collections import OrderedDict

# Plotting and metrics libraries
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for servers
import seaborn as sns

# Scikit-learn for metrics
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

# 2D image libs
from PIL import Image
from torchvision import transforms, models
import re

# =================================================================================
# CONFIGURATION
# =================================================================================
# 2D PNG root
TEST_IMG_DIR = "/home/user/lzhou/week9/Clean_Test_Data"

# Flipped labels CSV (1 = missing, 0 = present)
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"

# Path to the model trained with Dynamit Loss (2D)
MODEL_PATH = "/home/user/lzhou/week10/output/Train2D/dynamit/dynamit_loss_best_2d.pth"

# Output directory for reports
OUTPUT_DIR = "/home/user/lzhou/week10/output/Test2D/dynamit"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 5
IMG_SIZE = 320

# --- FDI Notation Mapping (Must be identical to the training script) ---
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}

# =================================================================================
# MODEL (2D ResNet multi-label)
# =================================================================================
class ResNetMultiLabel(nn.Module):
    def __init__(self, backbone="resnet18", num_teeth=32, pretrained=False):
        super().__init__()
        bb = (backbone or "resnet18").lower()
        if bb == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_teeth)
        self.net = net

    def forward(self, x):
        return self.net(x)

# =================================================================================
# HELPERS: ID normalization & IO
# =================================================================================
def _norm(s: str) -> str:
    return str(s).strip().lower()

def normalize_png_stem_to_newid(stem: str) -> str:
    """
    Convert file stem like:
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

def build_transform(img_size=320):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_image_tensor(img_path, transform):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)  # add batch dim

def load_test_labels(csv_path):
    """
    Loads ground truth labels into a dict keyed by normalized case_id.
    Accepts either 'new_id' or 'filename' as the identifier column.
    """
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    if 'new_id' in cols_lower:
        id_col = cols_lower['new_id']
        df[id_col] = df[id_col].astype(str).str.strip().str.lower()
    elif 'filename' in cols_lower:
        id_col = cols_lower['filename']
        df[id_col] = df[id_col].astype(str).str.strip()
    else:
        raise ValueError("Identifier column not found. Expecting 'new_id' or 'filename' in CSV.")
    df.columns = [str(c) if str(c).isdigit() else c for c in df.columns]
    labels_dict = {}
    for _, row in df.iterrows():
        if id_col.lower() == 'new_id':
            case_key = _norm(row[id_col])
        else:
            raw_stem = str(row[id_col]).replace('.png', '').replace('.ply', '')
            case_key = normalize_png_stem_to_newid(raw_stem)
        label_vector = np.zeros(NUM_TEETH, dtype=np.float32)
        for tooth_fdi in VALID_FDI_LABELS:
            key = str(tooth_fdi)
            if key in df.columns and pd.notna(row[key]) and int(row[key]) == 1:
                label_vector[FDI_TO_INDEX[tooth_fdi]] = 1.0
        labels_dict[case_key] = label_vector
    return labels_dict

# =================================================================================
# INFERENCE AND METRICS
# =================================================================================
def test_model(model, test_items, device, transform):
    model.eval(); all_preds, all_targets, all_ids = [], [], []
    with torch.no_grad():
        for img_path, case_id, labels in tqdm(test_items, desc="Testing"):
            x = load_image_tensor(img_path, transform).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy()[0])
            all_targets.append(labels)
            all_ids.append(case_id)
    return np.array(all_preds), np.array(all_targets), all_ids

def calculate_metrics(preds, targets):
    preds_bin = (preds > 0.5).astype(int); targets_bin = targets.astype(int)
    flat_preds, flat_targets = preds_bin.flatten(), targets_bin.flatten()
    p, r, f1, _ = precision_recall_fscore_support(flat_targets, flat_preds, average='binary', zero_division=0)
    acc = accuracy_score(flat_targets, flat_preds); per_tooth = OrderedDict()
    for idx in range(NUM_TEETH):
        fdi_label = INDEX_TO_FDI[idx]
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(targets_bin[:, idx], preds_bin[:, idx], average='binary', zero_division=0)
        per_tooth[fdi_label] = {'precision': float(p_t), 'recall': float(r_t), 'f1': float(f1_t),
                                'accuracy': accuracy_score(targets_bin[:, idx], preds_bin[:, idx]),
                                'support': int(targets_bin[:, idx].sum())}
    macro_p = np.mean([m['precision'] for m in per_tooth.values()])
    macro_r = np.mean([m['recall'] for m in per_tooth.values()])
    macro_f1 = np.mean([m['f1'] for m in per_tooth.values()])
    macro_acc = np.mean([m['accuracy'] for m in per_tooth.values()])
    return {'overall_micro': {'precision': float(p), 'recall': float(r), 'f1': float(f1), 'accuracy': float(acc)},
            'overall_macro': {'macro_precision': macro_p, 'macro_recall': macro_r, 'macro_f1': macro_f1, 'macro_accuracy': macro_acc},
            'per_tooth': per_tooth}

# =================================================================================
# REPORTING AND VISUALIZATION
# =================================================================================
def print_metrics_summary(metrics):
    print("\n" + "="*80 + "\n" + " "*25 + "TESTING METRICS SUMMARY" + "\n" + "="*80)
    micro = metrics['overall_micro']
    print("\n OVERALL (MICRO-AVERAGE) METRICS (Positive class: Missing Tooth):")
    print(f"  - Precision: {micro['precision']:.4f}\n  - Recall:    {micro['recall']:.4f}\n  - F1 Score:  {micro['f1']:.4f}\n  - Accuracy:  {micro['accuracy']:.4f}")
    macro = metrics['overall_macro']
    print("\n OVERALL (MACRO-AVERAGE) METRICS:")
    print(f"  - Macro Precision: {macro['macro_precision']:.4f}\n  - Macro Recall:    {macro['macro_recall']:.4f}\n  - Macro F1 Score:  {macro['macro_f1']:.4f}\n  - Macro Accuracy:  {macro['macro_accuracy']:.4f}")
    print("\n PER-TOOTH METRICS (FDI Notation):"); print("-" * 80)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}"); print("-" * 80)
    for fdi_label, m in metrics['per_tooth'].items():
        print(f"Tooth {fdi_label:<5} {m['precision']:>10.4f}   {m['recall']:>10.4f}   {m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}")
    print("=" * 80)

def print_sample_predictions(ids, preds, targets, num_samples):
    """Print a clear comparison for random samples."""
    print("\n" + "="*80 + "\n" + " "*28 + "SAMPLE PREDICTIONS" + "\n" + "="*80)
    if len(ids) < num_samples: num_samples = len(ids)
    sample_indices = random.sample(range(len(ids)), num_samples)
    for i in sample_indices:
        case_id = ids[i]
        target_labels = targets[i]
        pred_labels = (preds[i] > 0.5).astype(int)
        truth_missing = {INDEX_TO_FDI[j] for j, label in enumerate(target_labels) if label == 1}
        pred_missing = {INDEX_TO_FDI[j] for j, label in enumerate(pred_labels) if label == 1}
        correctly_found_missing = sorted(list(truth_missing.intersection(pred_missing)))
        missed_missing_teeth = sorted(list(truth_missing.difference(pred_missing)))      # FN
        wrongly_predicted_missing = sorted(list(pred_missing.difference(truth_missing))) # FP
        print(f"\n Case ID: {case_id}\n" + "-"*50)
        print(f"  Ground Truth (Missing): {sorted(list(truth_missing)) if truth_missing else 'None'}")
        print(f"  Prediction (Missing):   {sorted(list(pred_missing)) if pred_missing else 'None'}")
        print("-"*50)
        print(f"   Correctly Found (TP): {correctly_found_missing or 'None'}")
        print(f"   Missed Teeth (FN):    {missed_missing_teeth or 'None'}")
        print(f"   False Alarms (FP):    {wrongly_predicted_missing or 'None'}")
    print("\n" + "="*80)

def generate_test_plots(metrics, preds, targets, save_dir):
    """Generate and save plots summarizing test performance."""
    per_tooth_metrics = metrics['per_tooth']
    fDI = [str(label) for label in per_tooth_metrics.keys()]
    f1_scores = [m['f1'] for m in per_tooth_metrics.values()]
    precision_scores = [m['precision'] for m in per_tooth_metrics.values()]
    recall_scores = [m['recall'] for m in per_tooth_metrics.values()]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(fDI)); width = 0.25
    ax.bar(x - width, precision_scores, width, label='Precision', color='royalblue')
    ax.bar(x, recall_scores, width, label='Recall', color='limegreen')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='tomato')
    ax.set_ylabel('Scores', fontsize=14); ax.set_title('Per-Tooth Performance (Positive Class: Missing)', fontsize=18, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(fDI, rotation=45, ha='right'); ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05); ax.bar_label(rects3, padding=3, fmt='%.2f', fontsize=8); fig.tight_layout()
    plot_path = Path(save_dir) / "per_tooth_metrics.png"; plt.savefig(plot_path, dpi=300); plt.close()
    print(f"✓ Per-tooth metrics plot saved to {plot_path}")

    flat_targets = targets.flatten(); flat_probs = preds.flatten()
    flat_preds = (preds > 0.5).astype(int).flatten()
    cm = confusion_matrix(flat_targets, flat_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Present (0)', 'Pred Missing (1)'],
                yticklabels=['Actual Present (0)', 'Actual Missing (1)'])
    plt.title('Overall Confusion Matrix (All Teeth)', fontsize=16, fontweight='bold'); plt.ylabel('Ground Truth'); plt.xlabel('Prediction')
    plot_path = Path(save_dir) / "confusion_matrix.png"; plt.savefig(plot_path, dpi=300); plt.close()
    print(f" Confusion matrix plot saved to {plot_path}")

    precision, recall, _ = precision_recall_curve(flat_targets, flat_probs)
    avg_precision = average_precision_score(flat_targets, flat_probs)
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, lw=2, label=f'PR Curve (Avg Precision = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Overall Precision-Recall Curve', fontsize=16, fontweight='bold'); plt.legend(loc="lower left"); plt.grid(True, alpha=0.5)
    plot_path = Path(save_dir) / "precision_recall_curve.png"; plt.savefig(plot_path, dpi=300); plt.close()
    print(f" Precision-Recall curve saved to {plot_path}")

# =================================================================================
# MAIN
# =================================================================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model and load weights
    model = ResNetMultiLabel(num_teeth=NUM_TEETH).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    raw_sd = checkpoint.get('model_state_dict', checkpoint)
    # Strip DDP prefix
    if len(raw_sd) > 0 and next(iter(raw_sd)).startswith('module.'):
        raw_sd = {k[len('module.'):]: v for k, v in raw_sd.items()}

    # Remap common prefixes to this script's structure (self.net.*)
    def remap(sd):
        new = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith('model.'):
                nk = nk[len('model.'):]
            if nk.startswith('backbone.'):
                nk = 'net.' + nk[len('backbone.'):]
            elif nk.startswith('resnet.'):
                nk = 'net.' + nk[len('resnet.'):]
            # if already 'net.' or matches torchvision keys under net.*, keep as-is
            new[nk] = v
        return new

    state_dict = remap(raw_sd)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print("[warn] Strict load failed, trying non-strict...", e)
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(f"[warn] Non-strict load. Missing: {incompatible.missing_keys}, Unexpected: {incompatible.unexpected_keys}")
    print(f" Model loaded from {MODEL_PATH}")

    # Load labels
    labels_dict = load_test_labels(TEST_LABELS_CSV)
    print(f" Loaded labels for {len(labels_dict)} cases from {TEST_LABELS_CSV}")
    if len(labels_dict) > 0:
        print(" Example label IDs:", list(labels_dict.keys())[:3])

    # Build test items from PNGs
    transform = build_transform(IMG_SIZE)
    img_items = list_test_images(TEST_IMG_DIR)
    print(f" Found {len(img_items)} PNG files in {TEST_IMG_DIR}")

    test_items, mismatched = [], 0
    unmatched_samples = []
    for img_path, case_id_norm, orig_stem in img_items:
        key = _norm(case_id_norm)
        if key in labels_dict:
            labels = labels_dict[key]
            test_items.append((img_path, key, labels))
        else:
            mismatched += 1
            if len(unmatched_samples) < 5:
                unmatched_samples.append((orig_stem, case_id_norm))
    print(f" Prepared {len(test_items)} matching samples for testing. Skipped {mismatched} without labels.")
    if mismatched:
        print(f" [debug] Unmatched samples (png stem -> normalized) examples: {unmatched_samples}")
    if not test_items:
        print(" No matching samples found. Exiting."); return

    # Inference
    preds, targets, ids = test_model(model, test_items, device, transform)
    print(f" Inference complete on {len(ids)} samples.")

    # Metrics & reporting
    metrics = calculate_metrics(preds, targets)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, num_samples=NUM_SAMPLE_PREDICTIONS)

    # Plots
    print("\n" + "="*80 + "\n" + " "*28 + "GENERATING PLOTS" + "\n" + "="*80)
    generate_test_plots(metrics, preds, targets, OUTPUT_DIR)

    # Save results
    results_df = pd.DataFrame({'case_id': ids})
    for fdi_label in VALID_FDI_LABELS:
        idx = FDI_TO_INDEX[fdi_label]
        results_df[f'true_{fdi_label}'] = targets[:, idx]
        results_df[f'pred_{fdi_label}'] = (preds[:, idx] > 0.5).astype(int)
        results_df[f'prob_{fdi_label}'] = preds[:, idx]
    results_df.to_csv(Path(OUTPUT_DIR) / 'test_predictions_dynamit.csv', index=False)

    with open(Path(OUTPUT_DIR) / 'test_metrics_dynamit.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n Full results and metrics saved successfully to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()