"""
test_baseline_2d_fixed.py

配套 baseline_model_2d_fixed.py 使用的测试代码

核心逻辑：
- Upper jaw 图像 → 只评估 11-28 的预测
- Lower jaw 图像 → 只评估 31-48 的预测
- 和训练时的 Masked Loss 逻辑一致
"""

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
TEST_IMG_DIR    = "/home/user/lzhou/week15/render_output/test"
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"
# 使用修改后训练的模型
MODEL_PATH      = "/home/user/lzhou/week15-32/output/Train2D/fixed/resnet18_bce_best_2d_fixed.pth"
OUTPUT_DIR      = "/home/user/lzhou/week15-32/output/Test2D/fixed"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 5
IMG_SIZE = 320

# Multi-view settings
MULTIVIEW_STRATEGY = "max_per_tooth"
MAX_PER_TOOTH_TOPK = 2

# --- FDI Notation Mapping ---
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}

# Upper/Lower teeth
UPPER_FDI = sorted([11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28])
LOWER_FDI = sorted([31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48])
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
    left = left.strip().replace('-', '_').replace(' ', '_')
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
# MODEL DEFINITION
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
# DATA LOADING
# =================================================================================
def build_transform(img_size=320):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_image_tensor(img_path: Path, transform):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def group_images_by_case(img_dir: str):
    groups = {}
    for p in Path(img_dir).rglob("*.png"):
        case_id_norm, angle_deg, jaw_type = parse_case_id_and_angle(p)
        groups.setdefault(case_id_norm, []).append((p, angle_deg, jaw_type))
    return groups


def load_test_labels(csv_path):
    """加载CSV标签"""
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    if 'new_id' not in cols_lower:
        raise ValueError("CSV must contain a 'new_id' column.")
    id_col = cols_lower['new_id']
    df[id_col] = df[id_col].astype(str).str.strip()
    df.columns = [str(c) if str(c).isdigit() else c for c in df.columns]
    
    labels_dict = {}
    for _, row in df.iterrows():
        case_id = str(row[id_col]).strip().lower().replace('-', '_')
        label_vector = np.zeros(NUM_TEETH, dtype=np.float32)
        for tooth_fdi in VALID_FDI_LABELS:
            tooth_str = str(tooth_fdi)
            if tooth_str in df.columns and pd.notna(row[tooth_str]) and int(row[tooth_str]) == 1:
                label_vector[FDI_TO_INDEX[tooth_fdi]] = 1.0
        labels_dict[case_id] = label_vector
    return labels_dict


# =================================================================================
# JAW-AWARE HELPERS
# =================================================================================
def get_relevant_indices(jaw_type):
    if jaw_type == 'upper':
        return UPPER_IDX
    elif jaw_type == 'lower':
        return LOWER_IDX
    else:
        return list(range(NUM_TEETH))


def get_relevant_fdi(jaw_type):
    if jaw_type == 'upper':
        return UPPER_FDI
    elif jaw_type == 'lower':
        return LOWER_FDI
    else:
        return VALID_FDI_LABELS


# =================================================================================
# INFERENCE
# =================================================================================
def compute_jaw_confidence(probs: np.ndarray, jaw_type: str) -> float:
    if jaw_type == 'upper':
        jaw_probs = probs[UPPER_IDX]
    elif jaw_type == 'lower':
        jaw_probs = probs[LOWER_IDX]
    else:
        jaw_probs = probs
    return np.mean(np.abs(jaw_probs - 0.5))


def test_model_multiview(model, grouped_imgs, labels_dict, device, transform):
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []
    all_jaw_types = []
    
    matched = 0
    unmatched = []

    for case_id, views in sorted(grouped_imgs.items()):
        if len(views) == 0:
            continue
        
        if case_id not in labels_dict:
            unmatched.append(case_id)
            continue
        
        gt_labels = labels_dict[case_id]
        matched += 1

        probs_list = []
        jaw_conf_list = []
        jaw_type_list = []
        
        for img_path, angle_deg, jaw_type in views:
            x = load_image_tensor(img_path, transform).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            probs_list.append(probs)
            jaw_conf_list.append(compute_jaw_confidence(probs, jaw_type))
            jaw_type_list.append(jaw_type)

        probs_arr = np.stack(probs_list, axis=0)
        jaw_conf_arr = np.array(jaw_conf_list)
        case_jaw_type = max(set(jaw_type_list), key=jaw_type_list.count)

        if MULTIVIEW_STRATEGY == "max_per_tooth":
            K = min(MAX_PER_TOOTH_TOPK, len(views))
            topk_idx = np.argsort(jaw_conf_arr)[-K:]
            selected_probs = probs_arr[topk_idx]
            final_probs = selected_probs.max(axis=0)
        elif MULTIVIEW_STRATEGY == "avg_all":
            final_probs = probs_arr.mean(axis=0)
        else:
            best_idx = np.argmax(jaw_conf_arr)
            final_probs = probs_arr[best_idx]

        all_preds.append(final_probs)
        all_targets.append(gt_labels)
        all_ids.append(case_id)
        all_jaw_types.append(case_jaw_type)

    print(f"✓ Matched: {matched}, Unmatched: {len(unmatched)}")
    if unmatched and len(unmatched) <= 5:
        print(f"  Unmatched: {unmatched}")

    if matched == 0:
        return None, None, None, None

    return np.array(all_preds), np.array(all_targets), all_ids, all_jaw_types


# =================================================================================
# JAW-AWARE METRICS
# =================================================================================
def calculate_metrics_jaw_aware(preds, targets, jaw_types):
    if len(preds) == 0:
        return {}
    
    all_preds_flat = []
    all_targets_flat = []
    
    for i in range(len(preds)):
        relevant_idx = get_relevant_indices(jaw_types[i])
        pred_bin = (preds[i, relevant_idx] > 0.5).astype(int)
        target_bin = targets[i, relevant_idx].astype(int)
        all_preds_flat.extend(pred_bin)
        all_targets_flat.extend(target_bin)
    
    all_preds_flat = np.array(all_preds_flat)
    all_targets_flat = np.array(all_targets_flat)
    
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        all_targets_flat, all_preds_flat, average='binary', zero_division=0
    )
    acc_micro = accuracy_score(all_targets_flat, all_preds_flat)

    per_tooth = OrderedDict()
    for fdi in VALID_FDI_LABELS:
        idx = FDI_TO_INDEX[fdi]
        tooth_preds = []
        tooth_targets = []
        
        for i in range(len(preds)):
            relevant_idx = get_relevant_indices(jaw_types[i])
            if idx in relevant_idx:
                tooth_preds.append((preds[i, idx] > 0.5).astype(int))
                tooth_targets.append(targets[i, idx].astype(int))
        
        if len(tooth_preds) > 0:
            tooth_preds = np.array(tooth_preds)
            tooth_targets = np.array(tooth_targets)
            p_t, r_t, f1_t, _ = precision_recall_fscore_support(
                tooth_targets, tooth_preds, average='binary', zero_division=0
            )
            acc_t = accuracy_score(tooth_targets, tooth_preds)
            support = int(tooth_targets.sum())
            n_samples = len(tooth_targets)
        else:
            p_t, r_t, f1_t, acc_t = 0.0, 0.0, 0.0, 0.0
            support, n_samples = 0, 0
        
        per_tooth[fdi] = {
            'precision': float(p_t), 'recall': float(r_t), 'f1': float(f1_t),
            'accuracy': float(acc_t), 'support': support, 'n_samples': n_samples
        }

    valid_metrics = [m for m in per_tooth.values() if m['n_samples'] > 0]
    macro_p = np.mean([m['precision'] for m in valid_metrics]) if valid_metrics else 0.0
    macro_r = np.mean([m['recall'] for m in valid_metrics]) if valid_metrics else 0.0
    macro_f1 = np.mean([m['f1'] for m in valid_metrics]) if valid_metrics else 0.0
    macro_acc = np.mean([m['accuracy'] for m in valid_metrics]) if valid_metrics else 0.0

    return {
        'overall_micro': {
            'precision': float(p_micro), 'recall': float(r_micro),
            'f1': float(f1_micro), 'accuracy': float(acc_micro)
        },
        'overall_macro': {
            'macro_precision': macro_p, 'macro_recall': macro_r,
            'macro_f1': macro_f1, 'macro_accuracy': macro_acc
        },
        'per_tooth': per_tooth
    }


# =================================================================================
# REPORTING
# =================================================================================
def print_metrics_summary(metrics):
    print("\n" + "="*80)
    print(" "*20 + "TESTING METRICS SUMMARY (JAW-AWARE)")
    print("="*80)
    
    micro = metrics['overall_micro']
    print("\n OVERALL (MICRO-AVERAGE) METRICS:")
    print(f"  - Precision: {micro['precision']:.4f}")
    print(f"  - Recall:    {micro['recall']:.4f}")
    print(f"  - F1 Score:  {micro['f1']:.4f}")
    print(f"  - Accuracy:  {micro['accuracy']:.4f}")
    
    macro = metrics['overall_macro']
    print("\n OVERALL (MACRO-AVERAGE) METRICS:")
    print(f"  - Macro Precision: {macro['macro_precision']:.4f}")
    print(f"  - Macro Recall:    {macro['macro_recall']:.4f}")
    print(f"  - Macro F1 Score:  {macro['macro_f1']:.4f}")
    print(f"  - Macro Accuracy:  {macro['macro_accuracy']:.4f}")
    
    print("\n PER-TOOTH METRICS (FDI Notation):")
    print("-"*90)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10} {'Samples':<10}")
    print("-"*90)
    for fdi_label, m in metrics['per_tooth'].items():
        if m['n_samples'] > 0:
            print(f"Tooth {fdi_label:<5} {m['precision']:>10.4f}   {m['recall']:>10.4f}   "
                  f"{m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}   {m['n_samples']:>8}")
    print("="*90)


def print_sample_predictions(ids, preds, targets, jaw_types, num_samples):
    print("\n" + "="*80)
    print(" "*25 + "SAMPLE PREDICTIONS (JAW-AWARE)")
    print("="*80)
    
    if len(ids) < num_samples:
        num_samples = len(ids)
    
    sample_indices = random.sample(range(len(ids)), num_samples)
    
    for i in sample_indices:
        case_id = ids[i]
        jaw_type = jaw_types[i]
        relevant_idx = get_relevant_indices(jaw_type)
        relevant_fdi = get_relevant_fdi(jaw_type)
        
        truth_missing = {INDEX_TO_FDI[j] for j in relevant_idx if targets[i, j] == 1}
        pred_missing = {INDEX_TO_FDI[j] for j in relevant_idx if preds[i, j] > 0.5}
        
        correctly_found = sorted(list(truth_missing & pred_missing))
        missed = sorted(list(truth_missing - pred_missing))
        false_alarms = sorted(list(pred_missing - truth_missing))
        
        print(f"\n Case ID: {case_id} (Jaw: {jaw_type.upper()})")
        print(f"   Evaluating: {relevant_fdi[0]}-{relevant_fdi[-1]} ({len(relevant_fdi)} teeth)")
        print("-"*60)
        print(f"  Ground Truth (Missing): {sorted(list(truth_missing)) if truth_missing else 'None'}")
        print(f"  Prediction (Missing):   {sorted(list(pred_missing)) if pred_missing else 'None'}")
        print("-"*60)
        print(f"   Correctly Found (TP): {correctly_found or 'None'}")
        print(f"   Missed Teeth (FN):    {missed or 'None'}")
        print(f"   False Alarms (FP):    {false_alarms or 'None'}")
    
    print("\n" + "="*80)


def generate_test_plots(metrics, preds, targets, jaw_types, save_dir):
    per_tooth = metrics['per_tooth']
    
    for jaw, fdi_list in [('Upper', UPPER_FDI), ('Lower', LOWER_FDI)]:
        fdi_labels = [str(f) for f in fdi_list]
        f1_scores = [per_tooth[f]['f1'] for f in fdi_list]
        precision_scores = [per_tooth[f]['precision'] for f in fdi_list]
        recall_scores = [per_tooth[f]['recall'] for f in fdi_list]

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 8))
        x = np.arange(len(fdi_labels))
        width = 0.25
        ax.bar(x - width, precision_scores, width, label='Precision', color='royalblue')
        ax.bar(x, recall_scores, width, label='Recall', color='limegreen')
        rects = ax.bar(x + width, f1_scores, width, label='F1 Score', color='tomato')
        ax.set_ylabel('Scores', fontsize=14)
        ax.set_title(f'{jaw} Teeth Performance (JAW-AWARE)', fontsize=18, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(fdi_labels, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)
        fig.tight_layout()
        plt.savefig(Path(save_dir) / f"per_tooth_metrics_{jaw.lower()}.png", dpi=300)
        plt.close()
        print(f"✓ {jaw} teeth plot saved")

    all_preds_flat = []
    all_targets_flat = []
    for i in range(len(preds)):
        relevant_idx = get_relevant_indices(jaw_types[i])
        all_preds_flat.extend((preds[i, relevant_idx] > 0.5).astype(int))
        all_targets_flat.extend(targets[i, relevant_idx].astype(int))
    
    cm = confusion_matrix(all_targets_flat, all_preds_flat)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Present (0)', 'Pred Missing (1)'],
                yticklabels=['Actual Present (0)', 'Actual Missing (1)'])
    plt.title('Confusion Matrix (JAW-AWARE)', fontsize=16, fontweight='bold')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.savefig(Path(save_dir) / "confusion_matrix.png", dpi=300)
    plt.close()
    print(f"✓ Confusion matrix saved")

    all_probs_flat = []
    for i in range(len(preds)):
        relevant_idx = get_relevant_indices(jaw_types[i])
        all_probs_flat.extend(preds[i, relevant_idx])
    
    precision_curve, recall_curve, _ = precision_recall_curve(all_targets_flat, all_probs_flat)
    avg_precision = average_precision_score(all_targets_flat, all_probs_flat)
    
    plt.figure(figsize=(10, 7))
    plt.plot(recall_curve, precision_curve, color='darkorange', lw=2,
             label=f'PR Curve (AP = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (JAW-AWARE)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.5)
    plt.savefig(Path(save_dir) / "precision_recall_curve.png", dpi=300)
    plt.close()
    print(f"✓ PR curve saved")


# =================================================================================
# MAIN
# =================================================================================
def main():
    print("="*80)
    print(" "*15 + "2D TESTING (JAW-AWARE - Matching Fixed Training)")
    print("="*80)
    
    if torch.cuda.is_available():
        available_gpus = get_free_gpus(threshold_mb=1000, max_gpus=2)
        device = torch.device(f"cuda:{available_gpus[0]}" if available_gpus else "cuda:0")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = ResNetMultiLabel(num_teeth=NUM_TEETH).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    raw_sd = checkpoint.get('model_state_dict', checkpoint)
    
    if len(raw_sd) > 0 and next(iter(raw_sd)).startswith('module.'):
        raw_sd = {k[len('module.'):]: v for k, v in raw_sd.items()}
    
    def remap_keys(sd):
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith('net.'):
                nk = 'backbone.' + nk[len('net.'):]
            new_sd[nk] = v
        return new_sd
    
    state_dict = remap_keys(raw_sd)
    model.load_state_dict(state_dict, strict=False)
    print(f" Model loaded from {MODEL_PATH}")

    labels_dict = load_test_labels(TEST_LABELS_CSV)
    print(f" Loaded labels for {len(labels_dict)} cases")
    
    grouped_imgs = group_images_by_case(TEST_IMG_DIR)
    print(f" Found {len(grouped_imgs)} case_id(s)")

    csv_ids = set(labels_dict.keys())
    img_ids = set(grouped_imgs.keys())
    common = csv_ids & img_ids
    print(f"\n  CSV IDs: {len(csv_ids)}, Image IDs: {len(img_ids)}, Intersection: {len(common)}")
    
    print(f"\n  Sample CSV IDs: {sorted(list(csv_ids))[:3]}")
    print(f"  Sample Image IDs: {sorted(list(img_ids))[:3]}")

    transform = build_transform(IMG_SIZE)
    result = test_model_multiview(model, grouped_imgs, labels_dict, device, transform)
    
    if result[0] is None:
        print(" No matching samples. Check ID formats.")
        return
    
    preds, targets, ids, jaw_types = result
    
    upper_count = sum(1 for j in jaw_types if j == 'upper')
    lower_count = sum(1 for j in jaw_types if j == 'lower')
    print(f" Inference complete: {len(ids)} samples (Upper: {upper_count}, Lower: {lower_count})")

    metrics = calculate_metrics_jaw_aware(preds, targets, jaw_types)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, jaw_types, NUM_SAMPLE_PREDICTIONS)

    print("\n" + "="*80 + "\n" + " "*28 + "GENERATING PLOTS" + "\n" + "="*80)
    generate_test_plots(metrics, preds, targets, jaw_types, OUTPUT_DIR)

    results = pd.DataFrame({
        'case_id': ids,
        'jaw_type': jaw_types
    })
    for fdi in VALID_FDI_LABELS:
        idx = FDI_TO_INDEX[fdi]
        results[f'true_{fdi}'] = targets[:, idx].astype(int)
        results[f'pred_{fdi}'] = (preds[:, idx] > 0.5).astype(int)
        results[f'prob_{fdi}'] = preds[:, idx]
    results.to_csv(Path(OUTPUT_DIR) / 'test_predictions_detailed.csv', index=False)

    with open(Path(OUTPUT_DIR) / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n Results saved to {OUTPUT_DIR}")
    
    print("\n" + "="*80)
    print(" FINAL SUMMARY (JAW-AWARE):")
    print(f"  Micro F1:  {metrics['overall_micro']['f1']:.4f}")
    print(f"  Macro F1:  {metrics['overall_macro']['macro_f1']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()