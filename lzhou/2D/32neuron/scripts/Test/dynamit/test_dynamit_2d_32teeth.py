import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import random
import re
from collections import OrderedDict

# Plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Metrics
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score
)

# Image processing
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet50

# =================================================================================
# CONFIGURATION
# =================================================================================
TEST_IMG_DIR = "/home/user/lzhou/week15/render_output/test"
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"
MODEL_PATH = "/home/user/lzhou/week15-32/output/Train2D/32teeth_dynamit/dynamit_best_2d_32teeth.pth"
OUTPUT_DIR = "/home/user/lzhou/week16-32/output/Test2D/32teeth_dynamit_auto_best"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ========== HYPERPARAMETERS ==========
IMG_SIZE = 256  
NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 10
DROPOUT_RATE = 0.5  

# ========== FUSION STRATEGY SELECTION ==========
# 'balanced_accuracy', 'macro_recall', 'macro_f1', 'macro_precision'
SELECTION_METRIC = 'macro_f1'

# For best_n_angles strategy
BEST_N = 2
ACTIVE_STRATEGIES = ["best_n_angles"]

#  best_n_angles strategy
FIXED_STRATEGY = 'best_n_angles'

# FDI Notation
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}

# Upper/Lower teeth FDI codes
UPPER_FDI = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

# Upper/Lower indices
UPPER_INDICES = [FDI_TO_INDEX[fdi] for fdi in UPPER_FDI]
LOWER_INDICES = [FDI_TO_INDEX[fdi] for fdi in LOWER_FDI]

# =================================================================================
# FUSION STRATEGIES
# =================================================================================
def fuse_average(probs_list):
    """Simple average of all angles"""
    return np.mean(probs_list, axis=0)

def fuse_max_confidence(probs_list):
    """Per-tooth maximum confidence selection"""
    probs_array = np.array(probs_list)
    result = np.zeros(NUM_TEETH)
    for tooth_idx in range(NUM_TEETH):
        tooth_probs = probs_array[:, tooth_idx]
        confidences = np.abs(tooth_probs - 0.5)
        best_angle_idx = np.argmax(confidences)
        result[tooth_idx] = tooth_probs[best_angle_idx]
    return result

def fuse_best_angle(probs_list):
    """Select single best angle based on overall confidence"""
    probs_array = np.array(probs_list)
    angle_confidences = []
    for angle_idx in range(len(probs_list)):
        conf = np.mean(np.abs(probs_array[angle_idx] - 0.5))
        angle_confidences.append(conf)
    best_angle_idx = np.argmax(angle_confidences)
    return probs_array[best_angle_idx]

def fuse_best_n_angles(probs_list, n=2):
    """Average of top N most confident angles"""
    if len(probs_list) <= n:
        return np.mean(probs_list, axis=0)
    
    probs_array = np.array(probs_list)
    angle_confidences = []
    for angle_idx in range(len(probs_list)):
        conf = np.mean(np.abs(probs_array[angle_idx] - 0.5))
        angle_confidences.append(conf)
    
    top_n_indices = np.argsort(angle_confidences)[-n:]
    return np.mean(probs_array[top_n_indices], axis=0)

def fuse_majority_vote(probs_list):
    """Majority voting across angles"""
    probs_array = np.array(probs_list)
    binary_preds = (probs_array > 0.5).astype(int)
    vote_sum = np.sum(binary_preds, axis=0)
    threshold = len(probs_list) / 2
    return (vote_sum > threshold).astype(float)

def fuse_weighted_average(probs_list):
    """Confidence-weighted average"""
    probs_array = np.array(probs_list)
    weights = []
    for angle_idx in range(len(probs_list)):
        conf = np.mean(np.abs(probs_array[angle_idx] - 0.5))
        weights.append(conf)
    
    weights = np.array(weights)
    if weights.sum() == 0:
        weights = np.ones(len(probs_list))
    weights = weights / weights.sum()
    
    return np.average(probs_array, axis=0, weights=weights)

def fuse_jaw_confidence(probs_list, jaw_type):
    """Select based on relevant jaw teeth confidence"""
    probs_array = np.array(probs_list)
    
    if jaw_type == 'upper':
        relevant_indices = UPPER_INDICES
    elif jaw_type == 'lower':
        relevant_indices = LOWER_INDICES
    else:
        relevant_indices = list(range(NUM_TEETH))
    
    angle_confidences = []
    for angle_idx in range(len(probs_list)):
        relevant_probs = probs_array[angle_idx, relevant_indices]
        conf = np.mean(np.abs(relevant_probs - 0.5))
        angle_confidences.append(conf)
    
    best_angle_idx = np.argmax(angle_confidences)
    return probs_array[best_angle_idx]

FUSION_STRATEGIES = {
    'average': fuse_average,
    'max_confidence': fuse_max_confidence,
    'best_angle': fuse_best_angle,
    'best_n_angles': lambda x: fuse_best_n_angles(x, BEST_N),
    'majority_vote': fuse_majority_vote,
    'weighted_average': fuse_weighted_average,
    'jaw_confidence': None  # Special handling needed
}

# =================================================================================
# ID NORMALIZATION HELPER FUNCTION
# =================================================================================
def normalize_png_stem_to_newid(stem: str) -> str:
    """
    Cleans up filenames to match CSV IDs.
    """
    s = stem.replace('-', '_').strip()
    s = re.sub(r'\s+', ' ', s)
    
    s = re.sub(r'_rot\d+$', '', s)
    if s.endswith('_top'):
        s = s[:-4]

    jaw_key = ''
    lower_s = s.lower()
    
    if 'upperjawscan' in lower_s:
        match = re.search(r'upperjawscan(\d*)', lower_s, flags=re.IGNORECASE)
        suffix = match.group(1) if match else ''
        jaw_key = 'upper' + suffix
        s = re.sub(r'upperjawscan\d*', '', lower_s, flags=re.IGNORECASE)
    elif 'lowerjawscan' in lower_s:
        match = re.search(r'lowerjawscan(\d*)', lower_s, flags=re.IGNORECASE)
        suffix = match.group(1) if match else ''
        jaw_key = 'lower' + suffix
        s = re.sub(r'lowerjawscan\d*', '', lower_s, flags=re.IGNORECASE)
    elif lower_s.endswith('_upper'):
        jaw_key = 'upper'
        s = s[:-6]
    elif lower_s.endswith('_lower'):
        jaw_key = 'lower'
        s = s[:-6]
    
    s = s.strip().replace(' ', '_').replace('-', '_')
    while '__' in s:
        s = s.replace('__', '_')
    s = s.strip('_')
    
    if jaw_key:
        new_id = f"{s}_{jaw_key}"
    else:
        new_id = s
        
    return new_id.lower()

# =================================================================================
# MODEL DEFINITION
# =================================================================================
class ResNetMultiLabel(nn.Module):
    def __init__(self, backbone="resnet18", num_teeth=32, dropout_rate=0.5):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights=None)
        else:
            net = resnet18(weights=None)
        
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

# =================================================================================
# DATA LOADING
# =================================================================================
def load_test_labels(csv_path):
    """Load test labels with jaw type handling."""
    df = pd.read_csv(csv_path, dtype={'new_id': str})
    df['new_id'] = df['new_id'].astype(str).str.strip().str.lower().str.replace('-', '_')
    df.columns = [str(c) for c in df.columns]
    
    labels_dict = {}
    jaw_type_dict = {}
    
    for _, row in df.iterrows():
        case_id = row['new_id']
        
        is_upper = '_upper' in case_id and '_lower' not in case_id
        is_lower = '_lower' in case_id
        
        label_vector = np.zeros(NUM_TEETH, dtype=np.float32)
        
        for tooth_fdi in VALID_FDI_LABELS:
            tooth_str = str(tooth_fdi)
            if tooth_str in df.columns and pd.notna(row[tooth_str]):
                val = row[tooth_str]
                if int(val) == 1:
                    label_vector[FDI_TO_INDEX[tooth_fdi]] = 1.0

        if is_upper:
            for fdi in LOWER_FDI:
                label_vector[FDI_TO_INDEX[fdi]] = 1.0
            jaw_type_dict[case_id] = 'upper'
        elif is_lower:
            for fdi in UPPER_FDI:
                label_vector[FDI_TO_INDEX[fdi]] = 1.0
            jaw_type_dict[case_id] = 'lower'
        else:
            jaw_type_dict[case_id] = 'unknown'
        
        labels_dict[case_id] = label_vector
    
    print(f"  [Info] Upper jaw samples: {sum(1 for v in jaw_type_dict.values() if v == 'upper')}")
    print(f"  [Info] Lower jaw samples: {sum(1 for v in jaw_type_dict.values() if v == 'lower')}")
    
    return labels_dict, jaw_type_dict

def find_test_images(img_dir, labels_dict):
    """Scans directory and matches images to CSV IDs."""
    grouped = {}
    path = Path(img_dir)
    
    if not path.exists():
        print(f" Error: Directory not found: {img_dir}")
        return grouped

    files = sorted(list(path.glob("*.png")))
    print(f" Scanning {len(files)} files in: {img_dir}")
    
    matched_count = 0
    
    for img_path in files:
        raw_stem = img_path.stem
        norm_id = normalize_png_stem_to_newid(raw_stem)
        
        final_key = None
        if norm_id in labels_dict:
            final_key = norm_id
        elif raw_stem.lower() in labels_dict:
            final_key = raw_stem.lower()
            
        if final_key:
            if final_key not in grouped:
                grouped[final_key] = {'paths': []}
            grouped[final_key]['paths'].append(str(img_path))
            matched_count += 1
            
    print(f" Successfully matched {len(grouped)} unique cases from {matched_count} images.")
    
    return grouped

# =================================================================================
# INFERENCE WITH STRATEGY
# =================================================================================
def test_model_with_strategy(model, grouped_imgs, labels_dict, jaw_type_dict, device, transform, strategy_name):
    """Run inference with a specific fusion strategy."""
    model.eval()
    all_preds, all_targets, all_ids = [], [], []
    
    strategy_func = FUSION_STRATEGIES.get(strategy_name)
    
    with torch.no_grad():
        for case_id, data in tqdm(grouped_imgs.items(), desc=f"Testing ({strategy_name})", leave=True):
            if case_id not in labels_dict:
                continue
            
            labels = labels_dict[case_id]
            jaw_type = jaw_type_dict.get(case_id, 'unknown')
            
            probs_list = []
            for img_path in data['paths']:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    logits = model(img_tensor)
                    probs = torch.sigmoid(logits)
                    probs_list.append(probs.cpu().numpy()[0])
                except Exception as e:
                    print(f"Read Error: {e}")
            
            if not probs_list:
                continue

            # Apply fusion strategy
            if strategy_name == 'jaw_confidence':
                fused_probs = fuse_jaw_confidence(probs_list, jaw_type)
            else:
                fused_probs = strategy_func(probs_list)
            
            all_preds.append(fused_probs)
            all_targets.append(labels)
            all_ids.append(case_id)
            
    return np.array(all_preds), np.array(all_targets), all_ids

# =================================================================================
# METRICS CALCULATION
# =================================================================================
def calculate_metrics(preds, targets, jaw_type_dict, all_ids):
    """Calculate comprehensive metrics."""
    if len(preds) == 0:
        return {}
    
    preds_bin = (preds > 0.5).astype(int)
    targets_bin = targets.astype(int)
    
    # Overall micro metrics
    flat_p = preds_bin.flatten()
    flat_t = targets_bin.flatten()
    p, r, f1, _ = precision_recall_fscore_support(flat_t, flat_p, average='binary', zero_division=0)
    acc = accuracy_score(flat_t, flat_p)
    bal_acc = balanced_accuracy_score(flat_t, flat_p)

    # Per-tooth metrics
    per_tooth = OrderedDict()
    for idx in range(NUM_TEETH):
        fdi = INDEX_TO_FDI[idx]
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(
            targets_bin[:, idx], preds_bin[:, idx], average='binary', zero_division=0
        )
        acc_t = accuracy_score(targets_bin[:, idx], preds_bin[:, idx])
        support = int(targets_bin[:, idx].sum())
        per_tooth[fdi] = {
            'precision': float(p_t), 
            'recall': float(r_t), 
            'f1': float(f1_t), 
            'accuracy': float(acc_t), 
            'support': support
        }

    macro_f1 = np.mean([m['f1'] for m in per_tooth.values()])
    macro_precision = np.mean([m['precision'] for m in per_tooth.values()])
    macro_recall = np.mean([m['recall'] for m in per_tooth.values()])

    # Per-jaw type statistics
    upper_metrics = {'correct': 0, 'total': 0}
    lower_metrics = {'correct': 0, 'total': 0}
    
    for i, case_id in enumerate(all_ids):
        jaw_type = jaw_type_dict.get(case_id, 'unknown')
        pred = preds_bin[i]
        target = targets_bin[i]
        
        if jaw_type == 'upper':
            for fdi in UPPER_FDI:
                idx = FDI_TO_INDEX[fdi]
                upper_metrics['total'] += 1
                if pred[idx] == target[idx]:
                    upper_metrics['correct'] += 1
        elif jaw_type == 'lower':
            for fdi in LOWER_FDI:
                idx = FDI_TO_INDEX[fdi]
                lower_metrics['total'] += 1
                if pred[idx] == target[idx]:
                    lower_metrics['correct'] += 1
    
    upper_acc = upper_metrics['correct'] / upper_metrics['total'] if upper_metrics['total'] > 0 else 0
    lower_acc = lower_metrics['correct'] / lower_metrics['total'] if lower_metrics['total'] > 0 else 0
    
    return {
        'overall_micro': {
            'precision': float(p), 
            'recall': float(r), 
            'f1': float(f1), 
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc)
        },
        'overall_macro': {
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1)
        },
        'per_jaw': {
            'upper_jaw_accuracy': float(upper_acc),
            'upper_jaw_samples': upper_metrics['total'] // 16 if upper_metrics['total'] > 0 else 0,
            'lower_jaw_accuracy': float(lower_acc),
            'lower_jaw_samples': lower_metrics['total'] // 16 if lower_metrics['total'] > 0 else 0
        },
        'per_tooth': per_tooth
    }

# =================================================================================
# PRINTING FUNCTIONS
# =================================================================================
def print_strategy_comparison(all_results, selection_metric):
    """Print comparison table of all strategies."""
    print("\n" + "=" * 120)
    print(" " * 40 + "FUSION STRATEGY COMPARISON (Dynamic Loss)")
    print("=" * 120)
    
    print(f"\n{'Strategy':<25} {'Bal.Acc':<12} {'Macro F1':<12} {'Macro Rec':<12} {'Macro Prec':<12} {'Upper Acc':<12} {'Lower Acc':<12}")
    print("-" * 120)
    
    # Find best strategy
    best_strategy = None
    best_value = -1
    
    for strategy_name, result in all_results.items():
        metrics = result['metrics']
        
        if selection_metric == 'balanced_accuracy':
            current_value = metrics['overall_micro']['balanced_accuracy']
        elif selection_metric == 'macro_recall':
            current_value = metrics['overall_macro']['macro_recall']
        elif selection_metric == 'macro_f1':
            current_value = metrics['overall_macro']['macro_f1']
        elif selection_metric == 'macro_precision':
            current_value = metrics['overall_macro']['macro_precision']
        else:
            current_value = metrics['overall_micro']['balanced_accuracy']
        
        if current_value > best_value:
            best_value = current_value
            best_strategy = strategy_name
    
    for strategy_name, result in all_results.items():
        metrics = result['metrics']
        bal_acc = metrics['overall_micro']['balanced_accuracy']
        macro_f1 = metrics['overall_macro']['macro_f1']
        macro_rec = metrics['overall_macro']['macro_recall']
        macro_prec = metrics['overall_macro']['macro_precision']
        upper_acc = metrics['per_jaw']['upper_jaw_accuracy']
        lower_acc = metrics['per_jaw']['lower_jaw_accuracy']
        
        marker = " <-- BEST" if strategy_name == best_strategy else ""
        print(f"{strategy_name:<25} {bal_acc:<12.4f} {macro_f1:<12.4f} {macro_rec:<12.4f} {macro_prec:<12.4f} {upper_acc:<12.4f} {lower_acc:<12.4f}{marker}")
    
    print("-" * 120)
    print(f"\n Best Strategy (by {selection_metric}): {best_strategy} ({best_value:.4f})")
    print("=" * 120)
    
    return best_strategy

def print_detailed_metrics(metrics, strategy_name):
    """Print detailed metrics for the best strategy."""
    micro = metrics['overall_micro']
    macro = metrics['overall_macro']
    per_jaw = metrics['per_jaw']
    
    print("\n" + "=" * 80)
    print(f" " * 15 + f"DETAILED METRICS FOR BEST STRATEGY: {strategy_name}")
    print("=" * 80)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              OVERALL METRICS (MICRO - flattened all predictions)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Precision:        {micro['precision']:.4f}     (TP / (TP + FP))                            ║
║  Recall:           {micro['recall']:.4f}     (TP / (TP + FN))                            ║
║  F1:               {micro['f1']:.4f}     (Harmonic mean of Prec & Rec)               ║
║  Accuracy:         {micro['accuracy']:.4f}     (Less reliable due to imbalance)            ║
║  Balanced Acc:     {micro['balanced_accuracy']:.4f}     (Avg of TPR and TNR)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    print(f"""╔══════════════════════════════════════════════════════════════════════════════╗
║               OVERALL METRICS (MACRO - avg across all 32 teeth)              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Macro Precision:  {macro['macro_precision']:.4f}                                                    ║
║  Macro Recall:     {macro['macro_recall']:.4f}                                                    ║
║  Macro F1:         {macro['macro_f1']:.4f}                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    print(f"""╔══════════════════════════════════════════════════════════════════════════════╗
║                   PER-JAW ACCURACY (only relevant 16 teeth)                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Upper Jaw:        {per_jaw['upper_jaw_accuracy']:.4f}     ({per_jaw['upper_jaw_samples']} samples)                                    ║
║  Lower Jaw:        {per_jaw['lower_jaw_accuracy']:.4f}     ({per_jaw['lower_jaw_samples']} samples)                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Per-tooth metrics
    print("-" * 80)
    print(" PER-TOOTH METRICS:")
    print("-" * 80)
    print(f"{'FDI':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    
    per_tooth = metrics['per_tooth']
    
    print("─── UPPER JAW (11-28) ───")
    for fdi in UPPER_FDI:
        m = per_tooth[fdi]
        print(f"{fdi:<8} {m['precision']:>10.4f}   {m['recall']:>10.4f}   {m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}")
    
    print("─── LOWER JAW (31-48) ───")
    for fdi in LOWER_FDI:
        m = per_tooth[fdi]
        print(f"{fdi:<8} {m['precision']:>10.4f}   {m['recall']:>10.4f}   {m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}")
    
    print("=" * 80)

def print_sample_predictions(ids, preds, targets, jaw_type_dict, num_samples):
    """Print sample predictions with detailed breakdown."""
    print("\n" + "=" * 80)
    print(" " * 28 + "SAMPLE PREDICTIONS")
    print("=" * 80)
    
    if len(ids) == 0:
        return

    indices = random.sample(range(len(ids)), min(len(ids), num_samples))
    
    for i in indices:
        case_id = ids[i]
        jaw_type = jaw_type_dict.get(case_id, 'unknown')
        target_vec = targets[i]
        pred_vec = (preds[i] > 0.5).astype(int)
        
        if jaw_type == 'upper':
            relevant_fdi = UPPER_FDI
        elif jaw_type == 'lower':
            relevant_fdi = LOWER_FDI
        else:
            relevant_fdi = VALID_FDI_LABELS
        
        truth_missing = [INDEX_TO_FDI[j] for j, val in enumerate(target_vec) 
                        if val == 1 and INDEX_TO_FDI[j] in relevant_fdi]
        pred_missing = [INDEX_TO_FDI[j] for j, val in enumerate(pred_vec) 
                       if val == 1 and INDEX_TO_FDI[j] in relevant_fdi]
        
        truth_set = set(truth_missing)
        pred_set = set(pred_missing)
        
        tp = sorted(list(truth_set.intersection(pred_set)))
        fn = sorted(list(truth_set.difference(pred_set)))
        fp = sorted(list(pred_set.difference(truth_set)))
        
        def fmt_list(lst):
            return str(sorted(lst)) if lst else "None"
        
        correct = len(tp) + (len(relevant_fdi) - len(truth_set) - len(fp))
        case_acc = correct / len(relevant_fdi) if relevant_fdi else 0
        
        print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│ Case ID: {case_id:<57}│
│ Jaw Type: {jaw_type.upper():<56}│
├──────────────────────────────────────────────────────────────────────┤
│  Ground Truth (Missing): {str(truth_missing):<41}│
│  Prediction (Missing):   {str(pred_missing):<41}│
├──────────────────────────────────────────────────────────────────────┤
│   Correctly Found (TP): {fmt_list(tp):<41}│
│   Missed Teeth (FN):    {fmt_list(fn):<41}│
│   False Alarms (FP):    {fmt_list(fp):<41}│
│  Case Accuracy: {case_acc:.2%} ({correct}/{len(relevant_fdi)}){"":>37}│
└──────────────────────────────────────────────────────────────────────┘""")
    
    print("\n" + "=" * 80)

# =================================================================================
# PLOTTING FUNCTIONS
# =================================================================================
def generate_comparison_plot(all_results, save_dir):
    """Generate comparison plot for all strategies."""
    strategies = list(all_results.keys())
    
    bal_accs = [all_results[s]['metrics']['overall_micro']['balanced_accuracy'] for s in strategies]
    macro_f1s = [all_results[s]['metrics']['overall_macro']['macro_f1'] for s in strategies]
    macro_recs = [all_results[s]['metrics']['overall_macro']['macro_recall'] for s in strategies]
    macro_precs = [all_results[s]['metrics']['overall_macro']['macro_precision'] for s in strategies]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fusion Strategy Comparison (Dynamic Loss Model)', fontsize=14, fontweight='bold')
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    
    # Balanced Accuracy
    bars1 = axes[0, 0].bar(strategies, bal_accs, color=colors)
    axes[0, 0].set_title('Balanced Accuracy')
    axes[0, 0].set_ylim(0.75, 1.0)
    axes[0, 0].tick_params(axis='x', rotation=45)
    best_idx = np.argmax(bal_accs)
    bars1[best_idx].set_edgecolor('red')
    bars1[best_idx].set_linewidth(3)
    
    # Macro F1
    bars2 = axes[0, 1].bar(strategies, macro_f1s, color=colors)
    axes[0, 1].set_title('Macro F1')
    axes[0, 1].set_ylim(0.75, 1.0)
    axes[0, 1].tick_params(axis='x', rotation=45)
    best_idx = np.argmax(macro_f1s)
    bars2[best_idx].set_edgecolor('red')
    bars2[best_idx].set_linewidth(3)
    
    # Macro Recall
    bars3 = axes[1, 0].bar(strategies, macro_recs, color=colors)
    axes[1, 0].set_title('Macro Recall')
    axes[1, 0].set_ylim(0.75, 1.0)
    axes[1, 0].tick_params(axis='x', rotation=45)
    best_idx = np.argmax(macro_recs)
    bars3[best_idx].set_edgecolor('red')
    bars3[best_idx].set_linewidth(3)
    
    # Macro Precision
    bars4 = axes[1, 1].bar(strategies, macro_precs, color=colors)
    axes[1, 1].set_title('Macro Precision')
    axes[1, 1].set_ylim(0.75, 1.0)
    axes[1, 1].tick_params(axis='x', rotation=45)
    best_idx = np.argmax(macro_precs)
    bars4[best_idx].set_edgecolor('red')
    bars4[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "fusion_strategy_comparison_dynamit.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Comparison plot saved to {save_dir}/fusion_strategy_comparison_dynamit.png")

def generate_detailed_plots(metrics, preds, targets, save_dir):
    """Generate detailed plots for the best strategy."""
    per_tooth = metrics['per_tooth']
    
    # F1 Score per Jaw
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    upper_f1s = [per_tooth[fdi]['f1'] for fdi in UPPER_FDI]
    colors_upper = ['green' if f1 >= 0.85 else 'orange' if f1 >= 0.7 else 'red' for f1 in upper_f1s]
    axes[0].bar(range(len(UPPER_FDI)), upper_f1s, color=colors_upper)
    axes[0].set_xticks(range(len(UPPER_FDI)))
    axes[0].set_xticklabels([str(f) for f in UPPER_FDI], rotation=45)
    axes[0].set_title('F1 Score - Upper Jaw (11-28) - Dynamic Loss')
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=np.mean(upper_f1s), color='blue', linestyle='--', label=f'Mean: {np.mean(upper_f1s):.3f}')
    axes[0].axhline(y=0.85, color='green', linestyle=':', alpha=0.5, label='Good (0.85)')
    axes[0].legend()
    
    lower_f1s = [per_tooth[fdi]['f1'] for fdi in LOWER_FDI]
    colors_lower = ['green' if f1 >= 0.85 else 'orange' if f1 >= 0.7 else 'red' for f1 in lower_f1s]
    axes[1].bar(range(len(LOWER_FDI)), lower_f1s, color=colors_lower)
    axes[1].set_xticks(range(len(LOWER_FDI)))
    axes[1].set_xticklabels([str(f) for f in LOWER_FDI], rotation=45)
    axes[1].set_title('F1 Score - Lower Jaw (31-48) - Dynamic Loss')
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=np.mean(lower_f1s), color='blue', linestyle='--', label=f'Mean: {np.mean(lower_f1s):.3f}')
    axes[1].axhline(y=0.85, color='green', linestyle=':', alpha=0.5, label='Good (0.85)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "f1_score_per_jaw_dynamit.png", dpi=150)
    plt.close()
    
    # Recall per Jaw (Key Metric)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    upper_recalls = [per_tooth[fdi]['recall'] for fdi in UPPER_FDI]
    colors_upper = ['green' if r >= 0.85 else 'orange' if r >= 0.7 else 'red' for r in upper_recalls]
    axes[0].bar(range(len(UPPER_FDI)), upper_recalls, color=colors_upper)
    axes[0].set_xticks(range(len(UPPER_FDI)))
    axes[0].set_xticklabels([str(f) for f in UPPER_FDI], rotation=45)
    axes[0].set_title('Recall - Upper Jaw - Dynamic Loss')
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=np.mean(upper_recalls), color='blue', linestyle='--', label=f'Mean: {np.mean(upper_recalls):.3f}')
    axes[0].legend()
    
    lower_recalls = [per_tooth[fdi]['recall'] for fdi in LOWER_FDI]
    colors_lower = ['green' if r >= 0.85 else 'orange' if r >= 0.7 else 'red' for r in lower_recalls]
    axes[1].bar(range(len(LOWER_FDI)), lower_recalls, color=colors_lower)
    axes[1].set_xticks(range(len(LOWER_FDI)))
    axes[1].set_xticklabels([str(f) for f in LOWER_FDI], rotation=45)
    axes[1].set_title('Recall - Lower Jaw  - Dynamic Loss')
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=np.mean(lower_recalls), color='blue', linestyle='--', label=f'Mean: {np.mean(lower_recalls):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "recall_per_jaw_dynamit.png", dpi=150)
    plt.close()
    
    # Confusion Matrix
    preds_bin = (preds > 0.5).astype(int)
    cm = confusion_matrix(targets.flatten(), preds_bin.flatten())
    
    plt.figure(figsize=(10, 8))
    
    cm_percent = cm.astype('float') / cm.sum() * 100
    labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' for j in range(2)] for i in range(2)])
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Present (0)', 'Missing (1)'],
                yticklabels=['Present (0)', 'Missing (1)'],
                annot_kws={'size': 14})
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    plt.title(f'Confusion Matrix - Dynamic Loss\nSensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.savefig(Path(save_dir) / "confusion_matrix_dynamit.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Detailed plots saved to {save_dir}")

# =================================================================================
# MAIN
# =================================================================================
def main():
    print("\n" + "=" * 80)
    print(" " * 10 + "2D MODEL TESTING - Fixed Strategy (Dynamic Loss Model)")
    print("=" * 80)
    print(f" Strategy: {FIXED_STRATEGY}")
    print("=" * 80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/6] Using device: {device}")

    # Load Model 
    print(f"\n[2/6] Loading model from: {MODEL_PATH}")
    model = ResNetMultiLabel(
        backbone="resnet18",
        num_teeth=NUM_TEETH, 
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"   Model loaded successfully")
    except Exception as e:
        print(f"   Error loading model: {e}")
        return

    # Load Labels
    print(f"\n[3/6] Loading test labels from: {TEST_LABELS_CSV}")
    labels_dict, jaw_type_dict = load_test_labels(TEST_LABELS_CSV)
    print(f"   Loaded labels for {len(labels_dict)} cases")
    
    # Load Images & Match
    print(f"\n[4/6] Finding test images in: {TEST_IMG_DIR}")
    grouped_imgs = find_test_images(TEST_IMG_DIR, labels_dict)
    
    if len(grouped_imgs) == 0:
        print(" No matches found.")
        return

    # Transform
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # only run best_n_angles
    print(f"\n[5/6] Running fixed fusion strategy: {FIXED_STRATEGY} ...")
    preds, targets, ids = test_model_with_strategy(
        model, grouped_imgs, labels_dict, jaw_type_dict, device, transform, FIXED_STRATEGY
    )
    metrics = calculate_metrics(preds, targets, jaw_type_dict, ids)
    
    print(f"\n[6/6] Generating detailed output for strategy: {FIXED_STRATEGY}")
    print_detailed_metrics(metrics, FIXED_STRATEGY)
    print_sample_predictions(ids, preds, targets, jaw_type_dict, NUM_SAMPLE_PREDICTIONS)
    generate_detailed_plots(metrics, preds, targets, OUTPUT_DIR)
    
    # save results
    results_to_save = {
        'strategy': FIXED_STRATEGY,
        'metrics': metrics,
        'per_tooth': {str(k): v for k, v in metrics['per_tooth'].items()}
    }
    results_file = Path(OUTPUT_DIR) / "test_results_dynamit.json"
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n All results saved to {results_file}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "SUMMARY")
    print("=" * 80)
    print(f"  Strategy: {FIXED_STRATEGY}")
    print(f"  Balanced Accuracy: {metrics['overall_micro']['balanced_accuracy']:.4f}")
    print(f"  Macro Recall: {metrics['overall_macro']['macro_recall']:.4f}")
    print(f"  Macro F1: {metrics['overall_macro']['macro_f1']:.4f}")
    print(f"  Upper Jaw Accuracy: {metrics['per_jaw']['upper_jaw_accuracy']:.4f}")
    print(f"  Lower Jaw Accuracy: {metrics['per_jaw']['lower_jaw_accuracy']:.4f}")
    print("=" * 80)
    print(" " * 30 + "DONE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
