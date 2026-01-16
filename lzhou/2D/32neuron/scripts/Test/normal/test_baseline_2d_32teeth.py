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
from enum import Enum

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
# FUSION STRATEGY ENUM
# =================================================================================
class FusionStrategy(Enum):
    AVERAGE = "average"                    # average all angles
    MAX_CONFIDENCE = "max_confidence"      # choose per-tooth best angle based on max confidence
    BEST_ANGLE = "best_angle"              # choose single best angle based on overall confidence
    BEST_N_ANGLES = "best_n_angles"        # choose best N angles and average (default 2)
    MAJORITY_VOTE = "majority_vote"        # majority vote
    WEIGHTED_AVERAGE = "weighted_average"  # confidence weighted average
    JAW_CONFIDENCE = "jaw_confidence"      # choose based on jaw prediction confidence

# =================================================================================
# CONFIGURATION
# =================================================================================
TEST_IMG_DIR = "/home/user/lzhou/week15/render_output/test"
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"
MODEL_PATH = "/home/user/lzhou/week15-32/output/Train2D/32teeth/baseline_bce_best_2d_32teeth.pth"
OUTPUT_DIR = "/home/user/lzhou/week15-32/output/Test2D/32teeth_auto_best"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256  
NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 10
DROPOUT_RATE = 0.5  

# ========== FUSION CONFIGURATION ==========
BEST_N = 2  # Use for BEST_N_ANGLES

# ========== SELECTION METRIC ==========
# 'balanced_accuracy', 'macro_f1', 'macro_recall', 'macro_precision'
SELECTION_METRIC = 'macro_f1'

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

# =================================================================================
# ID NORMALIZATION HELPER
# =================================================================================
def normalize_png_stem_to_newid(stem: str) -> str:
    """Cleans up filenames to match CSV IDs."""
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
# FUSION METHODS
# =================================================================================
def calculate_confidence_score(probs):
    """calculate confidence score as mean distance from 0.5"""
    return np.mean(np.abs(probs - 0.5))

def fuse_predictions(probs_list, strategy=FusionStrategy.AVERAGE, n_best=2, jaw_type=None):
    """Fuse predictions from multiple angles based on the specified strategy"""
    if len(probs_list) == 0:
        return None
    
    if len(probs_list) == 1:
        return probs_list[0]
    
    probs_array = np.array(probs_list)
    
    if strategy == FusionStrategy.AVERAGE:
        return np.mean(probs_array, axis=0)
    
    elif strategy == FusionStrategy.MAX_CONFIDENCE:
        confidence_per_tooth = np.abs(probs_array - 0.5)
        best_angle_per_tooth = np.argmax(confidence_per_tooth, axis=0)
        result = np.array([probs_array[best_angle_per_tooth[i], i] for i in range(probs_array.shape[1])])
        return result
    
    elif strategy == FusionStrategy.BEST_ANGLE:
        confidences = [calculate_confidence_score(p) for p in probs_list]
        best_idx = np.argmax(confidences)
        return probs_list[best_idx]
    
    elif strategy == FusionStrategy.BEST_N_ANGLES:
        confidences = [calculate_confidence_score(p) for p in probs_list]
        n = min(n_best, len(probs_list))
        top_n_indices = np.argsort(confidences)[-n:]
        top_n_probs = [probs_list[i] for i in top_n_indices]
        return np.mean(top_n_probs, axis=0)
    
    elif strategy == FusionStrategy.MAJORITY_VOTE:
        binary_preds = (probs_array > 0.5).astype(float)
        vote_ratio = np.mean(binary_preds, axis=0)
        return vote_ratio
    
    elif strategy == FusionStrategy.WEIGHTED_AVERAGE:
        confidences = np.array([calculate_confidence_score(p) for p in probs_list])
        if confidences.sum() == 0:
            weights = np.ones(len(probs_list)) / len(probs_list)
        else:
            weights = confidences / confidences.sum()
        return np.average(probs_array, axis=0, weights=weights)
    
    elif strategy == FusionStrategy.JAW_CONFIDENCE:
        if jaw_type == 'upper':
            relevant_indices = [FDI_TO_INDEX[fdi] for fdi in UPPER_FDI]
        elif jaw_type == 'lower':
            relevant_indices = [FDI_TO_INDEX[fdi] for fdi in LOWER_FDI]
        else:
            relevant_indices = list(range(NUM_TEETH))
        
        jaw_confidences = []
        for probs in probs_list:
            relevant_probs = probs[relevant_indices]
            jaw_confidences.append(calculate_confidence_score(relevant_probs))
        
        best_idx = np.argmax(jaw_confidences)
        return probs_list[best_idx]
    
    else:
        return np.mean(probs_array, axis=0)

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
    """Scans directory and matches images to CSV IDs"""
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
# INFERENCE
# =================================================================================
def test_model(model, grouped_imgs, labels_dict, jaw_type_dict, device, transform, 
               strategy=FusionStrategy.AVERAGE, n_best=2, show_progress=True):
    """Run inference with specified fusion strategy."""
    model.eval()
    all_preds, all_targets, all_ids = [], [], []
    
    fusion_stats = {
        'num_angles_per_case': [],
        'confidence_scores': []
    }
    
    iterator = tqdm(grouped_imgs.items(), desc=f"Testing ({strategy.value})") if show_progress else grouped_imgs.items()
    
    with torch.no_grad():
        for case_id, data in iterator:
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
            
            fused_probs = fuse_predictions(
                probs_list, 
                strategy=strategy, 
                n_best=n_best,
                jaw_type=jaw_type
            )
            
            if fused_probs is None:
                continue
            
            all_preds.append(fused_probs)
            all_targets.append(labels)
            all_ids.append(case_id)
            
            fusion_stats['num_angles_per_case'].append(len(probs_list))
            fusion_stats['confidence_scores'].append(calculate_confidence_score(fused_probs))
            
    return np.array(all_preds), np.array(all_targets), all_ids, fusion_stats

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

def get_metric_value(metrics, metric_name):
    """Extract specific metric value for comparison."""
    if metric_name == 'balanced_accuracy':
        return metrics['overall_micro']['balanced_accuracy']
    elif metric_name == 'macro_f1':
        return metrics['overall_macro']['macro_f1']
    elif metric_name == 'macro_recall':
        return metrics['overall_macro']['macro_recall']
    elif metric_name == 'macro_precision':
        return metrics['overall_macro']['macro_precision']
    elif metric_name == 'accuracy':
        return metrics['overall_micro']['accuracy']
    else:
        return metrics['overall_micro']['balanced_accuracy']

# =================================================================================
# COMPARISON OF ALL STRATEGIES
# =================================================================================
def compare_all_strategies(model, grouped_imgs, labels_dict, jaw_type_dict, device, transform):
    """Compare all fusion strategies and collect results"""
    strategies = [
        FusionStrategy.AVERAGE,
        FusionStrategy.MAX_CONFIDENCE,
        FusionStrategy.BEST_ANGLE,
        FusionStrategy.BEST_N_ANGLES,
        FusionStrategy.MAJORITY_VOTE,
        FusionStrategy.WEIGHTED_AVERAGE,
        FusionStrategy.JAW_CONFIDENCE,
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing Strategy: {strategy.value}")
        print('='*60)
        
        preds, targets, ids, stats = test_model(
            model, grouped_imgs, labels_dict, jaw_type_dict, 
            device, transform, strategy=strategy, n_best=BEST_N
        )
        
        if len(preds) > 0:
            metrics = calculate_metrics(preds, targets, jaw_type_dict, ids)
            results[strategy.value] = {
                'metrics': metrics,
                'stats': stats,
                'preds': preds,
                'targets': targets,
                'ids': ids
            }
            
            print(f"  Balanced Accuracy: {metrics['overall_micro']['balanced_accuracy']:.4f}")
            print(f"  Macro F1: {metrics['overall_macro']['macro_f1']:.4f}")
            print(f"  Macro Recall: {metrics['overall_macro']['macro_recall']:.4f}")
            print(f"  Upper Jaw Acc: {metrics['per_jaw']['upper_jaw_accuracy']:.4f}")
            print(f"  Lower Jaw Acc: {metrics['per_jaw']['lower_jaw_accuracy']:.4f}")
    
    return results

def select_best_strategy(results, metric_name='balanced_accuracy'):
    """Select the best fusion strategy based on the specified metric."""
    best_strategy = None
    best_value = -1
    
    for strategy, data in results.items():
        value = get_metric_value(data['metrics'], metric_name)
        if value > best_value:
            best_value = value
            best_strategy = strategy
    
    return best_strategy, best_value

# =================================================================================
# PRINTING FUNCTIONS
# =================================================================================
def print_comparison_table(results, metric_name='balanced_accuracy'):
    """Print comparison table of all strategies."""
    print("\n" + "="*120)
    print(" "*45 + "FUSION STRATEGY COMPARISON")
    print("="*120)
    
    print(f"\n{'Strategy':<20} {'Bal.Acc':>12} {'Macro F1':>12} {'Macro Rec':>12} {'Macro Prec':>12} {'Upper Acc':>12} {'Lower Acc':>12}")
    print("-"*120)
    
    best_strategy, best_value = select_best_strategy(results, metric_name)
    
    for strategy, data in results.items():
        m = data['metrics']
        bal_acc = m['overall_micro']['balanced_accuracy']
        macro_f1 = m['overall_macro']['macro_f1']
        macro_rec = m['overall_macro']['macro_recall']
        macro_prec = m['overall_macro']['macro_precision']
        upper_acc = m['per_jaw']['upper_jaw_accuracy']
        lower_acc = m['per_jaw']['lower_jaw_accuracy']
        
        marker = " <-- BEST" if strategy == best_strategy else ""
        print(f"{strategy:<20} {bal_acc:>12.4f} {macro_f1:>12.4f} {macro_rec:>12.4f} {macro_prec:>12.4f} {upper_acc:>12.4f} {lower_acc:>12.4f}{marker}")
    
    print("-"*120)
    print(f"\n Best Strategy (by {metric_name}): {best_strategy} ({best_value:.4f})")
    print("="*120)
    
    return best_strategy

def print_metrics_summary(metrics, strategy_name):
    """Print detailed metrics summary for the best strategy."""
    micro = metrics['overall_micro']
    macro = metrics['overall_macro']
    per_jaw = metrics['per_jaw']
    
    print("\n" + "="*80)
    print(f" "*15 + f"DETAILED METRICS FOR BEST STRATEGY: {strategy_name}")
    print("="*80)
    
    print(f"\n╔{'═'*78}╗")
    print(f"║{' OVERALL METRICS (MICRO - flattened all predictions)':^78}║")
    print(f"╠{'═'*78}╣")
    print(f"║  Precision:        {micro['precision']:<10.4f} (TP / (TP + FP)){' '*28}║")
    print(f"║  Recall:           {micro['recall']:<10.4f} (TP / (TP + FN)){' '*12}║")
    print(f"║  F1:               {micro['f1']:<10.4f} (Harmonic mean of Prec & Rec){' '*15}║")
    print(f"║  Accuracy:         {micro['accuracy']:<10.4f} (Less reliable due to imbalance){' '*12}║")
    print(f"║  Balanced Acc:     {micro['balanced_accuracy']:<10.4f} (Avg of TPR and TNR){' '*24}║")
    print(f"╚{'═'*78}╝")
    
    print(f"\n╔{'═'*78}╗")
    print(f"║{' OVERALL METRICS (MACRO - avg across all 32 teeth)':^78}║")
    print(f"╠{'═'*78}╣")
    print(f"║  Macro Precision:  {macro['macro_precision']:<10.4f}{' '*48}║")
    print(f"║  Macro Recall:     {macro['macro_recall']:<10.4f}{' '*4}║")
    print(f"║  Macro F1:         {macro['macro_f1']:<10.4f}{' '*48}║")
    print(f"╚{'═'*78}╝")
    
    print(f"\n╔{'═'*78}╗")
    print(f"║{' PER-JAW ACCURACY (only relevant 16 teeth)':^78}║")
    print(f"╠{'═'*78}╣")
    print(f"║  Upper Jaw:        {per_jaw['upper_jaw_accuracy']:<10.4f} ({per_jaw['upper_jaw_samples']} samples){' '*36}║")
    print(f"║  Lower Jaw:        {per_jaw['lower_jaw_accuracy']:<10.4f} ({per_jaw['lower_jaw_samples']} samples){' '*36}║")
    print(f"╚{'═'*78}╝")
    
    print("\n" + "-" * 80)
    print(" PER-TOOTH METRICS:")
    print("-" * 80)
    print(f"{'FDI':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    
    print("─── UPPER JAW (11-28) ───")
    for fdi in UPPER_FDI:
        m = metrics['per_tooth'][fdi]
        print(f"{fdi:<8} {m['precision']:>10.4f}   {m['recall']:>10.4f}   {m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}")
    
    print("─── LOWER JAW (31-48) ───")
    for fdi in LOWER_FDI:
        m = metrics['per_tooth'][fdi]
        print(f"{fdi:<8} {m['precision']:>10.4f}   {m['recall']:>10.4f}   {m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}")
    
    print("=" * 80)

def print_sample_predictions(ids, preds, targets, jaw_type_dict, num_samples=10):
    """Print sample predictions."""
    print("\n" + "="*80)
    print(" "*28 + "SAMPLE PREDICTIONS")
    print("="*80)
    
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
        
        print(f"\n┌{'─'*70}┐")
        print(f"│ Case ID: {case_id:<59}│")
        print(f"│ Jaw Type: {jaw_type.upper():<58}│")
        print(f"├{'─'*70}┤")
        print(f"│  Ground Truth (Missing): {str(sorted(truth_missing)):<43}│")
        print(f"│  Prediction (Missing):   {str(sorted(pred_missing)):<43}│")
        print(f"├{'─'*70}┤")
        print(f"│   Correctly Found (TP): {fmt_list(tp):<43}│")
        print(f"│   Missed Teeth (FN):    {fmt_list(fn):<43}│")
        print(f"│   False Alarms (FP):    {fmt_list(fp):<43}│")
        
        correct = len(tp) + (len(relevant_fdi) - len(truth_set) - len(fp))
        case_acc = correct / len(relevant_fdi) if relevant_fdi else 0
        print(f"│  Case Accuracy: {case_acc:.2%} ({correct}/{len(relevant_fdi)}){' '*37}│")
        print(f"└{'─'*70}┘")
    
    print("\n" + "="*80)

# =================================================================================
# PLOTTING FUNCTIONS
# =================================================================================
def generate_comparison_plot(results, save_dir):
    """Generate a comparison plot for all strategies."""
    strategies = list(results.keys())
    
    bal_accs = [results[s]['metrics']['overall_micro']['balanced_accuracy'] for s in strategies]
    macro_f1s = [results[s]['metrics']['overall_macro']['macro_f1'] for s in strategies]
    macro_recs = [results[s]['metrics']['overall_macro']['macro_recall'] for s in strategies]
    macro_precs = [results[s]['metrics']['overall_macro']['macro_precision'] for s in strategies]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(strategies))
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    
    # Balanced Accuracy
    bars1 = axes[0, 0].bar(x, bal_accs, color=colors)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_title('Balanced Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim(0.7, 1)
    best_idx = np.argmax(bal_accs)
    bars1[best_idx].set_color('red')
    bars1[best_idx].set_edgecolor('darkred')
    bars1[best_idx].set_linewidth(2)
    
    # Macro F1
    bars2 = axes[0, 1].bar(x, macro_f1s, color=colors)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    axes[0, 1].set_title('Macro F1 Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0.7, 1)
    best_idx = np.argmax(macro_f1s)
    bars2[best_idx].set_color('red')
    bars2[best_idx].set_edgecolor('darkred')
    bars2[best_idx].set_linewidth(2)
    
    # Macro Recall
    bars3 = axes[1, 0].bar(x, macro_recs, color=colors)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_title('Macro Recall', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim(0.7, 1)
    best_idx = np.argmax(macro_recs)
    bars3[best_idx].set_color('red')
    bars3[best_idx].set_edgecolor('darkred')
    bars3[best_idx].set_linewidth(2)
    
    # Macro Precision
    bars4 = axes[1, 1].bar(x, macro_precs, color=colors)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_title('Macro Precision', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim(0.7, 1)
    best_idx = np.argmax(macro_precs)
    bars4[best_idx].set_color('red')
    bars4[best_idx].set_edgecolor('darkred')
    bars4[best_idx].set_linewidth(2)
    
    plt.suptitle('Fusion Strategy Comparison\n(Red = Best for each metric)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "fusion_strategy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Comparison plot saved to {save_dir}/fusion_strategy_comparison.png")

def generate_detailed_plots(metrics, preds, targets, save_dir, strategy_name):
    """Generate detailed plots for the best strategy."""
    per_tooth = metrics['per_tooth']
    
    # ===== F1 Score per Jaw =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Upper jaw
    upper_f1s = [per_tooth[fdi]['f1'] for fdi in UPPER_FDI]
    colors_upper = ['coral' if f < 0.85 else 'steelblue' for f in upper_f1s]
    axes[0].bar(range(len(UPPER_FDI)), upper_f1s, color=colors_upper)
    axes[0].set_xticks(range(len(UPPER_FDI)))
    axes[0].set_xticklabels([str(f) for f in UPPER_FDI], rotation=45)
    axes[0].set_title(f'F1 Score - Upper Jaw (11-28)\nStrategy: {strategy_name}', fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=np.mean(upper_f1s), color='red', linestyle='--', label=f'Mean: {np.mean(upper_f1s):.3f}')
    axes[0].axhline(y=0.85, color='orange', linestyle=':', alpha=0.7, label='Threshold: 0.85')
    axes[0].legend()
    axes[0].set_ylabel('F1 Score')
    
    # Lower jaw
    lower_f1s = [per_tooth[fdi]['f1'] for fdi in LOWER_FDI]
    colors_lower = ['coral' if f < 0.85 else 'seagreen' for f in lower_f1s]
    axes[1].bar(range(len(LOWER_FDI)), lower_f1s, color=colors_lower)
    axes[1].set_xticks(range(len(LOWER_FDI)))
    axes[1].set_xticklabels([str(f) for f in LOWER_FDI], rotation=45)
    axes[1].set_title(f'F1 Score - Lower Jaw (31-48)\nStrategy: {strategy_name}', fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=np.mean(lower_f1s), color='red', linestyle='--', label=f'Mean: {np.mean(lower_f1s):.3f}')
    axes[1].axhline(y=0.85, color='orange', linestyle=':', alpha=0.7, label='Threshold: 0.85')
    axes[1].legend()
    axes[1].set_ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "f1_score_per_jaw.png", dpi=150)
    plt.close()
    
    # ===== Recall per Tooth =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    upper_recalls = [per_tooth[fdi]['recall'] for fdi in UPPER_FDI]
    colors_upper = ['coral' if r < 0.85 else 'steelblue' for r in upper_recalls]
    axes[0].bar(range(len(UPPER_FDI)), upper_recalls, color=colors_upper)
    axes[0].set_xticks(range(len(UPPER_FDI)))
    axes[0].set_xticklabels([str(f) for f in UPPER_FDI], rotation=45)
    axes[0].set_title(f'Recall - Upper Jaw\nStrategy: {strategy_name}', fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=np.mean(upper_recalls), color='red', linestyle='--', label=f'Mean: {np.mean(upper_recalls):.3f}')
    axes[0].legend()
    axes[0].set_ylabel('Recall')
    
    lower_recalls = [per_tooth[fdi]['recall'] for fdi in LOWER_FDI]
    colors_lower = ['coral' if r < 0.85 else 'seagreen' for r in lower_recalls]
    axes[1].bar(range(len(LOWER_FDI)), lower_recalls, color=colors_lower)
    axes[1].set_xticks(range(len(LOWER_FDI)))
    axes[1].set_xticklabels([str(f) for f in LOWER_FDI], rotation=45)
    axes[1].set_title(f'Recall - Lower Jaw \nStrategy: {strategy_name}', fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=np.mean(lower_recalls), color='red', linestyle='--', label=f'Mean: {np.mean(lower_recalls):.3f}')
    axes[1].legend()
    axes[1].set_ylabel('Recall')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "recall_per_jaw.png", dpi=150)
    plt.close()
    
    # ===== Confusion Matrix =====
    preds_bin = (preds > 0.5).astype(int)
    targets_bin = targets.astype(int)
    cm = confusion_matrix(targets_bin.flatten(), preds_bin.flatten())
    
    plt.figure(figsize=(10, 8))
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    # 创建标注文本
    labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                        for j in range(2)] for i in range(2)])
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Present (0)', 'Missing (1)'],
                yticklabels=['Present (0)', 'Missing (1)'],
                annot_kws={'size': 14})
    
    plt.title(f'Confusion Matrix\nStrategy: {strategy_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    
    # calculate TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    plt.figtext(0.5, -0.05, 
                f'TN={tn} | FP={fp} | FN={fn} | TP={tp}\n'
                f'Sensitivity (Recall) = TP/(TP+FN) = {tp/(tp+fn):.4f}\n'
                f'Specificity = TN/(TN+FP) = {tn/(tn+fp):.4f}',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ===== Precision vs Recall Scatter =====
    fig, ax = plt.subplots(figsize=(10, 8))
    
    precisions = [per_tooth[fdi]['precision'] for fdi in VALID_FDI_LABELS]
    recalls = [per_tooth[fdi]['recall'] for fdi in VALID_FDI_LABELS]

    # distinguish upper and lower
    upper_prec = [per_tooth[fdi]['precision'] for fdi in UPPER_FDI]
    upper_rec = [per_tooth[fdi]['recall'] for fdi in UPPER_FDI]
    lower_prec = [per_tooth[fdi]['precision'] for fdi in LOWER_FDI]
    lower_rec = [per_tooth[fdi]['recall'] for fdi in LOWER_FDI]
    
    ax.scatter(upper_rec, upper_prec, c='steelblue', s=100, label='Upper Jaw', alpha=0.7)
    ax.scatter(lower_rec, lower_prec, c='seagreen', s=100, label='Lower Jaw', alpha=0.7)
    
    # note teeth with low precision or recall
    for fdi in VALID_FDI_LABELS:
        prec = per_tooth[fdi]['precision']
        rec = per_tooth[fdi]['recall']
        if rec < 0.85 or prec < 0.9:
            ax.annotate(str(fdi), (rec, prec), fontsize=8, ha='center')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision vs Recall per Tooth\nStrategy: {strategy_name}', fontweight='bold')
    ax.set_xlim(0.7, 1.02)
    ax.set_ylim(0.85, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot F1 score contours
    x_range = np.linspace(0.7, 1, 100)
    for f1_val in [0.85, 0.90, 0.95]:
        y_range = (f1_val * x_range) / (2 * x_range - f1_val)
        valid_mask = (y_range >= 0.85) & (y_range <= 1.02)
        ax.plot(x_range[valid_mask], y_range[valid_mask], '--', alpha=0.5, label=f'F1={f1_val}')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "precision_recall_scatter.png", dpi=150)
    plt.close()
    
    print(f" Detailed plots saved to {save_dir}")

# =================================================================================
# MAIN
# =================================================================================
def main():
    print("\n" + "="*80)
    print(" "*10 + "2D MODEL TESTING - AUTO-SELECT BEST FUSION STRATEGY")
    print("="*80)
    print(f" Selection Metric: {SELECTION_METRIC}")
    print("="*80)
    
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

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Compare all strategies
    print(f"\n[5/6] Comparing all fusion strategies...")
    results = compare_all_strategies(
        model, grouped_imgs, labels_dict, jaw_type_dict, device, transform
    )
    
    # Print comparison table and select best
    best_strategy = print_comparison_table(results, SELECTION_METRIC)
    
    # Generate comparison plot
    generate_comparison_plot(results, OUTPUT_DIR)
    
    # Get best strategy data
    best_data = results[best_strategy]
    best_metrics = best_data['metrics']
    best_preds = best_data['preds']
    best_targets = best_data['targets']
    best_ids = best_data['ids']
    
    # Print detailed metrics for best strategy
    print(f"\n[6/6] Generating detailed output for best strategy: {best_strategy}")
    print_metrics_summary(best_metrics, best_strategy)
    
    # Print sample predictions
    print_sample_predictions(best_ids, best_preds, best_targets, jaw_type_dict, NUM_SAMPLE_PREDICTIONS)
    
    # Generate detailed plots
    generate_detailed_plots(best_metrics, best_preds, best_targets, OUTPUT_DIR, best_strategy)
    
    # Save all results to JSON
    results_file = Path(OUTPUT_DIR) / "test_results.json"
    
    json_results = {
        'selection_metric': SELECTION_METRIC,
        'best_strategy': best_strategy,
        'all_strategies': {}
    }
    
    for strategy, data in results.items():
        json_results['all_strategies'][strategy] = {
            'metrics': {
                'overall_micro': data['metrics']['overall_micro'],
                'overall_macro': data['metrics']['overall_macro'],
                'per_jaw': data['metrics']['per_jaw'],
                'per_tooth': {str(k): v for k, v in data['metrics']['per_tooth'].items()}
            },
            'stats': {
                'avg_angles_per_case': float(np.mean(data['stats']['num_angles_per_case'])),
                'avg_confidence': float(np.mean(data['stats']['confidence_scores']))
            }
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n All results saved to {results_file}")
    
    # Summary
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY")
    print("="*80)
    print(f"  Best Strategy: {best_strategy}")
    print(f"  Balanced Accuracy: {best_metrics['overall_micro']['balanced_accuracy']:.4f}")
    print(f"  Macro Recall: {best_metrics['overall_macro']['macro_recall']:.4f}")
    print(f"  Macro F1: {best_metrics['overall_macro']['macro_f1']:.4f}")
    print(f"  Upper Jaw Accuracy: {best_metrics['per_jaw']['upper_jaw_accuracy']:.4f}")
    print(f"  Lower Jaw Accuracy: {best_metrics['per_jaw']['lower_jaw_accuracy']:.4f}")
    print("="*80)
    print(" "*30 + "DONE!")
    print("="*80)

if __name__ == "__main__":
    main()