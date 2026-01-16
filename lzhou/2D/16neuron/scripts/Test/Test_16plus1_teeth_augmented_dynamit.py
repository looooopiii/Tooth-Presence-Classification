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
    balanced_accuracy_score,
    precision_recall_curve,
    average_precision_score
)

# Image processing
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet50

# =================================================================================
# FUSION STRATEGY ENUM
# =================================================================================
class FusionStrategy(Enum):
    AVERAGE = "average"
    MAX_CONFIDENCE = "max_confidence"
    BEST_ANGLE = "best_angle"
    BEST_N_ANGLES = "best_n_angles"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    JAW_CONFIDENCE = "jaw_confidence"

# =================================================================================
# CONFIGURATION
# =================================================================================
TEST_IMG_DIR = "/home/user/lzhou/week15/render_output/test"
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"

# ========== Augmented Dynamic Loss model ==========
MODEL_PATH = "/home/user/lzhou/week15-17/output/Train2D/Augmented_16plus1teeth_dynamit/augmented_dynamit_best_2d_16plus1teeth.pth"
OUTPUT_DIR = "/home/user/lzhou/week15-17/output/Test2D/augmented_16plus1teeth_dynamit_auto_best"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
NUM_TEETH_POSITIONS = 16  # 16 positions
NUM_OUTPUTS = 17  # 16 teeth + 1 jaw
NUM_SAMPLE_PREDICTIONS = 10
DROPOUT_RATE = 0.5

# ========== FUSION CONFIGURATION ==========
BEST_N = 2

# ========== SELECTION METRIC ==========
SELECTION_METRIC = 'macro_f1'

# ========== METRIC SETTINGS ==========
MACRO_SUPPORT_MIN = 5  # Only include positions with >= this many positives in macro

# ========== PR CURVES / THRESHOLD TUNING ==========
ENABLE_PR_PLOTS = True
ENABLE_THRESHOLD_TUNING = True
THRESHOLD_STRATEGY = "max_f1"  # "min_precision" or "max_f1"
MIN_PRECISION = 0.2  # Only used for "min_precision"
THRESHOLDS_FILENAME = "per_tooth_thresholds.json"
CALIBRATION_RATIO = 0.2
CALIBRATION_SEED = 42


# FDI Notation - 16 positions mapping
POSITION_TO_FDI_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
POSITION_TO_FDI_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

# Create reverse mappings
FDI_UPPER_TO_POSITION = {fdi: pos for pos, fdi in enumerate(POSITION_TO_FDI_UPPER)}
FDI_LOWER_TO_POSITION = {fdi: pos for pos, fdi in enumerate(POSITION_TO_FDI_LOWER)}

# All valid FDI labels
UPPER_FDI = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
ALL_FDI = sorted(UPPER_FDI + LOWER_FDI)

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
# MODEL DEFINITION (16+1 outputs)
# =================================================================================
class ResNetMultiLabel16Plus1(nn.Module):
    def __init__(self, backbone="resnet18", num_outputs=17, dropout_rate=0.5):
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
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# =================================================================================
# LABEL CONVERSION: CSV (32 teeth) → 16+1 format
# =================================================================================
def convert_csv_labels_to_16plus1(csv_row, jaw_type):
    """Convert CSV labels (32 teeth format) to 16+1 format"""
    output_vector = np.zeros(NUM_OUTPUTS, dtype=np.float32)
    tooth_missing = np.zeros(NUM_TEETH_POSITIONS, dtype=np.float32)
    
    if jaw_type == 'upper':
        for pos, fdi in enumerate(POSITION_TO_FDI_UPPER):
            col_name = str(fdi)
            if col_name in csv_row.index and pd.notna(csv_row[col_name]):
                tooth_missing[pos] = float(csv_row[col_name])
            else:
                tooth_missing[pos] = 0.0
        jaw_label = 0.0  # Upper = 0
    else:  # lower
        for pos, fdi in enumerate(POSITION_TO_FDI_LOWER):
            col_name = str(fdi)
            if col_name in csv_row.index and pd.notna(csv_row[col_name]):
                tooth_missing[pos] = float(csv_row[col_name])
            else:
                tooth_missing[pos] = 0.0
        jaw_label = 1.0  # Lower = 1
    
    output_vector[:NUM_TEETH_POSITIONS] = tooth_missing
    output_vector[NUM_TEETH_POSITIONS] = jaw_label
    
    return output_vector

# =================================================================================
# FUSION METHODS (adapted for 16+1)
# =================================================================================
def calculate_confidence_score_16plus1(probs):
    """Calculate confidence for 16+1 output (only consider teeth positions)"""
    teeth_probs = probs[:NUM_TEETH_POSITIONS]
    return np.mean(np.abs(teeth_probs - 0.5))

def fuse_predictions_16plus1(probs_list, strategy=FusionStrategy.AVERAGE, n_best=2):
    """Fuse predictions from multiple angles (16+1 format)"""
    if len(probs_list) == 0:
        return None
    
    if len(probs_list) == 1:
        return probs_list[0]
    
    probs_array = np.array(probs_list)
    
    # Separate teeth (0-15) and jaw (16)
    teeth_probs = probs_array[:, :NUM_TEETH_POSITIONS]
    jaw_probs = probs_array[:, NUM_TEETH_POSITIONS]
    
    # Fuse teeth predictions
    if strategy == FusionStrategy.AVERAGE:
        fused_teeth = np.mean(teeth_probs, axis=0)
    
    elif strategy == FusionStrategy.MAX_CONFIDENCE:
        confidence_per_tooth = np.abs(teeth_probs - 0.5)
        best_angle_per_tooth = np.argmax(confidence_per_tooth, axis=0)
        fused_teeth = np.array([teeth_probs[best_angle_per_tooth[i], i] for i in range(NUM_TEETH_POSITIONS)])
    
    elif strategy == FusionStrategy.BEST_ANGLE:
        confidences = [calculate_confidence_score_16plus1(p) for p in probs_list]
        best_idx = np.argmax(confidences)
        fused_teeth = teeth_probs[best_idx]
    
    elif strategy == FusionStrategy.BEST_N_ANGLES:
        confidences = [calculate_confidence_score_16plus1(p) for p in probs_list]
        n = min(n_best, len(probs_list))
        top_n_indices = np.argsort(confidences)[-n:]
        fused_teeth = np.mean(teeth_probs[top_n_indices], axis=0)
    
    elif strategy == FusionStrategy.MAJORITY_VOTE:
        binary_preds = (teeth_probs > 0.5).astype(float)
        fused_teeth = np.mean(binary_preds, axis=0)
    
    elif strategy == FusionStrategy.WEIGHTED_AVERAGE:
        confidences = np.array([calculate_confidence_score_16plus1(p) for p in probs_list])
        if confidences.sum() == 0:
            weights = np.ones(len(probs_list)) / len(probs_list)
        else:
            weights = confidences / confidences.sum()
        fused_teeth = np.average(teeth_probs, axis=0, weights=weights)
    
    elif strategy == FusionStrategy.JAW_CONFIDENCE:
        confidences = [calculate_confidence_score_16plus1(p) for p in probs_list]
        best_idx = np.argmax(confidences)
        fused_teeth = teeth_probs[best_idx]
    
    else:
        fused_teeth = np.mean(teeth_probs, axis=0)
    
    # Fuse jaw predictions (majority vote)
    fused_jaw = np.mean(jaw_probs)
    
    # Combine
    fused_output = np.zeros(NUM_OUTPUTS, dtype=np.float32)
    fused_output[:NUM_TEETH_POSITIONS] = fused_teeth
    fused_output[NUM_TEETH_POSITIONS] = fused_jaw
    
    return fused_output

# =================================================================================
# DATA LOADING
# =================================================================================
def load_test_labels(csv_path):
    """Load test labels and convert to 16+1 format"""
    df = pd.read_csv(csv_path, dtype={'new_id': str})
    df['new_id'] = df['new_id'].astype(str).str.strip().str.lower().str.replace('-', '_')
    df.columns = [str(c) for c in df.columns]
    
    labels_dict = {}
    jaw_type_dict = {}
    
    for _, row in df.iterrows():
        case_id = row['new_id']
        
        is_upper = '_upper' in case_id and '_lower' not in case_id
        is_lower = '_lower' in case_id
        
        if is_upper:
            jaw_type = 'upper'
        elif is_lower:
            jaw_type = 'lower'
        else:
            jaw_type = 'unknown'
            continue
        
        label_vector = convert_csv_labels_to_16plus1(row, jaw_type)
        
        labels_dict[case_id] = label_vector
        jaw_type_dict[case_id] = jaw_type
    
    print(f"  [Info] Upper jaw samples: {sum(1 for v in jaw_type_dict.values() if v == 'upper')}")
    print(f"  [Info] Lower jaw samples: {sum(1 for v in jaw_type_dict.values() if v == 'lower')}")
    
    return labels_dict, jaw_type_dict

def find_test_images(img_dir, labels_dict):
    """Scans directory and matches images to CSV IDs"""
    grouped = {}
    path = Path(img_dir)
    
    if not path.exists():
        print(f"✗ Error: Directory not found: {img_dir}")
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


def split_case_ids(case_ids, calibration_ratio, seed):
    if calibration_ratio <= 0 or len(case_ids) < 2:
        return [], list(case_ids)
    ids = list(case_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n_cal = max(1, int(len(ids) * calibration_ratio))
    n_cal = min(n_cal, len(ids) - 1)
    return ids[:n_cal], ids[n_cal:]


def subset_by_ids(grouped_imgs, labels_dict, jaw_type_dict, keep_ids):
    keep = set(keep_ids)
    grouped_sub = {k: grouped_imgs[k] for k in keep if k in grouped_imgs}
    labels_sub = {k: labels_dict[k] for k in keep if k in labels_dict}
    jaw_sub = {k: jaw_type_dict[k] for k in keep if k in jaw_type_dict}
    return grouped_sub, labels_sub, jaw_sub

# =================================================================================
# INFERENCE
# =================================================================================
def test_model(model, grouped_imgs, labels_dict, jaw_type_dict, device, transform, 
               strategy=FusionStrategy.AVERAGE, n_best=2, show_progress=True):
    """Run inference with 16+1 model"""
    model.eval()
    all_preds, all_targets, all_ids = [], [], []
    
    fusion_stats = {
        'num_angles_per_case': [],
        'confidence_scores': [],
        'jaw_accuracy': []
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
                    print(f" Read Error: {e}")
            
            if not probs_list:
                continue
            
            fused_probs = fuse_predictions_16plus1(
                probs_list, 
                strategy=strategy, 
                n_best=n_best
            )
            
            if fused_probs is None:
                continue
            
            all_preds.append(fused_probs)
            all_targets.append(labels)
            all_ids.append(case_id)
            
            # Stats
            fusion_stats['num_angles_per_case'].append(len(probs_list))
            fusion_stats['confidence_scores'].append(calculate_confidence_score_16plus1(fused_probs))
            
            # Jaw classification accuracy
            pred_jaw = 1 if fused_probs[NUM_TEETH_POSITIONS] > 0.5 else 0
            true_jaw = int(labels[NUM_TEETH_POSITIONS])
            fusion_stats['jaw_accuracy'].append(1 if pred_jaw == true_jaw else 0)
            
    return np.array(all_preds), np.array(all_targets), all_ids, fusion_stats

# =================================================================================
# METRICS CALCULATION (16+1 format)
# =================================================================================
def calculate_metrics_16plus1(preds, targets, jaw_type_dict, all_ids, preds_bin=None):
    """Calculate metrics for 16+1 format"""
    if len(preds) == 0:
        return {}
    
    if preds_bin is None:
        preds_bin = (preds > 0.5).astype(int)
    else:
        preds_bin = preds_bin.astype(int)
    targets_bin = targets.astype(int)
    
    # Separate teeth and jaw
    teeth_preds = preds_bin[:, :NUM_TEETH_POSITIONS]
    teeth_targets = targets_bin[:, :NUM_TEETH_POSITIONS]
    jaw_preds = preds_bin[:, NUM_TEETH_POSITIONS]
    jaw_targets = targets_bin[:, NUM_TEETH_POSITIONS]
    
    # Jaw classification metrics
    jaw_acc = accuracy_score(jaw_targets, jaw_preds)
    jaw_p, jaw_r, jaw_f1, _ = precision_recall_fscore_support(
        jaw_targets, jaw_preds, average='binary', zero_division=0, pos_label=1
    )
    
    # Overall teeth metrics (missing class)
    flat_p = teeth_preds.flatten()
    flat_t = teeth_targets.flatten()
    # Missing teeth are labeled as 1; compute recall for the missing class.
    missing_p, missing_r, missing_f1, _ = precision_recall_fscore_support(
        flat_t, flat_p, average='binary', zero_division=0, pos_label=1
    )
    acc = accuracy_score(flat_t, flat_p)
    bal_acc = balanced_accuracy_score(flat_t, flat_p)

    # Per-position metrics
    per_position = OrderedDict()
    
    for i, case_id in enumerate(all_ids):
        jaw_type = jaw_type_dict.get(case_id, 'unknown')
        
        if jaw_type == 'upper':
            fdi_list = POSITION_TO_FDI_UPPER
            jaw_prefix = 'Upper'
        elif jaw_type == 'lower':
            fdi_list = POSITION_TO_FDI_LOWER
            jaw_prefix = 'Lower'
        else:
            continue
        
        for pos in range(NUM_TEETH_POSITIONS):
            fdi = fdi_list[pos]
            key = f"{jaw_prefix}_{fdi}"
            
            if key not in per_position:
                per_position[key] = {
                    'preds': [],
                    'targets': [],
                    'position': pos,
                    'fdi': fdi,
                    'jaw': jaw_type
                }
            
            per_position[key]['preds'].append(teeth_preds[i, pos])
            per_position[key]['targets'].append(teeth_targets[i, pos])
    
    # Calculate metrics for each position
    per_position_metrics = OrderedDict()
    for key, data in per_position.items():
        preds_arr = np.array(data['preds'])
        targets_arr = np.array(data['targets'])
        
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(
            targets_arr, preds_arr, average='binary', zero_division=0, pos_label=1
        )
        acc_t = accuracy_score(targets_arr, preds_arr)
        support = int(targets_arr.sum())
        
        per_position_metrics[key] = {
            'position': data['position'],
            'fdi': data['fdi'],
            'jaw': data['jaw'],
            'precision': float(p_t),
            'recall': float(r_t),
            'f1': float(f1_t),
            'accuracy': float(acc_t),
            'support': support
        }
    
    def macro_from_positions(metrics_dict, support_min=1, jaw_filter=None):
        keys = []
        for k, m in metrics_dict.items():
            if m['support'] >= support_min:
                if jaw_filter is None or m['jaw'] == jaw_filter:
                    keys.append(k)
        if not keys:
            return {
                'macro_precision': 0.0,
                'macro_recall': 0.0,
                'macro_f1': 0.0,
                'macro_positions': 0
            }
        return {
            'macro_precision': float(np.mean([metrics_dict[k]['precision'] for k in keys])),
            'macro_recall': float(np.mean([metrics_dict[k]['recall'] for k in keys])),
            'macro_f1': float(np.mean([metrics_dict[k]['f1'] for k in keys])),
            'macro_positions': len(keys)
        }
    
    macro_upper = macro_from_positions(per_position_metrics, support_min=MACRO_SUPPORT_MIN, jaw_filter='upper')
    macro_lower = macro_from_positions(per_position_metrics, support_min=MACRO_SUPPORT_MIN, jaw_filter='lower')
    macros_present = [m for m in (macro_upper, macro_lower) if m['macro_positions'] > 0]
    if macros_present:
        macro_all = {
            'macro_precision': float(np.mean([m['macro_precision'] for m in macros_present])),
            'macro_recall': float(np.mean([m['macro_recall'] for m in macros_present])),
            'macro_f1': float(np.mean([m['macro_f1'] for m in macros_present])),
            'macro_positions': int(np.mean([m['macro_positions'] for m in macros_present])),
            'macro_jaws': len(macros_present)
        }
    else:
        macro_all = {
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'macro_positions': 0,
            'macro_jaws': 0
        }
    
    # Per-jaw classification accuracy (upper vs lower), consistent with training.
    upper_mask = jaw_targets == 0
    lower_mask = jaw_targets == 1
    upper_acc = accuracy_score(jaw_targets[upper_mask], jaw_preds[upper_mask]) if upper_mask.any() else 0
    lower_acc = accuracy_score(jaw_targets[lower_mask], jaw_preds[lower_mask]) if lower_mask.any() else 0
    
    return {
        'overall_missing': {
            'missing_precision': float(missing_p),
            'missing_recall': float(missing_r),
            'missing_f1': float(missing_f1),
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc)
        },
        'overall_macro': {
            'macro_precision': macro_all['macro_precision'],
            'macro_recall': macro_all['macro_recall'],
            'macro_f1': macro_all['macro_f1'],
            'macro_support_min': MACRO_SUPPORT_MIN,
            'macro_positions_per_jaw': macro_all['macro_positions'],
            'macro_jaws_included': macro_all['macro_jaws']
        },
        'per_jaw_macro': {
            'upper': macro_upper,
            'lower': macro_lower
        },
        'jaw_classification': {
            'jaw_accuracy': float(jaw_acc),
            'jaw_precision': float(jaw_p),
            'jaw_recall': float(jaw_r),
            'jaw_f1': float(jaw_f1)
        },
        'per_jaw': {
            'upper_jaw_accuracy': float(upper_acc),
            'upper_jaw_samples': int(upper_mask.sum()),
            'lower_jaw_accuracy': float(lower_acc),
            'lower_jaw_samples': int(lower_mask.sum())
        },
        'per_position': per_position_metrics
    }

# =================================================================================
# PR CURVES & THRESHOLD TUNING
# =================================================================================
def build_per_position_arrays(preds, targets, jaw_type_dict, all_ids):
    per_position = {}
    for jaw in ('upper', 'lower'):
        for pos in range(NUM_TEETH_POSITIONS):
            per_position[(jaw, pos)] = {'scores': [], 'labels': []}
    
    for i, case_id in enumerate(all_ids):
        jaw = jaw_type_dict.get(case_id, 'unknown')
        if jaw not in ('upper', 'lower'):
            continue
        for pos in range(NUM_TEETH_POSITIONS):
            per_position[(jaw, pos)]['scores'].append(float(preds[i, pos]))
            per_position[(jaw, pos)]['labels'].append(int(targets[i, pos]))
    
    return per_position

def compute_pr_data(per_position_arrays):
    pr_data = {}
    for key, data in per_position_arrays.items():
        labels = np.array(data['labels'], dtype=int)
        scores = np.array(data['scores'], dtype=float)
        if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
            pr_data[key] = {
                'valid': False,
                'support': int(labels.sum())
            }
            continue
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        pr_data[key] = {
            'valid': True,
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'ap': float(ap),
            'support': int(labels.sum())
        }
    return pr_data

def pick_threshold(precision, recall, thresholds, strategy, min_precision):
    if len(thresholds) == 0:
        return 0.5
    prec = precision[:-1]
    rec = recall[:-1]
    
    if strategy == "min_precision":
        valid = np.where(prec >= min_precision)[0]
        if len(valid) > 0:
            idx = valid[np.argmax(rec[valid])]
            return float(thresholds[idx])
    
    f1 = (2 * prec * rec) / (prec + rec + 1e-8)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx])

def compute_thresholds(pr_data, strategy, min_precision):
    thresholds = {
        'upper': [0.5] * NUM_TEETH_POSITIONS,
        'lower': [0.5] * NUM_TEETH_POSITIONS
    }
    for jaw in ('upper', 'lower'):
        for pos in range(NUM_TEETH_POSITIONS):
            data = pr_data.get((jaw, pos))
            if not data or not data.get('valid'):
                continue
            thresholds[jaw][pos] = pick_threshold(
                data['precision'],
                data['recall'],
                data['thresholds'],
                strategy,
                min_precision
            )
    return thresholds

def binarize_with_thresholds(preds, jaw_type_dict, all_ids, thresholds):
    preds_bin = np.zeros_like(preds, dtype=int)
    for i, case_id in enumerate(all_ids):
        jaw = jaw_type_dict.get(case_id, 'unknown')
        if jaw not in ('upper', 'lower'):
            pos_thresholds = [0.5] * NUM_TEETH_POSITIONS
        else:
            pos_thresholds = thresholds[jaw]
        
        for pos in range(NUM_TEETH_POSITIONS):
            preds_bin[i, pos] = 1 if preds[i, pos] >= pos_thresholds[pos] else 0
        preds_bin[i, NUM_TEETH_POSITIONS] = 1 if preds[i, NUM_TEETH_POSITIONS] >= 0.5 else 0
    return preds_bin

def thresholds_to_fdi(thresholds):
    upper = {str(fdi): float(thresholds['upper'][pos]) for pos, fdi in enumerate(POSITION_TO_FDI_UPPER)}
    lower = {str(fdi): float(thresholds['lower'][pos]) for pos, fdi in enumerate(POSITION_TO_FDI_LOWER)}
    return {'upper': upper, 'lower': lower}

def plot_pr_grid(pr_data, jaw, fdi_list, save_path, title):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for idx in range(NUM_TEETH_POSITIONS):
        ax = axes[idx // 4, idx % 4]
        key = (jaw, idx)
        data = pr_data.get(key, {})
        fdi = fdi_list[idx]
        if data.get('valid'):
            ax.plot(data['recall'], data['precision'], color='steelblue', linewidth=1)
            ax.set_title(f"{fdi} (AP={data['ap']:.2f})", fontsize=9)
        else:
            ax.text(0.5, 0.5, "n/a", ha='center', va='center', fontsize=9)
            ax.set_title(f"{fdi} (no pos/neg)", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_micro_pr(preds, targets, save_path, title):
    scores = preds[:, :NUM_TEETH_POSITIONS].flatten()
    labels = targets[:, :NUM_TEETH_POSITIONS].astype(int).flatten()
    if labels.sum() == 0 or labels.sum() == len(labels):
        return
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='darkorange', linewidth=2, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def get_metric_value(metrics, metric_name):
    """Extract specific metric value"""
    if metric_name == 'balanced_accuracy':
        return metrics['overall_missing']['balanced_accuracy']
    elif metric_name == 'macro_f1':
        return metrics['overall_macro']['macro_f1']
    elif metric_name == 'macro_recall':
        return metrics['overall_macro']['macro_recall']
    elif metric_name == 'macro_precision':
        return metrics['overall_macro']['macro_precision']
    elif metric_name == 'accuracy':
        return metrics['overall_missing']['accuracy']
    else:
        return metrics['overall_missing']['balanced_accuracy']

# =================================================================================
# STRATEGY COMPARISON
# =================================================================================
def compare_all_strategies(model, grouped_imgs, labels_dict, jaw_type_dict, device, transform):
    """Compare all fusion strategies"""
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
            preds_bin = None
            thresholds = None
            if ENABLE_THRESHOLD_TUNING:
                per_pos_arrays = build_per_position_arrays(preds, targets, jaw_type_dict, ids)
                pr_data = compute_pr_data(per_pos_arrays)
                thresholds = compute_thresholds(pr_data, THRESHOLD_STRATEGY, MIN_PRECISION)
                preds_bin = binarize_with_thresholds(preds, jaw_type_dict, ids, thresholds)
            metrics = calculate_metrics_16plus1(preds, targets, jaw_type_dict, ids, preds_bin=preds_bin)
            results[strategy.value] = {
                'metrics': metrics,
                'stats': stats,
                'preds': preds,
                'preds_bin': preds_bin,
                'thresholds': thresholds,
                'targets': targets,
                'ids': ids
            }
            
            print(f"  Balanced Accuracy: {metrics['overall_missing']['balanced_accuracy']:.4f}")
            print(f"  Macro F1: {metrics['overall_macro']['macro_f1']:.4f}")
            print(f"  Macro Recall: {metrics['overall_macro']['macro_recall']:.4f}")
            print(f"  Jaw Accuracy: {metrics['jaw_classification']['jaw_accuracy']:.4f}")
            print(f"  Upper Jaw Acc: {metrics['per_jaw']['upper_jaw_accuracy']:.4f}")
            print(f"  Lower Jaw Acc: {metrics['per_jaw']['lower_jaw_accuracy']:.4f}")
    
    return results

def select_best_strategy(results, metric_name='balanced_accuracy'):
    """Select best strategy"""
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
    """Print comparison table"""
    print("\n" + "="*130)
    print(" "*45 + "FUSION STRATEGY COMPARISON (16+1 Architecture)")
    print("="*130)
    
    print(f"\n{'Strategy':<20} {'Bal.Acc':>12} {'Macro F1':>12} {'Macro Rec':>12} {'Jaw Acc':>12} {'Upper Cls':>12} {'Lower Cls':>12}")
    print("-"*130)
    
    best_strategy, best_value = select_best_strategy(results, metric_name)
    
    for strategy, data in results.items():
        m = data['metrics']
        bal_acc = m['overall_missing']['balanced_accuracy']
        macro_f1 = m['overall_macro']['macro_f1']
        macro_rec = m['overall_macro']['macro_recall']
        jaw_acc = m['jaw_classification']['jaw_accuracy']
        upper_acc = m['per_jaw']['upper_jaw_accuracy']
        lower_acc = m['per_jaw']['lower_jaw_accuracy']
        
        marker = " <-- BEST" if strategy == best_strategy else ""
        print(f"{strategy:<20} {bal_acc:>12.4f} {macro_f1:>12.4f} {macro_rec:>12.4f} {jaw_acc:>12.4f} {upper_acc:>12.4f} {lower_acc:>12.4f}{marker}")
    
    print("-"*130)
    print(f"\n Best Strategy (by {metric_name}): {best_strategy} ({best_value:.4f})")
    print("="*130)
    
    return best_strategy

def print_metrics_summary(metrics, strategy_name):
    """Print detailed metrics"""
    micro = metrics['overall_missing']
    macro = metrics['overall_macro']
    per_jaw_macro = metrics.get('per_jaw_macro', {})
    jaw = metrics['jaw_classification']
    per_jaw = metrics['per_jaw']
    
    print("\n" + "="*80)
    print(f" "*10 + f"DETAILED METRICS FOR BEST STRATEGY: {strategy_name}")
    print("="*80)
    
    print(f"\n╔{'═'*78}╗")
    print(f"║{' OVERALL METRICS (MISSING CLASS)':^78}║")
    print(f"╠{'═'*78}╣")
    print(f"║  Missing Precision:{micro['missing_precision']:<10.4f} (TP / (TP + FP)){' '*27}║")
    print(f"║  Missing Recall:   {micro['missing_recall']:<10.4f} (TP / (TP + FN)){' '*27}║")
    print(f"║  Missing F1:       {micro['missing_f1']:<10.4f} (Harmonic mean){' '*34}║")
    print(f"║  Accuracy:         {micro['accuracy']:<10.4f}{' '*48}║")
    print(f"║  Balanced Acc:     {micro['balanced_accuracy']:<10.4f} (Avg of TPR and TNR){' '*24}║")
    print(f"╚{'═'*78}╝")
    
    print(f"\n╔{'═'*78}╗")
    print(f"║{' OVERALL METRICS (MACRO - per jaw, 16 positions)':^78}║")
    print(f"╠{'═'*78}╣")
    print(f"║  Macro Precision:  {macro['macro_precision']:<10.4f}{' '*48}║")
    print(f"║  Macro Recall:     {macro['macro_recall']:<10.4f}{' '*48}║")
    print(f"║  Macro F1:         {macro['macro_f1']:<10.4f}{' '*48}║")
    print(f"║  Support Filter:   >= {macro['macro_support_min']:<5d} positives ({macro['macro_positions_per_jaw']}/16 positions, {macro['macro_jaws_included']} jaws){' '*6}║")
    print(f"╚{'═'*78}╝")

    if per_jaw_macro:
        upper = per_jaw_macro.get('upper', {})
        lower = per_jaw_macro.get('lower', {})
        if upper or lower:
            print(f"\n╔{'═'*78}╗")
            print(f"║{' PER-JAW MACRO (support-filtered)':^78}║")
            print(f"╠{'═'*78}╣")
            print(f"║  Upper Macro F1:   {upper.get('macro_f1', 0.0):<10.4f} ({upper.get('macro_positions', 0)}/16 positions){' '*28}║")
            print(f"║  Lower Macro F1:   {lower.get('macro_f1', 0.0):<10.4f} ({lower.get('macro_positions', 0)}/16 positions){' '*28}║")
            print(f"╚{'═'*78}╝")
    
    print(f"\n╔{'═'*78}╗")
    print(f"║{' JAW CLASSIFICATION METRICS':^78}║")
    print(f"╠{'═'*78}╣")
    print(f"║  Jaw Accuracy:     {jaw['jaw_accuracy']:<10.4f} (Upper=0, Lower=1){' '*29}║")
    print(f"║  Jaw Precision:    {jaw['jaw_precision']:<10.4f}{' '*48}║")
    print(f"║  Jaw Recall:       {jaw['jaw_recall']:<10.4f}{' '*48}║")
    print(f"║  Jaw F1:           {jaw['jaw_f1']:<10.4f}{' '*48}║")
    print(f"╚{'═'*78}╝")
    
    print(f"\n╔{'═'*78}╗")
    print(f"║{' PER-JAW CLASSIFICATION ACCURACY':^78}║")
    print(f"╠{'═'*78}╣")
    print(f"║  Upper Jaw:        {per_jaw['upper_jaw_accuracy']:<10.4f} ({per_jaw['upper_jaw_samples']} samples){' '*36}║")
    print(f"║  Lower Jaw:        {per_jaw['lower_jaw_accuracy']:<10.4f} ({per_jaw['lower_jaw_samples']} samples){' '*36}║")
    print(f"╚{'═'*78}╝")
    
    print("\n" + "-" * 95)
    print(" PER-POSITION METRICS (FDI Notation):")
    print("-" * 95)
    print(f"{'Jaw':<8} {'FDI':<8} {'Pos':<5} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 95)
    
    print("─── UPPER JAW (11-28) ───")
    for fdi in POSITION_TO_FDI_UPPER:
        key = f"Upper_{fdi}"
        if key in metrics['per_position']:
            m = metrics['per_position'][key]
            print(f"{'upper':<8} {m['fdi']:<8} {m['position']:<5} {m['precision']:>10.4f}   {m['recall']:>10.4f}   {m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}")
    
    print("─── LOWER JAW (31-48) ───")
    for fdi in POSITION_TO_FDI_LOWER:
        key = f"Lower_{fdi}"
        if key in metrics['per_position']:
            m = metrics['per_position'][key]
            print(f"{'lower':<8} {m['fdi']:<8} {m['position']:<5} {m['precision']:>10.4f}   {m['recall']:>10.4f}   {m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}")
    
    print("=" * 95)

def print_sample_predictions(ids, preds, targets, jaw_type_dict, num_samples=10, preds_bin=None):
    """Print sample predictions"""
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
        pred_vec = preds_bin[i] if preds_bin is not None else (preds[i] > 0.5).astype(int)
        
        teeth_target = target_vec[:NUM_TEETH_POSITIONS]
        teeth_pred = pred_vec[:NUM_TEETH_POSITIONS]
        
        if jaw_type == 'upper':
            fdi_list = POSITION_TO_FDI_UPPER
        elif jaw_type == 'lower':
            fdi_list = POSITION_TO_FDI_LOWER
        else:
            continue
        
        truth_missing = [fdi_list[j] for j, val in enumerate(teeth_target) if val == 1]
        pred_missing = [fdi_list[j] for j, val in enumerate(teeth_pred) if val == 1]
        
        truth_set = set(truth_missing)
        pred_set = set(pred_missing)
        
        tp = sorted(list(truth_set.intersection(pred_set)))
        fn = sorted(list(truth_set.difference(pred_set)))
        fp = sorted(list(pred_set.difference(truth_set)))
        
        pred_jaw = "Lower" if pred_vec[NUM_TEETH_POSITIONS] == 1 else "Upper"
        true_jaw = "Lower" if int(target_vec[NUM_TEETH_POSITIONS]) == 1 else "Upper"
        jaw_correct = "yes" if pred_jaw == true_jaw else "no"
        
        def fmt_list(lst):
            return str(sorted(lst)) if lst else "None"
        
        print(f"\n┌{'─'*70}┐")
        print(f"│ Case ID: {case_id:<59}│")
        print(f"│ Ground Truth Jaw: {true_jaw:<10} | Predicted Jaw: {pred_jaw:<10} {jaw_correct:<24}│")
        print(f"├{'─'*70}┤")
        print(f"│  Ground Truth (Missing): {str(sorted(truth_missing)):<43}│")
        print(f"│  Prediction (Missing):   {str(sorted(pred_missing)):<43}│")
        print(f"├{'─'*70}┤")
        print(f"│   Correctly Found (TP): {fmt_list(tp):<43}│")
        print(f"│   Missed Teeth (FN):    {fmt_list(fn):<43}│")
        print(f"│   False Alarms (FP):    {fmt_list(fp):<43}│")
        
        correct = len(tp) + (NUM_TEETH_POSITIONS - len(truth_set) - len(fp))
        case_acc = correct / NUM_TEETH_POSITIONS
        print(f"│  Case Accuracy: {case_acc:.2%} ({correct}/{NUM_TEETH_POSITIONS}){' '*37}│")
        print(f"└{'─'*70}┘")
    
    print("\n" + "="*80)

# =================================================================================
# PLOTTING
# =================================================================================
def generate_comparison_plot(results, save_dir):
    """Generate comparison plot"""
    strategies = list(results.keys())
    
    bal_accs = [results[s]['metrics']['overall_missing']['balanced_accuracy'] for s in strategies]
    macro_f1s = [results[s]['metrics']['overall_macro']['macro_f1'] for s in strategies]
    macro_recs = [results[s]['metrics']['overall_macro']['macro_recall'] for s in strategies]
    jaw_accs = [results[s]['metrics']['jaw_classification']['jaw_accuracy'] for s in strategies]
    
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
    axes[0, 1].set_ylim(0, 1)
    best_idx = np.argmax(macro_f1s)
    bars2[best_idx].set_color('red')
    
    # Macro Recall
    bars3 = axes[1, 0].bar(x, macro_recs, color=colors)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_title('Macro Recall', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim(0, 1)
    best_idx = np.argmax(macro_recs)
    bars3[best_idx].set_color('red')
    
    # Jaw Accuracy
    bars4 = axes[1, 1].bar(x, jaw_accs, color=colors)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_title('Jaw Classification Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim(0.7, 1)
    best_idx = np.argmax(jaw_accs)
    bars4[best_idx].set_color('red')
    
    plt.suptitle('Fusion Strategy Comparison (16+1 Architecture)\n(Red = Best)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "fusion_strategy_comparison_16plus1.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Comparison plot saved to {save_dir}/fusion_strategy_comparison_16plus1.png")

def generate_detailed_plots(metrics, preds, targets, save_dir, strategy_name, preds_bin=None):
    """Generate detailed plots for 16+1"""
    per_position = metrics['per_position']
    
    # F1 Score per Jaw
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Upper jaw
    upper_f1s = []
    for fdi in POSITION_TO_FDI_UPPER:
        key = f"Upper_{fdi}"
        if key in per_position:
            upper_f1s.append(per_position[key]['f1'])
        else:
            upper_f1s.append(0.0)
    
    colors_upper = ['coral' if f < 0.85 else 'steelblue' for f in upper_f1s]
    axes[0].bar(range(len(POSITION_TO_FDI_UPPER)), upper_f1s, color=colors_upper)
    axes[0].set_xticks(range(len(POSITION_TO_FDI_UPPER)))
    axes[0].set_xticklabels([str(f) for f in POSITION_TO_FDI_UPPER], rotation=45)
    axes[0].set_title(f'F1 Score - Upper Jaw (11-28)\nStrategy: {strategy_name}', fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=np.mean(upper_f1s), color='red', linestyle='--', label=f'Mean: {np.mean(upper_f1s):.3f}')
    axes[0].legend()
    
    # Lower jaw
    lower_f1s = []
    for fdi in POSITION_TO_FDI_LOWER:
        key = f"Lower_{fdi}"
        if key in per_position:
            lower_f1s.append(per_position[key]['f1'])
        else:
            lower_f1s.append(0.0)
    
    colors_lower = ['coral' if f < 0.85 else 'seagreen' for f in lower_f1s]
    axes[1].bar(range(len(POSITION_TO_FDI_LOWER)), lower_f1s, color=colors_lower)
    axes[1].set_xticks(range(len(POSITION_TO_FDI_LOWER)))
    axes[1].set_xticklabels([str(f) for f in POSITION_TO_FDI_LOWER], rotation=45)
    axes[1].set_title(f'F1 Score - Lower Jaw (31-48)\nStrategy: {strategy_name}', fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=np.mean(lower_f1s), color='red', linestyle='--', label=f'Mean: {np.mean(lower_f1s):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "f1_score_per_jaw_16plus1.png", dpi=150)
    plt.close()
    
    # Confusion Matrix
    if preds_bin is None:
        teeth_preds = (preds[:, :NUM_TEETH_POSITIONS] > 0.5).astype(int)
    else:
        teeth_preds = preds_bin[:, :NUM_TEETH_POSITIONS].astype(int)
    teeth_targets = targets[:, :NUM_TEETH_POSITIONS].astype(int)
    cm = confusion_matrix(teeth_targets.flatten(), teeth_preds.flatten())
    
    plt.figure(figsize=(10, 8))
    cm_percent = cm.astype('float') / cm.sum() * 100
    labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' for j in range(2)] for i in range(2)])
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Present (0)', 'Missing (1)'],
                yticklabels=['Present (0)', 'Missing (1)'],
                annot_kws={'size': 14})
    
    plt.title(f'Confusion Matrix (16+1 Architecture)\nStrategy: {strategy_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    
    tn, fp, fn, tp = cm.ravel()
    plt.figtext(0.5, -0.05, 
                f'TN={tn} | FP={fp} | FN={fn} | TP={tp}\n'
                f'Sensitivity (Recall) = {tp/(tp+fn):.4f} | Specificity = {tn/(tn+fp):.4f}',
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "confusion_matrix_16plus1.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Detailed plots saved to {save_dir}")

# =================================================================================
# MAIN
# =================================================================================
def main():
    print("\n" + "="*80)
    print(" "*5 + "2D MODEL TESTING - 16+1 ARCHITECTURE - AUTO-SELECT BEST STRATEGY")
    print("="*80)
    print(f" Selection Metric: {SELECTION_METRIC}")
    print("="*80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/6] Using device: {device}")

    print(f"\n[2/6] Loading 16+1 model from: {MODEL_PATH}")
    model = ResNetMultiLabel16Plus1(
        backbone="resnet18",
        num_outputs=NUM_OUTPUTS,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f" Model loaded successfully (16 teeth + 1 jaw = 17 outputs)")
    except Exception as e:
        print(f" Error loading model: {e}")
        return

    print(f"\n[3/6] Loading test labels from: {TEST_LABELS_CSV}")
    labels_dict, jaw_type_dict = load_test_labels(TEST_LABELS_CSV)
    print(f" Loaded labels for {len(labels_dict)} cases (converted to 16+1 format)")
    
    print(f"\n[4/6] Finding test images in: {TEST_IMG_DIR}")
    grouped_imgs = find_test_images(TEST_IMG_DIR, labels_dict)
    
    if len(grouped_imgs) == 0:
        print(" No matches found.")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_ids = sorted(grouped_imgs.keys())
    cal_ids, test_ids = split_case_ids(all_ids, CALIBRATION_RATIO, CALIBRATION_SEED)
    if not test_ids:
        print(" [Warn] Not enough cases for a calibration/test split; using all cases for testing.")
        test_ids = all_ids
        cal_ids = []

    cal_grouped, cal_labels, cal_jaws = subset_by_ids(
        grouped_imgs, labels_dict, jaw_type_dict, cal_ids
    )
    test_grouped, test_labels, test_jaws = subset_by_ids(
        grouped_imgs, labels_dict, jaw_type_dict, test_ids
    )

    if cal_grouped:
        print(f"\n[5/6] Comparing all fusion strategies on calibration split...")
        print(f" Calibration cases: {len(cal_grouped)} | Test cases: {len(test_grouped)}")
        results = compare_all_strategies(
            model, cal_grouped, cal_labels, cal_jaws, device, transform
        )
    else:
        print(f"\n[5/6] Comparing all fusion strategies on test set (no calibration split)...")
        results = compare_all_strategies(
            model, test_grouped, test_labels, test_jaws, device, transform
        )

    best_strategy = print_comparison_table(results, SELECTION_METRIC)
    generate_comparison_plot(results, OUTPUT_DIR)

    best_thresholds = results[best_strategy].get('thresholds')

    eval_grouped = test_grouped if test_grouped else cal_grouped
    eval_labels = test_labels if test_grouped else cal_labels
    eval_jaws = test_jaws if test_grouped else cal_jaws

    best_preds, best_targets, best_ids, best_stats = test_model(
        model, eval_grouped, eval_labels, eval_jaws, device, transform,
        strategy=FusionStrategy(best_strategy), n_best=BEST_N
    )
    best_preds_bin = None
    if ENABLE_THRESHOLD_TUNING and best_thresholds is not None:
        best_preds_bin = binarize_with_thresholds(best_preds, eval_jaws, best_ids, best_thresholds)

    best_metrics = calculate_metrics_16plus1(
        best_preds, best_targets, eval_jaws, best_ids, preds_bin=best_preds_bin
    )

    print(f"\n[6/6] Generating detailed output for best strategy: {best_strategy}")
    print_metrics_summary(best_metrics, best_strategy)
    print_sample_predictions(
        best_ids, best_preds, best_targets, eval_jaws, NUM_SAMPLE_PREDICTIONS, preds_bin=best_preds_bin
    )
    generate_detailed_plots(
        best_metrics, best_preds, best_targets, OUTPUT_DIR, best_strategy, preds_bin=best_preds_bin
    )

    pr_data = None
    if ENABLE_PR_PLOTS:
        per_pos_arrays = build_per_position_arrays(best_preds, best_targets, eval_jaws, best_ids)
        pr_data = compute_pr_data(per_pos_arrays)

    if ENABLE_PR_PLOTS and pr_data is not None:
        pr_dir = Path(OUTPUT_DIR) / "pr_curves"
        pr_dir.mkdir(parents=True, exist_ok=True)
        plot_pr_grid(
            pr_data,
            "upper",
            POSITION_TO_FDI_UPPER,
            pr_dir / "pr_upper_16plus1.png",
            "PR Curves - Upper Jaw (Missing=1)"
        )
        plot_pr_grid(
            pr_data,
            "lower",
            POSITION_TO_FDI_LOWER,
            pr_dir / "pr_lower_16plus1.png",
            "PR Curves - Lower Jaw (Missing=1)"
        )
        plot_micro_pr(
            best_preds,
            best_targets,
            pr_dir / "pr_micro_teeth.png",
            "Micro PR - Teeth (Missing=1)"
        )
        print(f" PR plots saved to {pr_dir}")

    if ENABLE_THRESHOLD_TUNING and best_thresholds is not None:
        tuned_bin = best_preds_bin
        tuned_metrics = calculate_metrics_16plus1(
            best_preds, best_targets, eval_jaws, best_ids, preds_bin=tuned_bin
        )
        print("\n" + "="*80)
        print(" "*10 + "THRESHOLD-TUNED METRICS (TEETH, CALIBRATION THRESHOLDS)")
        print("="*80)
        print(f"  Strategy: {THRESHOLD_STRATEGY} (min_precision={MIN_PRECISION})")
        print(f"  Balanced Accuracy: {tuned_metrics['overall_missing']['balanced_accuracy']:.4f}")
        print(f"  Macro Recall:      {tuned_metrics['overall_macro']['macro_recall']:.4f}")
        print(f"  Macro F1:          {tuned_metrics['overall_macro']['macro_f1']:.4f}")
        print(f"  Missing Precision: {tuned_metrics['overall_missing']['missing_precision']:.4f}")
        print(f"  Missing Recall:    {tuned_metrics['overall_missing']['missing_recall']:.4f}")
        print("="*80)

        thresholds_path = Path(OUTPUT_DIR) / THRESHOLDS_FILENAME
        with open(thresholds_path, 'w') as f:
            json.dump({
                'strategy': THRESHOLD_STRATEGY,
                'min_precision': MIN_PRECISION,
                'calibration_ratio': CALIBRATION_RATIO,
                'thresholds_fdi': thresholds_to_fdi(best_thresholds)
            }, f, indent=2)
        print(f" Thresholds saved to {thresholds_path}")
    
    # Save results
    results_file = Path(OUTPUT_DIR) / "test_results_16plus1.json"
    
    calibration_used = bool(cal_grouped)
    selection_source = "calibration" if calibration_used else "test"
    json_results = {
        'architecture': '16+1 (16 positions + 1 jaw classifier)',
        'selection_metric': SELECTION_METRIC,
        'calibration_info': {
            'used': calibration_used,
            'ratio': CALIBRATION_RATIO,
            'calibration_cases': len(cal_grouped),
            'test_cases': len(eval_grouped),
            'selection_source': selection_source
        },
        'best_strategy': best_strategy,
        'all_strategies': {},
        'test_best': {
            'strategy': best_strategy,
            'metrics': {
                'overall_missing': best_metrics['overall_missing'],
                'overall_macro': best_metrics['overall_macro'],
                'jaw_classification': best_metrics['jaw_classification'],
                'per_jaw': best_metrics['per_jaw'],
                'per_position': {str(k): v for k, v in best_metrics['per_position'].items()}
            },
            'stats': {
                'avg_angles_per_case': float(np.mean(best_stats['num_angles_per_case'])) if best_stats['num_angles_per_case'] else 0.0,
                'avg_confidence': float(np.mean(best_stats['confidence_scores'])) if best_stats['confidence_scores'] else 0.0,
                'jaw_accuracy': float(np.mean(best_stats['jaw_accuracy'])) if best_stats['jaw_accuracy'] else 0.0
            }
        }
    }
    
    for strategy, data in results.items():
        json_results['all_strategies'][strategy] = {
            'metrics': {
                'overall_missing': data['metrics']['overall_missing'],
                'overall_macro': data['metrics']['overall_macro'],
                'jaw_classification': data['metrics']['jaw_classification'],
                'per_jaw': data['metrics']['per_jaw'],
                'per_position': {str(k): v for k, v in data['metrics']['per_position'].items()}
            },
            'stats': {
                'avg_angles_per_case': float(np.mean(data['stats']['num_angles_per_case'])),
                'avg_confidence': float(np.mean(data['stats']['confidence_scores'])),
                'jaw_accuracy': float(np.mean(data['stats']['jaw_accuracy']))
            }
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n All results saved to {results_file}")
    
    # Summary
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY")
    print("="*80)
    print(f"  Architecture: 16+1 (16 positions + 1 jaw)")
    print(f"  Best Strategy: {best_strategy}")
    print(f"  Balanced Accuracy: {best_metrics['overall_missing']['balanced_accuracy']:.4f}")
    print(f"  Macro Recall: {best_metrics['overall_macro']['macro_recall']:.4f}")
    print(f"  Macro F1: {best_metrics['overall_macro']['macro_f1']:.4f}")
    print(f"  Jaw Classification: {best_metrics['jaw_classification']['jaw_accuracy']:.4f}")
    print(f"  Upper Jaw Acc: {best_metrics['per_jaw']['upper_jaw_accuracy']:.4f}")
    print(f"  Lower Jaw Acc: {best_metrics['per_jaw']['lower_jaw_accuracy']:.4f}")
    print("="*80)
    print(" "*30 + "DONE!")
    print("="*80)

if __name__ == "__main__":
    main()
