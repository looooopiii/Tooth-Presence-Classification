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


# =================================================================================
# CONFIGURATION
# =================================================================================
TEST_IMG_DIR = "/home/user/lzhou/week9/Clean_Test_Data"
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"
# Point to the new 17-output model
MODEL_PATH = "/home/user/lzhou/week11/output/Train2D/normal/resnet18_bce_best_2d.pth"
OUTPUT_DIR = "/home/user/lzhou/week11/output/Test2D/normal_17out"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Change from 32 teeth to 17 outputs
# NUM_TEETH = 32
NUM_OUTPUTS = 17
NUM_TEETH_PER_JAW = 16
NUM_SAMPLE_PREDICTIONS = 5  # How many random samples to print in the terminal
IMG_SIZE = 320

# Remove old 32-FDI maps and add new 16-FDI maps
# VALID_FDI_LABELS = sorted([...])
# FDI_TO_INDEX = ...
# INDEX_TO_FDI = ...

# new mapping for 16 teeth per jaw
# upperjaw: 18-11, 21-28 reflect to 0-15
UPPER_FDI_TO_IDX16 = {
    18: 0, 17: 1, 16: 2, 15: 3, 14: 4, 13: 5, 12: 6, 11: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15
}
# lowerjaw: 48-41, 31-38 reflect to 0-15
LOWER_FDI_TO_IDX16 = {
    48: 0, 47: 1, 46: 2, 45: 3, 44: 4, 43: 5, 42: 6, 41: 7,
    31: 8, 32: 9, 33: 10, 34: 11, 35: 12, 36: 13, 37: 14, 38: 15
}

# Add reverse maps for reporting
IDX16_TO_UPPER_FDI = {v: k for k, v in UPPER_FDI_TO_IDX16.items()}
IDX16_TO_LOWER_FDI = {v: k for k, v in LOWER_FDI_TO_IDX16.items()}

# Get all valid FDI labels for label loading
ALL_VALID_FDI = sorted(list(UPPER_FDI_TO_IDX16.keys()) + list(LOWER_FDI_TO_IDX16.keys()))


# =================================================================================
# STRING / NAME NORMALIZATION (Unchanged)
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
# MODEL DEFINITION
# =================================================================================

class ResNetMultiLabel(nn.Module):
    def __init__(self, backbone="resnet18", num_outputs=17, pretrained=True):
        super().__init__()
        backbone = (backbone or "resnet18").lower()
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            in_feats = self.backbone.fc.in_features
        else:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            in_feats = self.backbone.fc.in_features
        # Use num_outputs
        self.backbone.fc = nn.Linear(in_feats, num_outputs)

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

# rewritten for 17 outputs
def load_test_labels(csv_path):
    """
    Loads ground truth labels into a dict keyed by normalized case_id.
    Generates a 17-element label vector (16 teeth + 1 jaw).
    The jaw is inferred from the normalized case_id (e.g., "..._lower").
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
            raw_stem = str(row[id_col]).replace('.png', '').replace('.ply', '')
            case_key = normalize_png_stem_to_newid(raw_stem)

        # Initialize 17-output vector
        label_vector = np.zeros(NUM_OUTPUTS, dtype=np.float32)

        # Determine Jaw and Mapping
        mapping = None
        jaw_label = 0.0  # Default to upper
        if case_key.endswith('_upper'):
            mapping = UPPER_FDI_TO_IDX16
            jaw_label = 0.0
        elif case_key.endswith('_lower'):
            mapping = LOWER_FDI_TO_IDX16
            jaw_label = 1.0
        else:
            # Cannot determine jaw, skip this sample
            # print(f"[Warning] Skipping {case_key}: Cannot determine jaw from ID.")
            continue
            
        # Fill 16-teeth vector (1=missing, 0=present)
        present_teeth = np.zeros(NUM_TEETH_PER_JAW, dtype=np.float32)
        for tooth_fdi in mapping.keys(): # Only check FDIs for this jaw
            key = str(tooth_fdi)
            if key in df.columns and pd.notna(row[key]) and int(row[key]) == 1:
                idx16 = mapping[tooth_fdi]
                present_teeth[idx16] = 1.0
        
        # Invert to get *missing* teeth (1=missing)
        missing_teeth = 1.0 - present_teeth
        
        # Assign to final 17-element vector
        label_vector[:NUM_TEETH_PER_JAW] = missing_teeth
        label_vector[NUM_TEETH_PER_JAW] = jaw_label # Last element is jaw

        labels_dict[case_key] = label_vector
        
    return labels_dict

# =================================================================================
# INFERENCE AND METRICS
# =================================================================================

def test_model(model, test_items, device, transform):
    model.eval()
    all_preds, all_targets, all_ids = [], [], []
    with torch.no_grad():
        for img_path, case_id, labels in tqdm(test_items, desc="Testing"):
            x = load_image_tensor(img_path, transform).to(device)
            logits = model(x) # Shape [1, 17]
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy()[0])
            all_targets.append(labels)
            all_ids.append(case_id)
    return np.array(all_preds), np.array(all_targets), all_ids

def calculate_classification_metrics(pred_teeth, target_teeth):
    """Calculates micro-average metrics for 16 teeth predictions"""
    pred = (pred_teeth > 0.5).cpu().numpy().astype(int)
    target = target_teeth.cpu().numpy().astype(int)
    
    pred_flat, target_flat = pred.flatten(), target.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(target_flat, pred_flat, average='binary', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy_score(target_flat, pred_flat)}

def calculate_jaw_accuracy(pred_jaw, target_jaw):
    """Calculates accuracy for jaw classification"""
    pred = (pred_jaw > 0.5).cpu().numpy().astype(int)
    target = target_jaw.cpu().numpy().astype(int)
    return {'jaw_accuracy': accuracy_score(target, pred)}


# Rewritten to use the new metric functions
def calculate_metrics(preds, targets):
    """
    Calculates metrics separately for teeth (16) and jaw (1).
    preds/targets shape: [N, 17]
    """
    if len(preds) == 0: return {}

    # Split data
    # Raw probabilities
    probs_teeth = preds[:, :NUM_TEETH_PER_JAW]
    probs_jaw = preds[:, NUM_TEETH_PER_JAW]
    
    # Binary predictions
    preds_bin_teeth = (probs_teeth > 0.5)
    preds_bin_jaw = (probs_jaw > 0.5)

    # Targets
    targets_teeth = targets[:, :NUM_TEETH_PER_JAW]
    targets_jaw = targets[:, NUM_TEETH_PER_JAW]

    # Calculate metrics
    # Use torch tensors for simplicity
    teeth_metrics = calculate_classification_metrics(
        torch.from_numpy(preds_bin_teeth), 
        torch.from_numpy(targets_teeth)
    )
    
    jaw_metrics = calculate_jaw_accuracy(
        torch.from_numpy(preds_bin_jaw),
        torch.from_numpy(targets_jaw)
    )

    # Add raw data for plots
    teeth_metrics['flat_probs'] = probs_teeth.flatten()
    teeth_metrics['flat_targets'] = targets_teeth.flatten()
    jaw_metrics['probs'] = probs_jaw
    jaw_metrics['targets'] = targets_jaw

    return {
        'teeth_metrics': teeth_metrics,
        'jaw_metrics': jaw_metrics
    }

# =================================================================================
# REPORTING AND VISUALIZATION
# =================================================================================

def print_metrics_summary(metrics):
    """Prints a formatted summary of the new test metrics."""
    print("\n" + "="*80 + "\n" + " "*25 + "TESTING METRICS SUMMARY" + "\n" + "="*80)
    
    teeth_metrics = metrics['teeth_metrics']
    print("\n OVERALL TEETH METRICS (Micro-Avg, Positive: Missing Tooth):")
    print(f"  - Precision: {teeth_metrics['precision']:.4f}")
    print(f"  - Recall:    {teeth_metrics['recall']:.4f}")
    print(f"  - F1 Score:  {teeth_metrics['f1']:.4f}")
    print(f"  - Accuracy:  {teeth_metrics['accuracy']:.4f}")
    
    jaw_metrics = metrics['jaw_metrics']
    print("\n OVERALL JAW METRICS (0=Upper, 1=Lower):")
    print(f"  - Accuracy:  {jaw_metrics['jaw_accuracy']:.4f}")
    
    print("\n" + "="*80)


# Rewritten for 17-output logic
def print_sample_predictions(ids, preds, targets, num_samples):
    """Prints a clear, readable comparison for random samples."""
    print("\n" + "="*80 + "\n" + " "*28 + "SAMPLE PREDICTIONS" + "\n" + "="*80)
    if len(ids) < num_samples: num_samples = len(ids)
    
    sample_indices = random.sample(range(len(ids)), num_samples)
    for i in sample_indices:
        case_id = ids[i]
        target_labels_17 = targets[i]
        pred_probs_17 = preds[i]

        # --- JAW PREDICTION ---
        true_jaw_label = int(target_labels_17[NUM_TEETH_PER_JAW])
        pred_jaw_label = int(pred_probs_17[NUM_TEETH_PER_JAW] > 0.5)
        true_jaw_str = 'Lower' if true_jaw_label == 1 else 'Upper'
        pred_jaw_str = 'Lower' if pred_jaw_label == 1 else 'Upper'
        
        print(f"\n Case ID: {case_id}\n" + "-"*50)
        print(f"  Ground Truth Jaw: {true_jaw_str}")
        print(f"  Prediction Jaw:   {pred_jaw_str} (Prob: {pred_probs_17[NUM_TEETH_PER_JAW]:.3f})")
        print("-"*50)

        # --- TEETH PREDICTION ---
        # Use the TRUE jaw to map ground truth indices to FDI
        true_reverse_map = IDX16_TO_LOWER_FDI if true_jaw_str == 'Lower' else IDX16_TO_UPPER_FDI
        # Use the PREDICTED jaw to map prediction indices to FDI
        pred_reverse_map = IDX16_TO_LOWER_FDI if pred_jaw_str == 'Lower' else IDX16_TO_UPPER_FDI
        
        target_labels_16 = target_labels_17[:NUM_TEETH_PER_JAW]
        pred_labels_16 = (pred_probs_17[:NUM_TEETH_PER_JAW] > 0.5).astype(int)

        truth_missing = {true_reverse_map[j] for j, label in enumerate(target_labels_16) if label == 1}
        pred_missing = {pred_reverse_map[j] for j, label in enumerate(pred_labels_16) if label == 1}

        correctly_found_missing = sorted(list(truth_missing.intersection(pred_missing)))
        missed_missing_teeth = sorted(list(truth_missing.difference(pred_missing)))  # False Negatives
        wrongly_predicted_missing = sorted(list(pred_missing.difference(truth_missing))) # False Positives
        
        print(f"  Ground Truth (Missing): {sorted(list(truth_missing)) if truth_missing else 'None'}")
        print(f"  Prediction (Missing):   {sorted(list(pred_missing)) if pred_missing else 'None'}")
        print("-"*50)
        print(f"   Correctly Found (TP): {correctly_found_missing or 'None'}")
        print(f"   Missed Teeth (FN):      {missed_missing_teeth or 'None'}")
        print(f"   False Alarms (FP):      {wrongly_predicted_missing or 'None'}")
    print("\n" + "="*80)

# Rewritten to remove per-tooth plot and add jaw CM
def generate_test_plots(metrics, save_dir):
    """Generates and saves a collection of plots summarizing test performance."""
    teeth_metrics = metrics['teeth_metrics']
    jaw_metrics = metrics['jaw_metrics']
    
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- PLOT 1: Confusion Matrix (Teeth) ---
    flat_targets_teeth = teeth_metrics['flat_targets']
    flat_preds_teeth = (teeth_metrics['flat_probs'] > 0.5).astype(int)
    
    cm_teeth = confusion_matrix(flat_targets_teeth, flat_preds_teeth)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_teeth, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Present (0)', 'Predicted Missing (1)'], 
                yticklabels=['Actual Present (0)', 'Actual Missing (1)'])
    plt.title('Overall Confusion Matrix (All Teeth, Micro)', fontsize=16, fontweight='bold')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plot_path = Path(save_dir) / "confusion_matrix_teeth.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Teeth confusion matrix plot saved to {plot_path}")

    # --- PLOT 2: Confusion Matrix (Jaw) ---
    targets_jaw = jaw_metrics['targets']
    preds_jaw = (jaw_metrics['probs'] > 0.5).astype(int)

    cm_jaw = confusion_matrix(targets_jaw, preds_jaw)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_jaw, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Predicted Upper (0)', 'Predicted Lower (1)'],
                yticklabels=['Actual Upper (0)', 'Actual Lower (1)'])
    plt.title('Jaw Classification Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plot_path = Path(save_dir) / "confusion_matrix_jaw.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Jaw confusion matrix plot saved to {plot_path}")
    
    # --- PLOT 3: Precision-Recall Curve (Teeth) ---
    precision, recall, _ = precision_recall_curve(flat_targets_teeth, teeth_metrics['flat_probs'])
    avg_precision = average_precision_score(flat_targets_teeth, teeth_metrics['flat_probs'])
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (Avg Precision = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Overall Precision-Recall Curve (Teeth Only)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left"); plt.grid(True, alpha=0.5)
    plot_path = Path(save_dir) / "precision_recall_curve_teeth.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Precision-Recall curve saved to {plot_path}")

# =================================================================================
# MAIN EXECUTION
# =================================================================================


def main():
    """Orchestrates the entire testing pipeline."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with new num_outputs
    model = ResNetMultiLabel(num_outputs=NUM_OUTPUTS).to(device)
    
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
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if (hasattr(missing, 'missing_keys') and (missing.missing_keys or missing.unexpected_keys)):
            print(f"[warn] Strict load reported issues. Missing: {getattr(missing, 'missing_keys', [])}, Unexpected: {getattr(missing, 'unexpected_keys', [])}")
    except RuntimeError as e:
        print("[warn] Strict load failed, trying non-strict mapping...")
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(f"[warn] Non-strict load. Missing keys: {incompatible.missing_keys}; Unexpected keys: {incompatible.unexpected_keys}")

    print(f" Model loaded from {MODEL_PATH}")

    # load_test_labels now returns 17-element vectors
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
            labels = labels_dict[key] # This is now [17,]
            test_items.append((img_path, key, labels))
        else:
            mismatched += 1
            if len(unmatched_samples) < 5:
                unmatched_samples.append((orig_stem, case_id_norm))
    if mismatched:
        print(f" [debug] Unmatched samples (png stem -> normalized) examples: {unmatched_samples}")
    print(f" Prepared {len(test_items)} matching samples for testing. Skipped {mismatched} without labels.")
    if not test_items:
        print(" No matching samples found. Exiting."); return

    # preds and targets will be [N, 17]
    preds, targets, ids = test_model(model, test_items, device, transform)
    print(f" Inference complete on {len(ids)} samples.")

    metrics = calculate_metrics(preds, targets)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, num_samples=NUM_SAMPLE_PREDICTIONS)

    print("\n" + "="*80 + "\n" + " "*28 + "GENERATING PLOTS" + "\n" + "="*80)
    # Pass only metrics, not preds/targets
    generate_test_plots(metrics, OUTPUT_DIR)

    # Save raw 17-output results instead of 32-FDI
    results = pd.DataFrame({'case_id': ids})
    # Save jaw results
    results['true_jaw'] = targets[:, NUM_TEETH_PER_JAW].astype(int)
    results['pred_jaw'] = (preds[:, NUM_TEETH_PER_JAW] > 0.5).astype(int)
    results['prob_jaw'] = preds[:, NUM_TEETH_PER_JAW]
    
    # Save 16-teeth results (as indices)
    for i in range(NUM_TEETH_PER_JAW):
        results[f'true_tooth_{i}'] = targets[:, i].astype(int)
        results[f'pred_tooth_{i}'] = (preds[:, i] > 0.5).astype(int)
        results[f'prob_tooth_{i}'] = preds[:, i]
        
    results.to_csv(Path(OUTPUT_DIR) / 'test_predictions_detailed_17out.csv', index=False)

    with open(Path(OUTPUT_DIR) / 'test_metrics_17out.json', 'w') as f:
        # Need to convert numpy arrays in metrics to lists for JSON
        serializable_metrics = metrics.copy()
        serializable_metrics['teeth_metrics'].pop('flat_probs', None)
        serializable_metrics['teeth_metrics'].pop('flat_targets', None)
        serializable_metrics['jaw_metrics'].pop('probs', None)
        serializable_metrics['jaw_metrics'].pop('targets', None)
        json.dump(serializable_metrics, f, indent=4)

    print(f"\n Full results and metrics saved successfully to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()