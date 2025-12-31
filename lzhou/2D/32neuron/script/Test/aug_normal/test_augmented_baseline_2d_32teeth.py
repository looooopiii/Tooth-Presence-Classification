import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import random
import re
from collections import OrderedDict

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
from torchvision import transforms
from torchvision.models import resnet18, resnet50


# =================================================================================
# CONFIGURATION
# =================================================================================
TEST_IMG_DIR = "/home/user/lzhou/week15/render_output/test"
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"
MODEL_PATH = "/home/user/lzhou/week15-32/output/Train2D/aug_dynamit_32teeth/dynamit_loss_full_dataset_best.pth"
OUTPUT_DIR = "/home/user/lzhou/week15-32/output/Test2D/aug_dynamit_32teeth"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 5
DROPOUT_RATE = 0.5

# FDI Notation
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}


# =================================================================================
# ID NORMALIZATION
# =================================================================================
def normalize_png_stem_to_newid(stem: str) -> str:
    """Normalize PNG filename stem to match CSV new_id format"""
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
# MODEL
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
    """Load test labels from CSV - following teammate's structure"""
    df = pd.read_csv(csv_path)
    df['new_id'] = df['new_id'].astype(str).str.strip().str.lower().str.replace('-', '_')
    df.columns = [str(c) for c in df.columns]
    
    labels_dict = {}
    
    for _, row in df.iterrows():
        case_id = row['new_id']
        label_vector = np.zeros(NUM_TEETH, dtype=np.float32)
        
        for tooth_fdi in VALID_FDI_LABELS:
            tooth_str = str(tooth_fdi)
            if tooth_str in df.columns and pd.notna(row[tooth_str]):
                if int(row[tooth_str]) == 1:
                    label_vector[FDI_TO_INDEX[tooth_fdi]] = 1.0
        
        labels_dict[case_id] = label_vector
    
    return labels_dict


def find_test_images(img_dir, labels_dict):
    """Find and group test images by case ID"""
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
# INFERENCE AND METRICS
# =================================================================================
def test_model(model, grouped_imgs, labels_dict, device, transform):
    """Run inference on test data - following teammate's structure"""
    model.eval()
    all_preds, all_targets, all_ids = [], [], []
    
    with torch.no_grad():
        for case_id, data in tqdm(grouped_imgs.items(), desc="Testing"):
            if case_id not in labels_dict:
                continue
            
            labels = labels_dict[case_id]
            
            # Multi-view averaging (if multiple rotations exist)
            probs_list = []
            for img_path in data['paths']:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    logits = model(img_tensor)
                    probs = torch.sigmoid(logits)
                    probs_list.append(probs.cpu().numpy()[0])
                except Exception as e:
                    print(f" Error reading {img_path}: {e}")
            
            if not probs_list:
                continue

            avg_probs = np.mean(probs_list, axis=0)
            
            all_preds.append(avg_probs)
            all_targets.append(labels)
            all_ids.append(case_id)
    
    return np.array(all_preds), np.array(all_targets), all_ids


def calculate_metrics(preds, targets):
    """Calculate all metrics - following teammate's structure"""
    preds_bin = (preds > 0.5).astype(int)
    targets_bin = targets.astype(int)
    
    flat_preds = preds_bin.flatten()
    flat_targets = targets_bin.flatten()
    
    p, r, f1, _ = precision_recall_fscore_support(flat_targets, flat_preds, average='binary', zero_division=0)
    acc = accuracy_score(flat_targets, flat_preds)
    
    # Per-tooth metrics
    per_tooth = OrderedDict()
    for idx in range(NUM_TEETH):
        fdi_label = INDEX_TO_FDI[idx]
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(
            targets_bin[:, idx], preds_bin[:, idx], average='binary', zero_division=0
        )
        per_tooth[fdi_label] = {
            'precision': float(p_t),
            'recall': float(r_t),
            'f1': float(f1_t),
            'accuracy': accuracy_score(targets_bin[:, idx], preds_bin[:, idx]),
            'support': int(targets_bin[:, idx].sum())
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
            'accuracy': float(acc)
        },
        'overall_macro': {
            'macro_precision': macro_p,
            'macro_recall': macro_r,
            'macro_f1': macro_f1,
            'macro_accuracy': macro_acc
        },
        'per_tooth': per_tooth
    }


# =================================================================================
# REPORTING AND VISUALIZATION
# =================================================================================
def print_metrics_summary(metrics):
    """Print metrics summary - following teammate's style"""
    print("\n" + "=" * 80)
    print(" " * 25 + "TESTING METRICS SUMMARY")
    print("=" * 80)
    
    micro = metrics['overall_micro']
    print("\n OVERALL (MICRO-AVERAGE) METRICS (Positive class: Missing Tooth):")
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
    print("-" * 80)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    
    for fdi_label, m in metrics['per_tooth'].items():
        print(f"Tooth {fdi_label:<5} {m['precision']:>10.4f}   {m['recall']:>10.4f}   "
              f"{m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}")
    
    print("=" * 80)


def print_sample_predictions(ids, preds, targets, num_samples):
    """Print sample predictions - following teammate's style"""
    print("\n" + "=" * 80)
    print(" " * 28 + "SAMPLE PREDICTIONS")
    print("=" * 80)
    
    if len(ids) < num_samples:
        num_samples = len(ids)
    
    sample_indices = random.sample(range(len(ids)), num_samples)
    
    for i in sample_indices:
        case_id = ids[i]
        target_labels = targets[i]
        pred_labels = (preds[i] > 0.5).astype(int)
        
        # Get sets of FDI labels for missing teeth (where label is 1)
        truth_missing = {INDEX_TO_FDI[j] for j, label in enumerate(target_labels) if label == 1}
        pred_missing = {INDEX_TO_FDI[j] for j, label in enumerate(pred_labels) if label == 1}
        
        # TP, FN, FP
        correctly_found_missing = sorted(list(truth_missing.intersection(pred_missing)))
        missed_missing_teeth = sorted(list(truth_missing.difference(pred_missing)))
        wrongly_predicted_missing = sorted(list(pred_missing.difference(truth_missing)))
        
        print(f"\n Case ID: {case_id}")
        print("-" * 50)
        print(f"  Ground Truth (Missing): {sorted(list(truth_missing)) if truth_missing else 'None'}")
        print(f"  Prediction (Missing):   {sorted(list(pred_missing)) if pred_missing else 'None'}")
        print("-" * 50)
        print(f"   Correctly Found (TP): {correctly_found_missing or 'None'}")
        print(f"   Missed Teeth (FN):    {missed_missing_teeth or 'None'}")
        print(f"   False Alarms (FP):    {wrongly_predicted_missing or 'None'}")
    
    print("\n" + "=" * 80)


def generate_test_plots(metrics, preds, targets, save_dir):
    """Generate test plots - following teammate's style"""
    per_tooth_metrics = metrics['per_tooth']
    fdi_labels = [str(label) for label in per_tooth_metrics.keys()]
    f1_scores = [m['f1'] for m in per_tooth_metrics.values()]
    precision_scores = [m['precision'] for m in per_tooth_metrics.values()]
    recall_scores = [m['recall'] for m in per_tooth_metrics.values()]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(fdi_labels))
    width = 0.25
    
    ax.bar(x - width, precision_scores, width, label='Precision', color='royalblue')
    ax.bar(x, recall_scores, width, label='Recall', color='limegreen')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='tomato')
    
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title('Per-Tooth Performance (Positive Class: Missing) - Dynamit Loss', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(fdi_labels, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.bar_label(rects3, padding=3, fmt='%.2f', fontsize=8)
    fig.tight_layout()
    
    plot_path = Path(save_dir) / "per_tooth_metrics_dynamit.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Per-tooth metrics plot saved to {plot_path}")

    # Confusion Matrix
    flat_targets = targets.flatten()
    flat_preds = (preds > 0.5).astype(int).flatten()
    cm = confusion_matrix(flat_targets, flat_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Present (0)', 'Predicted Missing (1)'],
                yticklabels=['Actual Present (0)', 'Actual Missing (1)'])
    plt.title('Overall Confusion Matrix (All Teeth) - Dynamit Loss', fontsize=16, fontweight='bold')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    
    plot_path = Path(save_dir) / "confusion_matrix_dynamit.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Confusion matrix plot saved to {plot_path}")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(flat_targets, preds.flatten())
    avg_precision = average_precision_score(flat_targets, preds.flatten())
    
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR Curve (Avg Precision = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Overall Precision-Recall Curve - Dynamit Loss', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.5)
    
    plot_path = Path(save_dir) / "precision_recall_curve_dynamit.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Precision-Recall curve saved to {plot_path}")


# =================================================================================
# MAIN
# =================================================================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ResNetMultiLabel(backbone="resnet18", num_teeth=NUM_TEETH, dropout_rate=DROPOUT_RATE).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f" Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f" Error loading model: {e}")
        return

    # Load labels
    labels_dict = load_test_labels(TEST_LABELS_CSV)
    print(f" Loaded labels for {len(labels_dict)} cases from {TEST_LABELS_CSV}")
    
    # Find test images
    grouped_imgs = find_test_images(TEST_IMG_DIR, labels_dict)
    
    if not grouped_imgs:
        print(" No matching samples found. Exiting.")
        return
    
    # Run inference
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    preds, targets, ids = test_model(model, grouped_imgs, labels_dict, device, transform)
    print(f" Inference complete on {len(ids)} samples.")
    
    # Calculate and print metrics
    metrics = calculate_metrics(preds, targets)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, num_samples=NUM_SAMPLE_PREDICTIONS)
    
    # Generate plots
    print("\n" + "=" * 80)
    print(" " * 28 + "GENERATING PLOTS")
    print("=" * 80)
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