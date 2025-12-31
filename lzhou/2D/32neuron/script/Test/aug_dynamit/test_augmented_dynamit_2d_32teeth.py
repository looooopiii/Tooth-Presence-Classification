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

# ==================== CONFIGURATION ====================
TEST_IMG_DIR = "/home/user/lzhou/week15/render_output/test"
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"
# Model path for Dynamit
MODEL_PATH = "/home/user/lzhou/week15-32/output/Train2D/aug_dynamit_32teeth/dynamit_loss_full_dataset_best.pth"
OUTPUT_DIR = "/home/user/lzhou/week15-32/output/Test2D/aug_dynamit_32teeth"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Hyperparameters
IMG_SIZE = 256
NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 10
DROPOUT_RATE = 0.5 

VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi: i for i, fdi in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi for fdi, i in FDI_TO_INDEX.items()}
UPPER_FDI = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

# ==================== HELPERS  ====================
def normalize_png_stem_to_newid(stem: str) -> str:
    s = stem.replace('-', '_').strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'_rot\d+$', '', s)
    if s.endswith('_top'): s = s[:-4]
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
        jaw_key = 'upper'; s = s[:-6]
    elif lower_s.endswith('_lower'):
        jaw_key = 'lower'; s = s[:-6]
    s = s.strip().replace(' ', '_').replace('-', '_')
    while '__' in s: s = s.replace('__', '_')
    s = s.strip('_')
    return f"{s}_{jaw_key}" if jaw_key else s

# ==================== DATA LOADING (WITH JAW LOGIC) ====================
def load_test_labels(csv_path):
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
                if int(row[tooth_str]) == 1:
                    label_vector[FDI_TO_INDEX[tooth_fdi]] = 1.0

        if is_upper:
            for fdi in LOWER_FDI: label_vector[FDI_TO_INDEX[fdi]] = 1.0
            jaw_type_dict[case_id] = 'upper'
        elif is_lower:
            for fdi in UPPER_FDI: label_vector[FDI_TO_INDEX[fdi]] = 1.0
            jaw_type_dict[case_id] = 'lower'
        else:
            jaw_type_dict[case_id] = 'unknown'
        
        labels_dict[case_id] = label_vector
    
    # Logging
    print(f"  [Info] Upper jaw samples: {sum(1 for v in jaw_type_dict.values() if v == 'upper')}")
    print(f"  [Info] Lower jaw samples: {sum(1 for v in jaw_type_dict.values() if v == 'lower')}")
    return labels_dict, jaw_type_dict

def find_test_images(img_dir, labels_dict):
    grouped = {}
    path = Path(img_dir)
    if not path.exists():
        print(f" Error: Directory not found: {img_dir}"); return grouped
        
    files = sorted(list(path.glob("*.png")))
    print(f" Scanning {len(files)} files in: {img_dir}")
    matched_count = 0
    
    for img_path in files:
        raw_stem = img_path.stem
        norm_id = normalize_png_stem_to_newid(raw_stem)
        final_key = None
        if norm_id in labels_dict: final_key = norm_id
        elif raw_stem.lower() in labels_dict: final_key = raw_stem.lower()
        
        if final_key:
            if final_key not in grouped: grouped[final_key] = {'paths': []}
            grouped[final_key]['paths'].append(str(img_path))
            matched_count += 1
            
    print(f" Successfully matched {len(grouped)} unique cases from {matched_count} images.")
    return grouped

# ==================== MODEL ====================
class ResNetMultiLabel(nn.Module):
    def __init__(self, backbone="resnet18", num_teeth=32, dropout_rate=0.5):
        super().__init__()
        if backbone == "resnet50": net = resnet50(weights=None)
        else: net = resnet18(weights=None)
        in_feats = net.fc.in_features; net.fc = nn.Identity(); self.backbone = net
        self.classifier = nn.Sequential(
            nn.Linear(in_feats, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(256, num_teeth)
        )
    def forward(self, x): return self.classifier(self.backbone(x))

# ==================== INFERENCE ====================
def test_model(model, grouped_imgs, labels_dict, device, transform):
    model.eval()
    all_preds, all_targets, all_ids = [], [], []
    with torch.no_grad():
        for case_id, data in tqdm(grouped_imgs.items(), desc="Testing"):
            if case_id not in labels_dict: continue
            labels = labels_dict[case_id]
            probs_list = []
            for img_path in data['paths']:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    logits = model(img_tensor)
                    probs = torch.sigmoid(logits)
                    probs_list.append(probs.cpu().numpy()[0])
                except Exception as e: print(f"Read Error: {e}")
            if not probs_list: continue
            avg_probs = np.mean(probs_list, axis=0) 
            all_preds.append(avg_probs); all_targets.append(labels); all_ids.append(case_id)
    return np.array(all_preds), np.array(all_targets), all_ids

def calculate_metrics(preds, targets, jaw_type_dict, all_ids):
    if len(preds) == 0: return {}
    preds_bin = (preds > 0.5).astype(int)
    targets_bin = targets.astype(int)
    flat_p, flat_t = preds_bin.flatten(), targets_bin.flatten()
    p, r, f1, _ = precision_recall_fscore_support(flat_t, flat_p, average='binary', zero_division=0)
    acc = accuracy_score(flat_t, flat_p)
    bal_acc = balanced_accuracy_score(flat_t, flat_p)
    
    per_tooth = OrderedDict()
    for idx in range(NUM_TEETH):
        fdi = INDEX_TO_FDI[idx]
        pt, rt, f1t, _ = precision_recall_fscore_support(targets_bin[:, idx], preds_bin[:, idx], average='binary', zero_division=0)
        acct = accuracy_score(targets_bin[:, idx], preds_bin[:, idx])
        support = int(targets_bin[:, idx].sum())
        per_tooth[fdi] = {'precision': float(pt), 'recall': float(rt), 'f1': float(f1t), 'accuracy': float(acct), 'support': support}
    
    macro_f1 = np.mean([m['f1'] for m in per_tooth.values()])
    macro_precision = np.mean([m['precision'] for m in per_tooth.values()])
    macro_recall = np.mean([m['recall'] for m in per_tooth.values()])
    
    u_corr, u_tot = 0, 0
    l_corr, l_tot = 0, 0
    for i, cid in enumerate(all_ids):
        jtype = jaw_type_dict.get(cid, 'unknown')
        if jtype == 'upper':
            for fdi in UPPER_FDI:
                idx = FDI_TO_INDEX[fdi]; u_tot += 1
                if preds_bin[i][idx] == targets_bin[i][idx]: u_corr += 1
        elif jtype == 'lower':
            for fdi in LOWER_FDI:
                idx = FDI_TO_INDEX[fdi]; l_tot += 1
                if preds_bin[i][idx] == targets_bin[i][idx]: l_corr += 1
    
    return {
        'overall_micro': {'precision': float(p), 'recall': float(r), 'f1': float(f1), 'accuracy': float(acc), 'balanced_accuracy': float(bal_acc)},
        'overall_macro': {'macro_precision': float(macro_precision), 'macro_recall': float(macro_recall), 'macro_f1': float(macro_f1)},
        'per_jaw': {
            'upper_jaw_accuracy': u_corr/u_tot if u_tot else 0, 'upper_jaw_samples': u_tot//16,
            'lower_jaw_accuracy': l_corr/l_tot if l_tot else 0, 'lower_jaw_samples': l_tot//16
        },
        'per_tooth': per_tooth
    }

# ==================== DISPLAY & PLOTTING ====================
def print_metrics_summary(metrics):
    micro = metrics['overall_micro']
    macro = metrics['overall_macro']
    per_jaw = metrics['per_jaw']
    
    print("\n" + "="*80)
    print(" "*20 + "TESTING METRICS SUMMARY (Aug Dynamit Loss)")
    print("="*80)
    print(f"\n OVERALL (MICRO):")
    print(f"   Precision: {micro['precision']:.4f}")
    print(f"   Recall:    {micro['recall']:.4f}")
    print(f"   F1:        {micro['f1']:.4f}")
    print(f"   Accuracy:  {micro['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {micro['balanced_accuracy']:.4f}")
    print(f"\n OVERALL (MACRO):")
    print(f"   Macro Precision: {macro['macro_precision']:.4f}")
    print(f"   Macro Recall:    {macro['macro_recall']:.4f}")
    print(f"   Macro F1:        {macro['macro_f1']:.4f}")
    print(f"\n PER-JAW ACCURACY (only look at the 16 teeth corresponding to the jaw):")
    print(f"   Upper Jaw: {per_jaw['upper_jaw_accuracy']:.4f} ({per_jaw['upper_jaw_samples']} samples)")
    print(f"   Lower Jaw: {per_jaw['lower_jaw_accuracy']:.4f} ({per_jaw['lower_jaw_samples']} samples)")
    print("\n" + "-" * 80)
    print(" PER-TOOTH METRICS:")
    print("-" * 80)
    print(f"{'FDI':<8} {'Prec':<10} {'Rec':<10} {'F1':<10} {'Acc':<10} {'Supp':<8}")
    print("-" * 80)
    print("--- UPPER JAW (11-28) ---")
    for fdi in UPPER_FDI:
        m = metrics['per_tooth'][fdi]
        print(f"{fdi:<8} {m['precision']:>8.4f}   {m['recall']:>8.4f}   {m['f1']:>8.4f}   {m['accuracy']:>8.4f}   {m['support']:>6}")
    print("--- LOWER JAW (31-48) ---")
    for fdi in LOWER_FDI:
        m = metrics['per_tooth'][fdi]
        print(f"{fdi:<8} {m['precision']:>8.4f}   {m['recall']:>8.4f}   {m['f1']:>8.4f}   {m['accuracy']:>8.4f}   {m['support']:>6}")
    print("=" * 80)

def print_sample_predictions(ids, preds, targets, jaw_type_dict, num_samples):
    print("\n" + "="*80)
    print(" "*28 + "SAMPLE PREDICTIONS")
    print("="*80)
    if len(ids) == 0: return
    indices = random.sample(range(len(ids)), min(len(ids), num_samples))
    for i in indices:
        case_id = ids[i]
        jaw_type = jaw_type_dict.get(case_id, 'unknown')
        target_vec = targets[i]; pred_vec = (preds[i] > 0.5).astype(int)
        relevant_fdi = UPPER_FDI if jaw_type == 'upper' else (LOWER_FDI if jaw_type == 'lower' else VALID_FDI_LABELS)
        
        truth_missing = [INDEX_TO_FDI[j] for j, val in enumerate(target_vec) if val == 1 and INDEX_TO_FDI[j] in relevant_fdi]
        pred_missing = [INDEX_TO_FDI[j] for j, val in enumerate(pred_vec) if val == 1 and INDEX_TO_FDI[j] in relevant_fdi]
        
        truth_set = set(truth_missing); pred_set = set(pred_missing)
        tp = sorted(list(truth_set.intersection(pred_set)))
        fn = sorted(list(truth_set.difference(pred_set)))
        fp = sorted(list(pred_set.difference(truth_set)))
        def fmt_list(lst): return str(sorted(lst)) if lst else "None"
        
        print(f"\n Case ID: {case_id}")
        print(f"   Jaw Type: {jaw_type.upper()}")
        print("-" * 60)
        print(f"  Ground Truth (Missing in {jaw_type}): {fmt_list(truth_missing)}")
        print(f"  Prediction (Missing in {jaw_type}):   {fmt_list(pred_missing)}")
        print("-" * 60)
        print(f"   Correctly Found (TP): {fmt_list(tp)}")
        print(f"   Missed Teeth (FN):    {fmt_list(fn)}")
        print(f"   False Alarms (FP):    {fmt_list(fp)}")
        correct = len(tp) + (len(relevant_fdi) - len(truth_set) - len(fp))
        case_acc = correct / len(relevant_fdi) if relevant_fdi else 0
        print(f"   Case Accuracy: {case_acc:.2%} ({correct}/{len(relevant_fdi)})")
    print("\n" + "="*80)

def generate_plots(metrics, preds, targets, save_dir):
    per_tooth = metrics['per_tooth']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    upper_f1s = [per_tooth[fdi]['f1'] for fdi in UPPER_FDI]
    axes[0].bar(range(len(UPPER_FDI)), upper_f1s, color='steelblue')
    axes[0].set_xticks(range(len(UPPER_FDI))); axes[0].set_xticklabels([str(f) for f in UPPER_FDI], rotation=45)
    axes[0].set_title('F1 Score - Upper Jaw (11-28) - Aug Dynamit')
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=np.mean(upper_f1s), color='red', linestyle='--', label=f'Mean: {np.mean(upper_f1s):.3f}')
    axes[0].legend()
    
    lower_f1s = [per_tooth[fdi]['f1'] for fdi in LOWER_FDI]
    axes[1].bar(range(len(LOWER_FDI)), lower_f1s, color='coral')
    axes[1].set_xticks(range(len(LOWER_FDI))); axes[1].set_xticklabels([str(f) for f in LOWER_FDI], rotation=45)
    axes[1].set_title('F1 Score - Lower Jaw (31-48) - Aug Dynamit')
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=np.mean(lower_f1s), color='red', linestyle='--', label=f'Mean: {np.mean(lower_f1s):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "f1_score_per_jaw_aug_dynamit.png", dpi=150)
    plt.close()
    
    cm = confusion_matrix(targets.flatten(), (preds > 0.5).astype(int).flatten())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Present (0)', 'Missing (1)'], yticklabels=['Present (0)', 'Missing (1)'])
    plt.title('Overall Confusion Matrix - Aug Dynamit'); plt.xlabel('Predicted'); plt.ylabel('Ground Truth')
    plt.savefig(Path(save_dir) / "confusion_matrix_aug_dynamit.png", dpi=150)
    plt.close()
    print(f" Plots saved to {save_dir}")

# ==================== MAIN ====================
def main():
    print("\n" + "="*80)
    print(" "*10 + "2D MODEL TESTING (32 Teeth - Aug Dynamit Loss)")
    print("="*80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/5] Using device: {device}")

    print(f"\n[2/5] Loading model from: {MODEL_PATH}")
    model = ResNetMultiLabel(backbone="resnet18", num_teeth=NUM_TEETH, dropout_rate=DROPOUT_RATE).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if list(state_dict.keys())[0].startswith('module.'): state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"   Model loaded successfully")
    except Exception as e: print(f"   Error loading model: {e}"); return

    print(f"\n[3/5] Loading test labels from: {TEST_LABELS_CSV}")
    labels_dict, jaw_type_dict = load_test_labels(TEST_LABELS_CSV)
    print(f"   Loaded labels for {len(labels_dict)} cases")
    
    print(f"\n[4/5] Finding test images in: {TEST_IMG_DIR}")
    grouped_imgs = find_test_images(TEST_IMG_DIR, labels_dict)
    if len(grouped_imgs) == 0: return

    print(f"\n[5/5] Running inference...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    preds, targets, ids = test_model(model, grouped_imgs, labels_dict, device, transform)
    print(f"   Tested on {len(ids)} cases")
    
    metrics = calculate_metrics(preds, targets, jaw_type_dict, ids)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, jaw_type_dict, num_samples=NUM_SAMPLE_PREDICTIONS)
    generate_plots(metrics, preds, targets, OUTPUT_DIR)
    
    metrics_file = Path(OUTPUT_DIR) / "test_metrics_aug_dynamit.json"
    with open(metrics_file, 'w') as f:
        metrics_save = metrics.copy()
        metrics_save['per_tooth'] = {str(k): v for k, v in metrics['per_tooth'].items()}
        json.dump(metrics_save, f, indent=2)
    print(f"\n Metrics saved to {metrics_file}")
    
    print("\n" + "="*80)
    print(" "*30 + "DONE!")
    print("="*80)

if __name__ == "__main__":
    main()