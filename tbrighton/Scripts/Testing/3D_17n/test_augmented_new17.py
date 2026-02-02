import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import json
from collections import OrderedDict

matplotlib.use('Agg')

# ============= CONFIGURATION =============
# Input Data
TEST_PLY_DIR = "/home/user/tbrighton/blender_outputs/parsed_ply"
TEST_LABELS_CSV = "/home/user/tbrighton/Scripts/Testing/3D_17n/label_flipped_filtered.csv"

# Model Path (DynamIT Augmented)
MODEL_PATH = "/home/user/tbrighton/Scripts/Training/3D_17n/trained_models_dynamit_augmented/dynamit_aug_model_best.pth"

# Output Directory
OUTPUT_DIR = "/home/user/tbrighton/Scripts/Testing/3D_17n/dynamit_aug_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Rotation
BEST_ROT = (-90, 0, 180)

# Hyperparameters
NUM_POINTS = 4096
DEVICE = torch.device("cuda:1")
THRESHOLD = 0.5 

# FDI Mappings
UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
ALL_FDI = sorted(UPPER_FDI + LOWER_FDI)

UPPER_TO_IDX = {fdi: i for i, fdi in enumerate(UPPER_FDI)}
LOWER_TO_IDX = {fdi: i for i, fdi in enumerate(LOWER_FDI)}

# =================================================================================
# MODEL ARCHITECTURE (Must match Training)
# =================================================================================

class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, feature_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1); self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1); self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 1); self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, feature_dim, 1); self.bn5 = nn.BatchNorm1d(feature_dim)
    def forward(self, x):
        x = x.transpose(2, 1) 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return torch.max(x, 2)[0]

class ToothClassificationModel(nn.Module):
    def __init__(self, output_dim=17, feature_dim=1024):
        super().__init__()
        self.encoder = PointNetEncoder(input_dim=3, feature_dim=feature_dim)
        self.fc_shared = nn.Linear(feature_dim, 512); self.bn_shared = nn.BatchNorm1d(512); self.drop = nn.Dropout(0.3)
        self.fc_teeth = nn.Linear(512, 256); self.bn_teeth = nn.BatchNorm1d(256); self.out_teeth = nn.Linear(256, 16)
        self.fc_jaw = nn.Linear(512, 128); self.bn_jaw = nn.BatchNorm1d(128); self.out_jaw = nn.Linear(128, 1)

    def forward(self, x):
        features = self.encoder(x)
        shared = self.drop(F.relu(self.bn_shared(self.fc_shared(features))))
        teeth = self.out_teeth(self.drop(F.relu(self.bn_teeth(self.fc_teeth(shared)))))
        jaw = self.out_jaw(self.drop(F.relu(self.bn_jaw(self.fc_jaw(shared)))))
        return torch.cat([teeth, jaw], dim=1)

# =================================================================================
# UTILITIES
# =================================================================================

def load_ply_file(ply_path):
    try:
        mesh = trimesh.load(ply_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0: return np.array([], dtype=np.float32)
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        return np.array(mesh.vertices, dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)

def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
    return points_centered / max_dist if max_dist > 0 else points_centered

def sample_points(points, num_points=4096):
    if len(points) == 0: return np.zeros((num_points, 3), dtype=np.float32)
    replace_flag = len(points) < num_points
    indices = np.random.choice(len(points), num_points, replace=replace_flag)
    return points[indices]

def apply_fixed_rotation(points, angles_deg):
    """Apply the specific single rotation requested"""
    if np.allclose(angles_deg, [0, 0, 0]): return points
    rotation = R.from_euler('xyz', angles_deg, degrees=True)
    return rotation.apply(points).astype(np.float32)

# =================================================================================
# PLOTTING & HELPERS
# =================================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def plot_per_tooth_metrics(metrics_dict, save_path):
    fdi_labels, f1_scores = [], []
    sorted_keys = sorted(metrics_dict.keys(), key=lambda x: int(x))
    
    for k in sorted_keys:
        val = metrics_dict[k]['f1']
        fdi_labels.append(k)
        f1_scores.append(val if val != "N/A" else 0.0)
        
    plt.figure(figsize=(14, 6))
    colors = ['#1f77b4' if int(x) < 30 else '#ff7f0e' for x in fdi_labels]
    bars = plt.bar(fdi_labels, f1_scores, color=colors, alpha=0.8)
    
    valid_scores = [s for s in f1_scores if s > 0]
    mean_val = np.mean(valid_scores) if valid_scores else 0.0
    
    plt.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean Valid F1: {mean_val:.3f}')
    plt.title(f'Augmented DynamIT Test: Per-Tooth F1 Score', fontsize=14)
    plt.xlabel('FDI Tooth Number', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(save_path) / "dynamit_aug_per_tooth_f1.png", dpi=300)
    plt.close()

def plot_confusion_matrices(y_true, y_pred, save_path, title_suffix="Teeth"):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(6, 5))
    labels = ['Upper', 'Lower'] if title_suffix == "Jaw" else ['Present', 'Missing']
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues', vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
    plt.title(f'Normalized Confusion Matrix ({title_suffix})', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(Path(save_path) / f"dynamit_aug_confusion_matrix_{title_suffix.lower()}.png", dpi=300)
    plt.close()

# =================================================================================
# METRICS REPORTING
# =================================================================================

def calculate_and_print_metrics(all_data):
    print("\n" + "="*90)
    print("PER-TOOTH METRICS (JAW-AWARE, Support > 0)")
    print("-" * 90)
    print(f"{'FDI':<6} {'Prec':<10} {'Recall':<10} {'F1':<10} {'Acc':<10} {'Supp':<6} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5}")
    print("-" * 90)

    per_tooth_dict = OrderedDict()
    valid_precs, valid_recs, valid_f1s, valid_accs = [], [], [], []
    valid_targets_flat, valid_preds_flat = [], []

    for section_name, fdi_list, jaw_label in [("UPPER", UPPER_FDI, 0.0), ("LOWER", LOWER_FDI, 1.0)]:
        print(f"\n{section_name} JAW:")
        for fdi in fdi_list:
            is_upper = (jaw_label == 0.0)
            local_idx = UPPER_TO_IDX[fdi] if is_upper else LOWER_TO_IDX[fdi]
            
            tooth_true, tooth_pred = [], []
            for case in all_data:
                # Group by GROUND TRUTH JAW
                if case['jaw_true'] == jaw_label:
                    truth = case['teeth_true'][local_idx]
                    pred = case['teeth_pred'][local_idx]
                    if truth != -1: 
                        tooth_true.append(truth)
                        tooth_pred.append(pred)

            if not tooth_true: continue

            support = int(sum(tooth_true))
            acc = accuracy_score(tooth_true, tooth_pred)
            tn, fp, fn, tp = confusion_matrix(tooth_true, tooth_pred, labels=[0, 1]).ravel()
            
            valid_targets_flat.extend(tooth_true)
            valid_preds_flat.extend(tooth_pred)
            
            stats = {
                'precision': 'N/A', 'recall': 'N/A', 'f1': 'N/A', 
                'accuracy': float(acc), 'support': int(support), 
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
            }

            if support > 0:
                p, r, f, _ = precision_recall_fscore_support(tooth_true, tooth_pred, average='binary', zero_division=0)
                valid_precs.append(p); valid_recs.append(r); valid_f1s.append(f); valid_accs.append(acc)
                p_s, r_s, f_s = f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"
                stats.update({'precision': float(p), 'recall': float(r), 'f1': float(f)})
            else:
                p_s, r_s, f_s = "N/A", "N/A", "N/A"
                
            per_tooth_dict[str(fdi)] = stats
            print(f"{fdi:<6} {p_s:<10} {r_s:<10} {f_s:<10} {acc:<10.4f} {support:<6} {tp:<5} {fp:<5} {fn:<5} {tn:<5}")

    m_prec = np.mean(valid_precs) if valid_precs else 0.0
    m_rec = np.mean(valid_recs) if valid_recs else 0.0
    m_f1 = np.mean(valid_f1s) if valid_f1s else 0.0
    m_acc = np.mean(valid_accs) if valid_accs else 0.0
    
    # --- TOOTH DETECTION METRICS (Balanced Accuracy) ---
    tooth_bal_acc = balanced_accuracy_score(valid_targets_flat, valid_preds_flat)

    # --- JAW METRICS ---
    jaw_true_list = [c['jaw_true'] for c in all_data]
    jaw_pred_list = [c['jaw_pred'] for c in all_data]
    jaw_acc = accuracy_score(jaw_true_list, jaw_pred_list)

    print("\n" + "="*90)
    print(" AUGMENTED DYNAMIT OVERALL SUMMARY")
    print("=" * 90)
    print(f"Overall Macro F1:              {m_f1:.4f}")
    print(f"Macro Precision:               {m_prec:.4f}")
    print(f"Macro Recall:                  {m_rec:.4f}")
    print(f"Macro Accuracy:                {m_acc:.4f}")
    print(f"Tooth Balanced Accuracy:       {tooth_bal_acc:.4f}")
    print("-" * 90)
    print(f"Jaw Classification Accuracy:   {jaw_acc:.4f}")
    print("=" * 90 + "\n")
    
    with open(Path(OUTPUT_DIR) / "dynamit_aug_metrics.json", 'w') as f:
        json.dump({
            'overall_f1': m_f1, 
            'jaw_accuracy': jaw_acc, 
            'tooth_balanced_accuracy': tooth_bal_acc,
            'per_tooth_metrics': per_tooth_dict
        }, f, indent=2, cls=NumpyEncoder)
        
    return per_tooth_dict, valid_targets_flat, valid_preds_flat, jaw_true_list, jaw_pred_list

# =================================================================================
# MAIN
# =================================================================================

def main():
    print("="*90)
    print(f" TESTING AUGMENTED DYNAMIT MODEL ")
    print("="*90)
    print(f"Device: {DEVICE}")
    print(f"Model Path: {MODEL_PATH}")
    
    model = ToothClassificationModel(output_dim=17).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    print("\n[1/3] Filtering Test Cases (Match Disk Files ∩ CSV)...")
    ply_dir = Path(TEST_PLY_DIR)
    files_on_disk = {f.stem: f for f in ply_dir.glob('*.ply')}
    
    df = pd.read_csv(TEST_LABELS_CSV)
    df['new_id'] = df['new_id'].astype(str).str.strip()
    
    if 'jaw' not in df.columns:
        raise ValueError("CRITICAL: 'jaw' column missing from CSV!")
        
    df = df.drop_duplicates(subset=['new_id'], keep='first')
    df_indexed = df.set_index('new_id')
    csv_ids = set(df_indexed.index)
    
    valid_case_ids = sorted(list(set(files_on_disk.keys()) & csv_ids))
    print(f"✓ Found {len(valid_case_ids)} valid cases")
    
    if len(valid_case_ids) == 0: return

    print(f"\n[2/3] Running Inference...")
    all_results = []
    
    for case_id in tqdm(valid_case_ids, ncols=80, desc="Testing"):
        ply_path = files_on_disk[case_id]
        row = df_indexed.loc[case_id]
        
        # Ground Truth Jaw
        try:
            expected_jaw_label = int(row['jaw']) # 0 or 1
        except ValueError:
            continue
            
        points = load_ply_file(ply_path)
        if len(points) < 100: continue
        
        # --- FIXED ROTATION STEP ---
        # 1. Rotate FIRST
        points_rotated = apply_fixed_rotation(points, BEST_ROT)
        # 2. Normalize AFTER rotation
        points_norm = normalize_point_cloud(points_rotated)
        # 3. Sample
        points_sampled = sample_points(points_norm, NUM_POINTS)
        
        points_tensor = torch.from_numpy(points_sampled).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            logits = model(points_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        pred_jaw = 1 if probs[16] > THRESHOLD else 0
        pred_teeth_binary = (probs[:16] > THRESHOLD).astype(int)
        
        gt_teeth_array = np.full(16, -1)
        current_mapping = UPPER_FDI if expected_jaw_label == 0 else LOWER_FDI
        for idx, fdi in enumerate(current_mapping):
            col = str(fdi)
            if col in row and not pd.isna(row[col]):
                gt_teeth_array[idx] = int(float(row[col]))
                
        all_results.append({
            'case_id': case_id,
            'jaw_true': expected_jaw_label,
            'jaw_pred': pred_jaw,
            'teeth_true': gt_teeth_array,
            'teeth_pred': pred_teeth_binary
        })

    print("\n[3/3] Calculating metrics...")
    metrics_dict, y_true_teeth, y_pred_teeth, y_true_jaw, y_pred_jaw = calculate_and_print_metrics(all_results)
    
    plot_per_tooth_metrics(metrics_dict, OUTPUT_DIR)
    plot_confusion_matrices(y_true_teeth, y_pred_teeth, OUTPUT_DIR, "Teeth")
    plot_confusion_matrices(y_true_jaw, y_pred_jaw, OUTPUT_DIR, "Jaw")
    print(f"\n✓ All results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()