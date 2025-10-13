# testing baseline model with dynamit loss
# using labels with flipped teeth labels 1 absent and 0 present


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import trimesh
import random
from collections import OrderedDict

# Plotting and metrics libraries
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for servers
import seaborn as sns

# Scikit-learn for metrics and alignment
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
from sklearn.decomposition import PCA


# =================================================================================
# CONFIGURATION
# =================================================================================
TEST_PLY_DIR = "/home/user/tbrighton/blender_outputs/parsed_ply"
TEST_LABELS_CSV = "/home/user/tbrighton/Scripts/Testing/3D/label_flipped.csv"
# --- UPDATED: Path to the new model trained with Dynamit Loss ---
MODEL_PATH = "/home/user/tbrighton/Scripts/Training/3D/trained_models/dynamit_loss_best_1.pth"
# ----------------------------------------------------------------
OUTPUT_DIR = "/home/user/tbrighton/Scripts/Testing/3D/test_results_dynamit"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


NUM_POINTS = 4096
NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 5


# --- FDI Notation Mapping (Must be identical to the training script) ---
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}



# =================================================================================
# MODEL DEFINITION (Must be identical to the training script)
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
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))); x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x))); return torch.max(x, 2)[0]

class ToothClassificationModel(nn.Module):
    def __init__(self, num_teeth=32, feature_dim=1024):
        super().__init__()
        self.encoder = PointNetEncoder(input_dim=3, feature_dim=feature_dim)
        self.fc1 = nn.Linear(feature_dim, 512); self.bn1 = nn.BatchNorm1d(512); self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256); self.bn2 = nn.BatchNorm1d(256); self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_teeth)
    def forward(self, x):
        features = self.encoder(x)
        x = self.dropout1(F.relu(self.bn1(self.fc1(features))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x)))); return self.fc3(x)


# =================================================================================
# DATA PREPROCESSING AND LOADING
# =================================================================================

def align_point_cloud_robust(points):
    if len(points) < 10: return points
    centroid = np.mean(points, axis=0); points_centered = points - centroid
    pca = PCA(n_components=3); pca.fit(points_centered)
    components = pca.components_; z_axis = components[2]
    if np.mean(((points_centered @ z_axis) ** 3)) > 0: z_axis *= -1
    y_axis = components[1]; proj_y = points_centered @ y_axis
    if np.mean(proj_y[proj_y > np.quantile(proj_y, 0.9)]) < 0: y_axis *= -1
    x_axis = np.cross(y_axis, z_axis)
    return points_centered @ np.stack([x_axis, y_axis, z_axis], axis=0).T

def load_ply_file(ply_path):
    try:
        mesh = trimesh.load(ply_path, process=False)
        return np.array(mesh.vertices, dtype=np.float32)
    except Exception as e:
        print(f"❌ Error loading {ply_path}: {e}")
        return np.array([], dtype=np.float32)

def normalize_point_cloud(points):
    if len(points) == 0: return points
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    if max_dist > 0: points /= max_dist
    return points

def sample_points(points, num_points):
    if len(points) == 0: return np.zeros((num_points, 3), dtype=np.float32)
    indices = np.random.choice(len(points), num_points, replace=len(points) < num_points)
    return points[indices]

def load_test_labels(csv_path):
    df = pd.read_csv(csv_path); df['new_id'] = df['new_id'].astype(str).str.strip()
    df.columns = [str(c) if c.isdigit() else c for c in df.columns]
    labels_dict = {}
    for _, row in df.iterrows():
        case_id = row['new_id']
        label_vector = np.zeros(NUM_TEETH, dtype=np.float32)
        for tooth_fdi in VALID_FDI_LABELS:
            if str(tooth_fdi) in df.columns and int(row[str(tooth_fdi)]) == 1:
                label_vector[FDI_TO_INDEX[tooth_fdi]] = 1.0
        labels_dict[case_id] = label_vector
    return labels_dict

# =================================================================================
# INFERENCE AND METRICS
# =================================================================================

def test_model(model, test_data, device):
    model.eval(); all_preds, all_targets, all_ids = [], [], []
    with torch.no_grad():
        for case_id, points, labels in tqdm(test_data, desc="Testing"):
            points_tensor = torch.from_numpy(points).unsqueeze(0).float().to(device)
            logits = model(points_tensor)
            all_preds.append(torch.sigmoid(logits).cpu().numpy()[0])
            all_targets.append(labels); all_ids.append(case_id)
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
    macro_p = np.mean([m['precision'] for m in per_tooth.values()]); macro_r = np.mean([m['recall'] for m in per_tooth.values()])
    macro_f1 = np.mean([m['f1'] for m in per_tooth.values()]); macro_acc = np.mean([m['accuracy'] for m in per_tooth.values()])
    return {'overall_micro': {'precision': float(p), 'recall': float(r), 'f1': float(f1), 'accuracy': float(acc)},
            'overall_macro': {'macro_precision': macro_p, 'macro_recall': macro_r, 'macro_f1': macro_f1, 'macro_accuracy': macro_acc},
            'per_tooth': per_tooth}

# =================================================================================
# REPORTING AND VISUALIZATION
# =================================================================================

def print_metrics_summary(metrics):
    print("\n" + "="*80 + "\n" + " "*25 + "TESTING METRICS SUMMARY" + "\n" + "="*80)
    micro = metrics['overall_micro']
    print("\n📊 OVERALL (MICRO-AVERAGE) METRICS (Positive class: Missing Tooth):")
    print(f"  - Precision: {micro['precision']:.4f}\n  - Recall:    {micro['recall']:.4f}\n  - F1 Score:  {micro['f1']:.4f}\n  - Accuracy:  {micro['accuracy']:.4f}")
    macro = metrics['overall_macro']
    print("\n📈 OVERALL (MACRO-AVERAGE) METRICS:")
    print(f"  - Macro Precision: {macro['macro_precision']:.4f}\n  - Macro Recall:    {macro['macro_recall']:.4f}\n  - Macro F1 Score:  {macro['macro_f1']:.4f}\n  - Macro Accuracy:  {macro['macro_accuracy']:.4f}")
    print("\n🦷 PER-TOOTH METRICS (FDI Notation):"); print("-" * 80)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}"); print("-" * 80)
    for fdi_label, m in metrics['per_tooth'].items():
        print(f"Tooth {fdi_label:<5} {m['precision']:>10.4f}   {m['recall']:>10.4f}   {m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}")
    print("=" * 80)


# --- REWRITTEN FOR CLARITY ---
def print_sample_predictions(ids, preds, targets, num_samples):
    """Prints a clear, readable comparison for random samples."""
    print("\n" + "="*80 + "\n" + " "*28 + "SAMPLE PREDICTIONS" + "\n" + "="*80)
    if len(ids) < num_samples: num_samples = len(ids)
    
    sample_indices = random.sample(range(len(ids)), num_samples)
    for i in sample_indices:
        case_id = ids[i]
        target_labels = targets[i]
        pred_labels = (preds[i] > 0.5).astype(int)
        
        # Get sets of FDI labels for missing teeth (where label is 1)
        truth_missing = {INDEX_TO_FDI[j] for j, label in enumerate(target_labels) if label == 1}
        pred_missing = {INDEX_TO_FDI[j] for j, label in enumerate(pred_labels) if label == 1}
        
        # --- Logic for True Positives, False Negatives, False Positives ---
        correctly_found_missing = sorted(list(truth_missing.intersection(pred_missing)))
        missed_missing_teeth = sorted(list(truth_missing.difference(pred_missing)))      # False Negatives
        wrongly_predicted_missing = sorted(list(pred_missing.difference(truth_missing))) # False Positives
        
        print(f"\n📁 Case ID: {case_id}\n" + "-"*50)
        print(f"  Ground Truth (Missing): {sorted(list(truth_missing)) if truth_missing else 'None'}")
        print(f"  Prediction (Missing):   {sorted(list(pred_missing)) if pred_missing else 'None'}")
        print("-"*50)
        print(f"  ✅ Correctly Found (TP): {correctly_found_missing or 'None'}")
        print(f"  ❌ Missed Teeth (FN):      {missed_missing_teeth or 'None'}")
        print(f"  ⚠️ False Alarms (FP):      {wrongly_predicted_missing or 'None'}")
    print("\n" + "="*80)



def generate_test_plots(metrics, preds, targets, save_dir):
    """Generates and saves a collection of plots summarizing test performance."""
    per_tooth_metrics = metrics['per_tooth']
    fdi_labels = [str(label) for label in per_tooth_metrics.keys()]
    f1_scores = [m['f1'] for m in per_tooth_metrics.values()]
    precision_scores = [m['precision'] for m in per_tooth_metrics.values()]
    recall_scores = [m['recall'] for m in per_tooth_metrics.values()]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(fdi_labels)); width = 0.25
    ax.bar(x - width, precision_scores, width, label='Precision', color='royalblue')
    ax.bar(x, recall_scores, width, label='Recall', color='limegreen')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='tomato')
    ax.set_ylabel('Scores', fontsize=14); ax.set_title('Per-Tooth Performance (Positive Class: Missing)', fontsize=18, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(fdi_labels, rotation=45, ha='right'); ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05); ax.bar_label(rects3, padding=3, fmt='%.2f', fontsize=8); fig.tight_layout()
    plot_path = Path(save_dir) / "per_tooth_metrics.png"; plt.savefig(plot_path, dpi=300); plt.close()
    print(f"✓ Per-tooth metrics plot saved to {plot_path}")

    # --- UPDATED CM LABELS ---
    flat_targets = targets.flatten(); flat_preds = (preds > 0.5).astype(int).flatten()
    cm = confusion_matrix(flat_targets, flat_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Present (0)', 'Predicted Missing (1)'], 
                yticklabels=['Actual Present (0)', 'Actual Missing (1)'])
    plt.title('Overall Confusion Matrix (All Teeth)', fontsize=16, fontweight='bold'); plt.ylabel('Ground Truth'); plt.xlabel('Prediction')
    plot_path = Path(save_dir) / "confusion_matrix.png"; plt.savefig(plot_path, dpi=300); plt.close()
    print(f"✓ Confusion matrix plot saved to {plot_path}")
    
    precision, recall, _ = precision_recall_curve(flat_targets, preds.flatten())
    avg_precision = average_precision_score(flat_targets, preds.flatten())
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (Avg Precision = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Overall Precision-Recall Curve', fontsize=16, fontweight='bold'); plt.legend(loc="lower left"); plt.grid(True, alpha=0.5)
    plot_path = Path(save_dir) / "precision_recall_curve.png"; plt.savefig(plot_path, dpi=300); plt.close()
    print(f"✓ Precision-Recall curve saved to {plot_path}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ToothClassificationModel(num_teeth=NUM_TEETH).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"✅ Model loaded from {MODEL_PATH}")

    labels_dict = load_test_labels(TEST_LABELS_CSV)
    print(f"✅ Loaded labels for {len(labels_dict)} cases from {TEST_LABELS_CSV}")
    
    test_data = []
    for ply_file in sorted(Path(TEST_PLY_DIR).glob("*.ply")):
        case_id = ply_file.stem
        if case_id in labels_dict:
            points = load_ply_file(ply_file)
            if len(points) > 0:
                points = align_point_cloud_robust(points)
                if "lower" in case_id.lower(): points[:, 1:] *= -1
                points = normalize_point_cloud(points)
                points = sample_points(points, NUM_POINTS)
                test_data.append((case_id, points, labels_dict[case_id]))
    
    print(f"✅ Prepared {len(test_data)} matching samples for testing.")
    if not test_data: print("❌ No matching samples found. Exiting."); return

    preds, targets, ids = test_model(model, test_data, device)
    print(f"✅ Inference complete on {len(ids)} samples.")
    
    metrics = calculate_metrics(preds, targets)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, num_samples=NUM_SAMPLE_PREDICTIONS)
    
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
        
    print(f"\n📊 Full results and metrics saved successfully to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()