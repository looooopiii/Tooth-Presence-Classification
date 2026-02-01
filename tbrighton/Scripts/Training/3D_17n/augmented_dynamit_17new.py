import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import OrderedDict

matplotlib.use('Agg')

# ============= CONFIGURATION =============

# 1. Original Data Paths
ORIGINAL_DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]

# 2. Augmented Data Sources
AUGMENTED_DATA_SOURCES = [
    ("/home/user/tbrighton/blender_outputs/augment_test_fixed/train_labels_augmented.csv", 
     "/home/user/tbrighton/blender_outputs/augment_test_fixed"),
    ("/home/user/tbrighton/blender_outputs/augment_random_fixed/train_labels_random.csv", 
     "/home/user/tbrighton/blender_outputs/augment_random_fixed")
]

# Output Directories
OUTPUT_DIR = "/home/user/tbrighton/Scripts/Training/3D_17n/trained_models_dynamit_augmented"
PLOT_DIR = "/home/user/tbrighton/Scripts/Training/3D_17n/plots_dynamit_augmented"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

# Filenames
BEST_MODEL_FILENAME = "dynamit_aug_model_best.pth"
LAST_MODEL_FILENAME = "dynamit_aug_model_last.pth"
METRICS_FILENAME = "dynamit_aug_metrics.json"

# Hyperparameters
NUM_POINTS = 4096
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 40

# Early Stopping
EARLY_STOP_PATIENCE = 30
EARLY_STOP_MIN_DELTA = 0.0001

# Model Config
NUM_TEETH_PER_JAW = 16
TOTAL_OUTPUTS = 17

# FDI Mappings
UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
ALL_FDI = sorted(UPPER_FDI + LOWER_FDI)

UPPER_TO_IDX = {fdi: i for i, fdi in enumerate(UPPER_FDI)}
LOWER_TO_IDX = {fdi: i for i, fdi in enumerate(LOWER_FDI)}

# =========================================
#  PLOTTING FUNCTIONS
# =========================================

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
    
    plt.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label='Mean Valid F1')
    plt.title('Per-Tooth F1 Score (DynamIT + Augmentation)', fontsize=14)
    plt.xlabel('FDI Tooth Number', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(['Mean F1', 'Upper Jaw', 'Lower Jaw'])
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path / "dynamit_per_tooth_f1.png", dpi=300)
    plt.close()

def plot_confusion_matrices(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues', vmin=0, vmax=1,
                xticklabels=['Present', 'Missing'], yticklabels=['Present', 'Missing'])
    plt.title('Normalized Confusion Matrix (Teeth)', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path / "dynamit_confusion_matrix_teeth.png", dpi=300)
    plt.close()

# =========================================
#  LOSS FUNCTION (DYNAMIT)
# =========================================

class Dynamit_Loss(nn.Module):
    def __init__(self, device):
        super(Dynamit_Loss, self).__init__()
        self.device = device

    def forward(self, predictions, targets):
        teeth_targets = targets[:, :16]
        S_pos = (teeth_targets == 1).sum().float() 
        S_neg = (teeth_targets == 0).sum().float() 

        if S_pos > 0 and S_neg > 0:
            pos_coeff_val = min(1.0, (S_neg / S_pos).item())
            neg_coeff_val = min(1.0, (S_pos / S_neg).item())
        elif S_pos == 0:
            pos_coeff_val = 0.1; neg_coeff_val = 1.0
        elif S_neg == 0:
            pos_coeff_val = 1.0; neg_coeff_val = 0.1
        else:
            pos_coeff_val = 1.0; neg_coeff_val = 1.0

        pos_coeff = torch.tensor(pos_coeff_val, device=self.device)
        neg_coeff = torch.tensor(neg_coeff_val, device=self.device)

        weights = torch.where(targets == 1, pos_coeff, neg_coeff)
        weights[:, 16] = 1.0
        
        return F.binary_cross_entropy_with_logits(predictions, targets, weight=weights)

# =========================================
#  UTILS & OPTIMIZED DATASET (RAM CACHED)
# =========================================

class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0001):
        self.patience = patience; self.min_delta = min_delta; self.counter = 0; self.best_score = None
    
    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric; return False
        if val_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: return True
        else:
            self.best_score = val_metric; self.counter = 0
        return False

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class CombinedToothDataset(Dataset):
    def __init__(self, original_paths, augmented_sources, num_points=NUM_POINTS):
        self.num_points = num_points
        self.data_cache = []  # RAM Cache List
        
        print("\n[Init] Starting RAM Caching (This takes time once, but speeds up training)...")
        
        # ===== LOAD ORIGINAL DATA INTO RAM =====
        print("  -> Loading Original Data...")
        for data_path_str in original_paths:
            data_path = Path(data_path_str)
            if not data_path.exists(): continue
            
            case_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
            
            for case_dir in tqdm(case_dirs, desc=f"Loading {data_path.name}"):
                case_id = case_dir.name
                for jaw_type in ['upper', 'lower']:
                    obj_file = case_dir / f"{case_id}_{jaw_type}.obj"
                    json_file = case_dir / f"{case_id}_{jaw_type}.json"
                    
                    if obj_file.exists() and json_file.exists():
                        try:
                            # LOAD FILE CONTENT NOW
                            points = self.load_obj_vertices(str(obj_file))
                            present_teeth = self.load_labels_json(str(json_file))
                            targets = self.create_16plus1_targets(present_teeth, jaw_type)
                            
                            # CACHE IT
                            self.data_cache.append({'points': points, 'targets': targets})
                        except Exception as e:
                            print(f"Error loading {case_id}: {e}")

        # ===== LOAD AUGMENTED DATA INTO RAM =====
        print("  -> Loading Augmented Data...")
        for csv_path, base_dir in augmented_sources:
            if not Path(csv_path).exists(): continue
                
            df = pd.read_csv(csv_path)
            df['new_id'] = df['new_id'].astype(str).str.strip()
            
            if 'filename' not in df.columns: continue
            
            df.columns = [str(c) if str(c).isdigit() else c for c in df.columns]
            base_path = Path(base_dir)
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {Path(csv_path).name}"):
                relative_path = row['filename']
                obj_path = base_path / relative_path
                
                case_id = row['new_id']
                jaw_type = 'lower' if 'lower' in str(case_id).lower() else 'upper'
                
                if obj_path.exists():
                    try:
                        # LOAD FILE CONTENT NOW
                        points = self.load_obj_vertices(str(obj_path))
                        present_teeth = self.load_labels_csv(row, jaw_type)
                        targets = self.create_16plus1_targets(present_teeth, jaw_type)
                        
                        # CACHE IT
                        self.data_cache.append({'points': points, 'targets': targets})
                    except Exception as e:
                        pass
        
        print(f"\n[Done] Cached {len(self.data_cache)} samples in RAM.")

    def __len__(self): return len(self.data_cache)

    def load_obj_vertices(self, obj_path):
        # Fast OBJ loader
        vertices = []
        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    if line.startswith('v '): 
                        vertices.append([float(p) for p in line.strip().split()[1:4]])
        except: return np.array([], dtype=np.float32)
        return np.array(vertices, dtype=np.float32)

    def load_labels_json(self, json_path):
        with open(json_path, 'r') as f: return set(json.load(f).get('labels', []))

    def load_labels_csv(self, csv_row, jaw_type):
        fdi_list = LOWER_FDI if jaw_type == 'lower' else UPPER_FDI
        present_teeth = set()
        for fdi in fdi_list:
            if str(fdi) in csv_row.index and float(csv_row[str(fdi)]) == 0:
                present_teeth.add(fdi)
        return present_teeth

    def create_16plus1_targets(self, present_teeth_set, jaw_type):
        is_lower = 1.0 if jaw_type == 'lower' else 0.0
        tooth_presence = np.zeros(NUM_TEETH_PER_JAW, dtype=np.float32)
        mapping = LOWER_TO_IDX if jaw_type == 'lower' else UPPER_TO_IDX
        
        for fdi_label in present_teeth_set:
            if fdi_label in mapping: 
                tooth_presence[mapping[fdi_label]] = 1.0
        
        tooth_absence = 1.0 - tooth_presence
        return np.concatenate([tooth_absence, [is_lower]]).astype(np.float32)

    def normalize_and_sample(self, points):
        # 1. Normalize
        if len(points) == 0: return np.zeros((self.num_points, 3), dtype=np.float32)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
        points_norm = points_centered / max_dist if max_dist > 0 else points_centered
        
        # 2. Sample
        if len(points_norm) == 0: return np.zeros((self.num_points, 3), dtype=np.float32)
        replace_flag = len(points_norm) < self.num_points
        indices = np.random.choice(len(points_norm), self.num_points, replace=replace_flag)
        return points_norm[indices]

    def __getitem__(self, idx):
        # FAST ACCESS: No disk I/O here
        sample = self.data_cache[idx]
        points_raw = sample['points']
        targets = sample['targets']
        
        # We process (normalize/sample) here to allow for future augmentation jitter if needed
        points_processed = self.normalize_and_sample(points_raw)
        
        return torch.from_numpy(points_processed).float(), torch.from_numpy(targets).float()

# =========================================
#  MODEL
# =========================================

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

# =========================================
#  METRICS & LOOP
# =========================================

def calculate_metrics(logits, targets):
    pred = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy().astype(int)
    tgt = targets.float().cpu().numpy().astype(int)
    _, _, f1, _ = precision_recall_fscore_support(tgt[:,:16].flatten(), pred[:,:16].flatten(), average='binary', zero_division=0)
    jaw_acc = accuracy_score(tgt[:,16], pred[:,16])
    return {'f1': f1, 'jaw_acc': jaw_acc}

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, all_logits, all_targets = 0, [], []
    
    if is_train: optimizer.zero_grad()
    
    with torch.set_grad_enabled(is_train):
        for points, labels in tqdm(loader, leave=False, desc="Train" if is_train else "Val"):
            points, labels = points.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=is_train):
                logits = model(points)
                loss = criterion(logits, labels)
            
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_loss += loss.item()
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(labels.detach().float().cpu())

    if not all_logits: return {'loss': 0, 'f1': 0, 'jaw_acc': 0}
    full_logits = torch.cat(all_logits); full_targets = torch.cat(all_targets)
    metrics = calculate_metrics(full_logits, full_targets)
    metrics['loss'] = total_loss / len(loader)
    return metrics

def print_final_report(model, loader, device):
    print(f"\n[5/5] Calculating JAW-AWARE detailed metrics on best model...")
    model.eval()
    all_preds_list, all_targets_list = [], []
    with torch.no_grad():
        for points, labels in tqdm(loader, desc="Final Eval", leave=False):
            points = points.to(device)
            with torch.amp.autocast('cuda'): logits = model(points)
            all_preds_list.append((torch.sigmoid(logits.float()) > 0.5).cpu().numpy().astype(int))
            all_targets_list.append(labels.cpu().numpy().astype(int))
    
    all_preds = np.concatenate(all_preds_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)

    # Variables for overall averaging
    valid_precs, valid_recs, valid_f1s, valid_accs = [], [], [], []
    valid_targets_flat, valid_preds_flat = [], []

    print("\n" + "="*90); print("🦷 PER-TOOTH METRICS"); print("-" * 90)
    print(f"{'FDI':<6} {'Prec':<10} {'Recall':<10} {'F1':<10} {'Acc':<10} {'Supp':<6}"); print("-" * 90)

    per_tooth_dict = OrderedDict()

    for section_name, fdi_list, jaw_label in [("UPPER", UPPER_FDI, 0.0), ("LOWER", LOWER_FDI, 1.0)]:
        print(f"\n{section_name} JAW:")
        for fdi in fdi_list:
            is_upper = (jaw_label == 0.0)
            local_idx = UPPER_TO_IDX[fdi] if is_upper else LOWER_TO_IDX[fdi]
            
            jaw_mask = (all_targets[:, 16] == jaw_label)
            if jaw_mask.sum() == 0: continue
            
            t_p, t_t = all_preds[jaw_mask, local_idx], all_targets[jaw_mask, local_idx]
            support = int(np.sum(t_t == 1))
            acc = accuracy_score(t_t, t_p)
            
            # Aggregate for overall calculation
            valid_targets_flat.extend(t_t)
            valid_preds_flat.extend(t_p)
            
            stats = {'precision': 'N/A', 'recall': 'N/A', 'f1': 'N/A', 'accuracy': acc, 'support': support}
            
            if support > 0:
                p, r, f, _ = precision_recall_fscore_support(t_t, t_p, average='binary', zero_division=0)
                valid_precs.append(p); valid_recs.append(r); valid_f1s.append(f); valid_accs.append(acc)
                stats.update({'precision': p, 'recall': r, 'f1': f})
                print(f"{fdi:<6} {p:<10.4f} {r:<10.4f} {f:<10.4f} {acc:<10.4f} {support:<6}")
            else:
                print(f"{fdi:<6} {'N/A':<10} {'N/A':<10} {'N/A':<10} {acc:<10.4f} {support:<6}")
            
            per_tooth_dict[str(fdi)] = stats

    # --- GLOBAL CALCULATIONS ---
    m_prec = np.mean(valid_precs) if valid_precs else 0.0
    m_rec = np.mean(valid_recs) if valid_recs else 0.0
    m_f1 = np.mean(valid_f1s) if valid_f1s else 0.0
    m_acc = np.mean(valid_accs) if valid_accs else 0.0
    
    # Tooth Balanced Accuracy (Flattened)
    if valid_targets_flat:
        tooth_bal_acc = balanced_accuracy_score(valid_targets_flat, valid_preds_flat)
    else:
        tooth_bal_acc = 0.0

    # Jaw Classification
    jaw_acc = accuracy_score(all_targets[:, 16], all_preds[:, 16])

    print("\n" + "="*90); print("📊 DYNAMIT AUGMENTED OVERALL SUMMARY"); print("=" * 90)
    print(f"Overall Macro F1:              {m_f1:.4f}")
    print(f"Macro Precision:               {m_prec:.4f}")
    print(f"Macro Recall:                  {m_rec:.4f}")
    print(f"Macro Accuracy:                {m_acc:.4f}")
    print(f"Tooth Balanced Accuracy:       {tooth_bal_acc:.4f}")
    print("-" * 90)
    print(f"Jaw Classification Accuracy:   {jaw_acc:.4f}")
    print("=" * 90 + "\n")
    
    # Save Metrics
    with open(Path(OUTPUT_DIR) / METRICS_FILENAME, 'w') as f:
        json.dump({
            'overall_f1': m_f1, 
            'jaw_accuracy': jaw_acc, 
            'tooth_balanced_accuracy': tooth_bal_acc,
            'per_tooth_metrics': per_tooth_dict
        }, f, indent=2)
    
    plot_per_tooth_metrics(per_tooth_dict, Path(PLOT_DIR))
    plot_confusion_matrices(valid_targets_flat, valid_preds_flat, Path(PLOT_DIR))
    print(f"✓ Results saved to: {OUTPUT_DIR}")

# =========================================
#  MAIN
# =========================================

def main():
    set_seed(SEED)
    device = torch.device("cuda:1") 
    print(f"Using device: {device}")
    
    dataset = CombinedToothDataset(ORIGINAL_DATA_PATHS, AUGMENTED_DATA_SOURCES, num_points=NUM_POINTS)
    if len(dataset) == 0: return

    train_size = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size], generator=torch.Generator().manual_seed(SEED))
    
    # Num workers can be increased now as RAM caching removes disk bottleneck
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    model = ToothClassificationModel(output_dim=TOTAL_OUTPUTS).to(device)
    criterion = Dynamit_Loss(device=device) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA)
    
    print(f"\n[3/5] Training for {NUM_EPOCHS} epochs (AUGMENTED, DYNAMIT)...")
    best_f1 = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        t_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, is_train=True)
        v_metrics = run_epoch(model, val_loader, criterion, device, is_train=False)
        scheduler.step()
        
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Loss: {t_metrics['loss']:.4f}/{v_metrics['loss']:.4f} | F1: {t_metrics['f1']:.4f}/{v_metrics['f1']:.4f} | Jaw: {t_metrics['jaw_acc']:.4f}/{v_metrics['jaw_acc']:.4f}")
        
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_f1': v_metrics['f1']}, Path(OUTPUT_DIR) / LAST_MODEL_FILENAME)

        if v_metrics['f1'] > best_f1:
            best_f1 = v_metrics['f1']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_f1': best_f1}, Path(OUTPUT_DIR) / BEST_MODEL_FILENAME)
            print(f"        → ✓ Best F1 saved: {best_f1:.4f}")
            
        if early_stop(v_metrics['f1']): 
            print("Early stopping triggered.")
            break

    print("\nLoading best model for final report...")
    checkpoint = torch.load(Path(OUTPUT_DIR) / BEST_MODEL_FILENAME)
    model.load_state_dict(checkpoint['model_state_dict'])
    print_final_report(model, val_loader, device)

if __name__ == "__main__":
    main()