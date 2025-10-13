# baseline_model.py using 0 as missing, 1 as present for 32 permanent teeth in FDI notation
#pointnet with bceloss


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import subprocess
from collections import OrderedDict

# ============= CONFIGURATION =============
DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]
OUTPUT_DIR = "/home/user/tbrighton/Scripts/Training/3D/trained_models"
PLOT_DIR = "/home/user/tbrighton/Scripts/Training/3D/plots"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 35
LEARNING_RATE = 0.001
NUM_POINTS = 4096
NUM_TEETH = 32 # This is correct, as there are 32 permanent teeth.
SEED = 41

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 0.0001

# --- NEW: FDI Notation Mapping ---
# Canonical list of all 32 permanent teeth in FDI notation, sorted
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])

# Create a mapping from each FDI label to a unique 0-31 index for the model's output layer
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
# Create a reverse mapping for correctly reporting results
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}

# GPU Configuration
available_gpus = []
device = None
# =========================================

# ... get_free_gpus and set_seed functions are fine ...
def get_free_gpus(threshold_mb=1000, max_gpus=2):
    # ... (code is correct)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        free_gpus = [int(line.split(', ')[0]) for line in result.stdout.strip().split('\n') if int(line.split(', ')[1]) < threshold_mb]
        if len(free_gpus) > max_gpus: free_gpus = free_gpus[:max_gpus]
        if not free_gpus: print("Warning: No free GPUs found, using GPU 0"); return [0]
        print(f"✓ Free GPUs detected: {free_gpus}"); return free_gpus
    except Exception as e:
        print(f"Error detecting free GPUs: {e}\nFalling back to GPU 0"); return [0]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class ToothClassificationDataset(Dataset):
    """Dataset for tooth presence/absence classification using correct FDI notation."""
    
    def __init__(self, data_paths, num_points=2048):
        self.num_points = num_points
        self.samples = []
        # --- CORRECTED file discovery logic ---
        for data_path_str in data_paths:
            data_path = Path(data_path_str)
            if not data_path.exists(): continue
            jaw_type = "lower" if "lower" in str(data_path) else "upper"
            for case_dir in sorted(data_path.iterdir()):
                if case_dir.is_dir():
                    case_id = case_dir.name
                    obj_file = case_dir / f"{case_id}_{jaw_type}.obj"
                    json_file = case_dir / f"{case_id}_{jaw_type}.json"
                    if obj_file.exists() and json_file.exists():
                        self.samples.append({'obj': str(obj_file), 'json': str(json_file)})
        print(f"[Info] Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def load_obj_vertices(self, obj_path):
        vertices = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertices.append([float(p) for p in parts[1:4]])
        return np.array(vertices, dtype=np.float32)
    
    def load_labels(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Return a set of unique labels, which is all we need
        return set(data.get('labels', []))
    
    def normalize_point_cloud(self, points):
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
        if max_dist > 0:
            points_centered /= max_dist
        return points_centered
    
    # --- COMPLETELY REWRITTEN AND CORRECTED ---
    def create_tooth_presence_labels(self, vertex_labels_set):
        """Create a correct binary label vector (0-31) from a set of FDI labels."""
        tooth_presence = np.zeros(NUM_TEETH, dtype=np.float32)
        for fdi_label in vertex_labels_set:
            # Look up the correct 0-31 index for the given FDI label
            index = FDI_TO_INDEX.get(fdi_label)
            # If the label is a valid permanent tooth, set the corresponding index to 1
            if index is not None:
                tooth_presence[index] = 1.0
        return tooth_presence
    
    def sample_points(self, points):
        num_vertices = len(points)
        if num_vertices == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)
        replace = num_vertices < self.num_points
        indices = np.random.choice(num_vertices, self.num_points, replace=replace)
        return points[indices]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = self.load_obj_vertices(sample['obj'])
        vertex_labels_set = self.load_labels(sample['json'])
        
        # This function now produces the correct label vector for the model
        tooth_labels = self.create_tooth_presence_labels(vertex_labels_set)
        
        points = self.normalize_point_cloud(points)
        points = self.sample_points(points)
        
        points = torch.from_numpy(points).float()
        tooth_labels = torch.from_numpy(tooth_labels).float()
        
        # The case_id is not used in training, so we can remove it from the return
        return points, tooth_labels


# ... Your PointNetEncoder and ToothClassificationModel are fine ...
# Their architecture is independent of the labeling system, as they just output a 32-dim vector.
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

# ... calculate_classification_metrics is fine (it's for the micro-average) ...
def calculate_classification_metrics(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(int); target = target.cpu().numpy().astype(int)
    pred_flat, target_flat = pred.flatten(), target.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(target_flat, pred_flat, average='binary', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy_score(target_flat, pred_flat)}


# --- COMPLETELY REWRITTEN AND CORRECTED ---
def calculate_per_tooth_metrics(pred, target, num_teeth=32):
    """
    Calculate per-tooth and macro-averaged metrics using the correct FDI labels.
    """
    pred = (pred > 0.5).cpu().numpy().astype(int)
    target = target.cpu().numpy().astype(int)
    
    # Use an OrderedDict to keep the report sorted by FDI label
    per_tooth_metrics = OrderedDict()
    
    # Loop through the model's output indices (0-31)
    for tooth_idx in range(num_teeth):
        # Use the reverse map to get the correct FDI label for this index
        fdi_label = INDEX_TO_FDI[tooth_idx]
        
        tooth_pred = pred[:, tooth_idx]
        tooth_target = target[:, tooth_idx]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            tooth_target, tooth_pred, average='binary', zero_division=0
        )
        accuracy = accuracy_score(tooth_target, tooth_pred)
        support = int(tooth_target.sum())
        
        # Store metrics using the correct FDI label as the key
        per_tooth_metrics[fdi_label] = {
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': accuracy, 'support': support
        }
    
    # Macro-average calculation is correct
    macro_precision = np.mean([m['precision'] for m in per_tooth_metrics.values()])
    macro_recall = np.mean([m['recall'] for m in per_tooth_metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in per_tooth_metrics.values()])
    macro_accuracy = np.mean([m['accuracy'] for m in per_tooth_metrics.values()])
    
    macro_metrics = {
        'macro_precision': macro_precision, 'macro_recall': macro_recall,
        'macro_f1': macro_f1, 'macro_accuracy': macro_accuracy
    }
    
    return per_tooth_metrics, macro_metrics

# ... train_epoch is fine ...
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train(); total_loss = 0
    all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
    # Removed case_id from dataloader loop
    for points, labels in tqdm(dataloader, desc="Training", leave=False):
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(points), labels); loss.backward(); optimizer.step()
        metrics = calculate_classification_metrics(torch.sigmoid(model(points)), labels)
        total_loss += loss.item()
        for key in all_metrics: all_metrics[key] += metrics[key]
    num_batches = len(dataloader)
    avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
    avg_metrics['loss'] = total_loss / num_batches
    return avg_metrics


# ... validate is fine ...
def validate(model, dataloader, criterion, device):
    model.eval(); total_loss = 0; all_preds, all_targets = [], []
    with torch.no_grad():
        # Removed case_id from dataloader loop
        for points, labels in tqdm(dataloader, desc="Validating", leave=False):
            points, labels = points.to(device), labels.to(device)
            logits = model(points); loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            total_loss += loss.item()
            all_preds.append(probs); all_targets.append(labels)
    all_preds = torch.cat(all_preds, dim=0); all_targets = torch.cat(all_targets, dim=0)
    avg_metrics = calculate_classification_metrics(all_preds, all_targets)
    avg_metrics['loss'] = total_loss / len(dataloader)
    return avg_metrics, all_preds, all_targets

# ... plot_training_curves is fine ...
def plot_training_curves(history, save_dir):
    # ... (code is correct)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12)); fig.suptitle('PointNet Training Metrics', fontsize=16, fontweight='bold')
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss'); axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1'); axes[0, 1].plot(epochs, history['val_f1'], 'r-', label='Val F1')
    axes[0, 1].set_title('F1 Score (Micro)'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(epochs, history['train_acc'], 'b-', label='Train Acc'); axes[1, 0].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1, 0].set_title('Accuracy'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(epochs, history['train_precision'], 'b--', label='Train Precision'); axes[1, 1].plot(epochs, history['val_precision'], 'r--', label='Val Precision')
    axes[1, 1].plot(epochs, history['train_recall'], 'b:', label='Train Recall'); axes[1, 1].plot(epochs, history['val_recall'], 'r:', label='Val Recall')
    axes[1, 1].set_title('Precision and Recall'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(Path(save_dir) / 'training_metrics.png', dpi=300); plt.close()
    print(f"\n✓ Training plots saved to: {Path(save_dir) / 'training_metrics.png'}")


def main():
    global available_gpus, device
    set_seed(SEED)
    
    print("\n[0/5] Detecting free GPUs...")
    available_gpus = get_free_gpus(); device = torch.device(f"cuda:{available_gpus[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Primary device: {device}")
    
    print("\n[1/5] Loading dataset...")
    dataset = ToothClassificationDataset(DATA_PATHS, num_points=NUM_POINTS)
    train_size = int(0.8 * len(dataset)); val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    print("\n[2/5] Initializing PointNet model...")
    model = ToothClassificationModel(num_teeth=NUM_TEETH).to(device)
    if len(available_gpus) > 1: model = torch.nn.DataParallel(model, device_ids=available_gpus)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 'train_acc': [], 'val_acc': [], 'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': []}
    
    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs..."); print("=" * 80)
    best_f1 = 0.0; best_val_preds = None; best_val_targets = None
    
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_preds, val_targets = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        for key in ['loss', 'f1', 'acc', 'precision', 'recall']:
            history[f'train_{key}'].append(train_metrics[key.replace('acc', 'accuracy')])
            history[f'val_{key}'].append(val_metrics[key.replace('acc', 'accuracy')])
            
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']; best_val_preds = val_preds; best_val_targets = val_targets
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict()}, Path(OUTPUT_DIR) / "best_model_f1.pth")
            print(f"         → ✓ Best F1 model saved (F1: {val_metrics['f1']:.4f})")

    print("\n" + "=" * 80 + f"\n[4/5] Training complete!\nBest validation F1 (micro): {best_f1:.4f}")
    
    print("\n" + "=" * 80 + "\n[4.5/5] Calculating per-tooth metrics on best model...")
    
    per_tooth_metrics, macro_metrics = calculate_per_tooth_metrics(best_val_preds, best_val_targets, num_teeth=NUM_TEETH)
    
    print("\n📊 MACRO-AVERAGED METRICS (across all 32 teeth):"); print("-" * 80)
    print(f"  Macro Precision: {macro_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {macro_metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {macro_metrics['macro_f1']:.4f}")
    print(f"  Macro Accuracy:  {macro_metrics['macro_accuracy']:.4f}")
    
    # --- CORRECTED Per-Tooth Metrics Printing ---
    print("\n🦷 PER-TOOTH METRICS (FDI Notation):"); print("-" * 80)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    # The keys in per_tooth_metrics are now the correct FDI labels
    for fdi_label, metrics in per_tooth_metrics.items():
        print(f"Tooth {fdi_label:<5} "
              f"{metrics['precision']:>10.4f}   "
              f"{metrics['recall']:>10.4f}   "
              f"{metrics['f1']:>10.4f}   "
              f"{metrics['accuracy']:>10.4f}   "
              f"{metrics['support']:>8}")
    
    metrics_file = Path(OUTPUT_DIR) / "detailed_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({'macro_metrics': macro_metrics, 'per_tooth_metrics': per_tooth_metrics}, f, indent=2)
    print(f"\n✓ Detailed metrics saved to: {metrics_file}")
    
    print(f"\n[5/5] Generating training plots...")
    plot_training_curves(history, PLOT_DIR)
    print("\n✓ All done!")


if __name__ == "__main__":
    main()