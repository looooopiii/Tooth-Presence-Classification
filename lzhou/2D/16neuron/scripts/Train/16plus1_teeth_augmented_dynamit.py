import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import subprocess
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet50
import json
import os

# Dataset Training Configuration
USE_ORIGINAL = True   # if use original dataset
USE_AUGMENTED = True  # if use augmented dataset (2400 samples)
USE_RANDOM = True     # if use random augmented (3600 samples)

# Original dataset roots
JSON_ROOT_LOWER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower"
JSON_ROOT_UPPER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"

IMG_ROOT_LOWER = "/home/user/lzhou/week15/render_output/train/lowerjaw"
IMG_ROOT_UPPER = "/home/user/lzhou/week15/render_output/train/upperjaw"

# CSV files
AUGMENTED_CSV = "/home/user/lzhou/week10/output/augment_test/train_labels_augmented.csv"
RANDOM_CSV = "/home/user/lzhou/week10/output/augment_random/train_labels_random.csv"

# Rendered image directories
RENDER_ROOT = "/home/user/lzhou/week15/render_output/render_aug_random"
RENDER_ROOT_TEST = "/home/user/lzhou/week15/render_output/render_aug_test"

# Output directories
OUTPUT_DIR = "/home/user/lzhou/week15-17/output/Train2D/Augmented_16plus1teeth_dynamit"
PLOT_DIR = f"{OUTPUT_DIR}/plots"

# Model filenames
BEST_MODEL_FILENAME = "augmented_dynamit_best_2d_16plus1teeth.pth"
LAST_MODEL_FILENAME = "augmented_dynamit_last_2d_16plus1teeth.pth"
PLOT_FILENAME = "training_metrics_augmented_dynamit_16plus1.png"
METRICS_FILENAME = "detailed_metrics_augmented_dynamit_16plus1.json"

# ==================== HYPERPARAMETERS ====================
BATCH_SIZE = 32       # Increased due to more data
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
IMG_SIZE = 256
NUM_TEETH_POSITIONS = 16  # 16 tooth positions
NUM_OUTPUTS = 17  # 16 teeth + 1 jaw classifier
SEED = 41
BACKBONE = "resnet18"

# Early stopping
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001
EARLY_STOPPING_METRIC = "val_pr_auc_macro"  # "val_pr_auc_macro" or "val_macro_f1"

# Dropout
DROPOUT_RATE = 0.5

# Macro metrics settings
MACRO_SUPPORT_MIN = 1  # Only include positions with >= this many positives

# Threshold tuning
THRESHOLD_STRATEGY = "per_position"  # "fixed" or "per_position"
FIXED_TEETH_THRESHOLD = 0.5
JAW_THRESHOLD = 0.5
THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)

# FDI Notation - 16 positions mapping
POSITION_TO_FDI_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
POSITION_TO_FDI_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

# Create reverse mappings
FDI_UPPER_TO_POSITION = {fdi: pos for pos, fdi in enumerate(POSITION_TO_FDI_UPPER)}
FDI_LOWER_TO_POSITION = {fdi: pos for pos, fdi in enumerate(POSITION_TO_FDI_LOWER)}

# For CSV reading (32 teeth in original format)
UPPER_TEETH = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_TEETH = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
ALL_TEETH = sorted(UPPER_TEETH + LOWER_TEETH)

# GPU config
available_gpus = []
device = None


def _parse_cuda_visible_devices():
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_value is None:
        return None
    env_value = env_value.strip()
    if not env_value:
        return []
    parts = [part.strip() for part in env_value.split(",") if part.strip()]
    if not all(part.isdigit() for part in parts):
        return None
    return [int(part) for part in parts]


def get_free_gpus(threshold_mb=1000, max_gpus=2):
    """Get free GPU indices"""
    visible_physical = _parse_cuda_visible_devices()
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        used_by_index = {}
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 2:
                continue
            idx_str, mem_str = parts
            if idx_str.isdigit() and mem_str.isdigit():
                used_by_index[int(idx_str)] = int(mem_str)

        free_physical = [
            idx for idx, used in used_by_index.items()
            if used < threshold_mb
        ]

        if visible_physical is not None:
            visible_free = [idx for idx in visible_physical if idx in free_physical]
            if not visible_free:
                print("Warning: No free visible GPUs found, using visible GPU 0")
                return [0]
            free_gpus = [visible_physical.index(idx) for idx in visible_free]
        else:
            free_gpus = free_physical

        if len(free_gpus) > max_gpus:
            free_gpus = free_gpus[:max_gpus]
        if not free_gpus:
            print("Warning: No free GPUs found, using GPU 0")
            return [0]
        print(f" Free GPUs detected: {free_gpus}")
        return free_gpus
    except Exception as e:
        print(f"Error detecting free GPUs: {e}\nFalling back to GPU 0")
        return [0]


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_case_and_jaw(png_path):
    """Parse case ID and jaw type from PNG filename"""
    name = png_path.stem
    if name.endswith("_top"):
        core = name[:-4]
    else:
        core = name
    if core.endswith("_lower"):
        return core[:-6], "lower"
    if core.endswith("_upper"):
        return core[:-6], "upper"
    parts = core.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def load_json_labels(json_path):
    """Load labels from JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return set(data.get('labels', []))
    except Exception as e:
        print(f"  Warning: failed to read {json_path}: {e}")
        return set()


def create_tooth_missing_vector_16plus1_from_json(vertex_labels_set, jaw_type):
    """
    Create 16+1 vector from JSON labels (original dataset)
    """
    output_vector = np.zeros(NUM_OUTPUTS, dtype=np.float32)
    tooth_presence = np.zeros(NUM_TEETH_POSITIONS, dtype=np.float32)
    
    if jaw_type == "upper":
        for fdi_label in vertex_labels_set:
            if fdi_label in FDI_UPPER_TO_POSITION:
                pos = FDI_UPPER_TO_POSITION[fdi_label]
                tooth_presence[pos] = 1.0
        jaw_label = 0.0  # Upper jaw = 0
    else:  # lower
        for fdi_label in vertex_labels_set:
            if fdi_label in FDI_LOWER_TO_POSITION:
                pos = FDI_LOWER_TO_POSITION[fdi_label]
                tooth_presence[pos] = 1.0
        jaw_label = 1.0  # Lower jaw = 1
    
    tooth_missing = 1.0 - tooth_presence
    output_vector[:NUM_TEETH_POSITIONS] = tooth_missing
    output_vector[NUM_TEETH_POSITIONS] = jaw_label
    
    return output_vector


def create_tooth_missing_vector_16plus1_from_csv(csv_row, jaw_type):
    """
    Create 16+1 vector from CSV row (augmented dataset)
    CSV has 32 teeth columns, convert to 16 positions + jaw
    """
    output_vector = np.zeros(NUM_OUTPUTS, dtype=np.float32)
    tooth_missing = np.zeros(NUM_TEETH_POSITIONS, dtype=np.float32)
    
    if jaw_type == "upper":
        for pos, fdi in enumerate(POSITION_TO_FDI_UPPER):
            col_name = str(fdi)
            if col_name in csv_row:
                tooth_missing[pos] = float(csv_row[col_name])
            else:
                tooth_missing[pos] = 0.0
        jaw_label = 0.0  # Upper = 0
    else:  # lower
        for pos, fdi in enumerate(POSITION_TO_FDI_LOWER):
            col_name = str(fdi)
            if col_name in csv_row:
                tooth_missing[pos] = float(csv_row[col_name])
            else:
                tooth_missing[pos] = 0.0
        jaw_label = 1.0  # Lower = 1
    
    output_vector[:NUM_TEETH_POSITIONS] = tooth_missing
    output_vector[NUM_TEETH_POSITIONS] = jaw_label
    
    return output_vector


def build_original_samples(img_roots, json_roots):
    """Build sample list from original dataset"""
    samples = []
    lower_img_root, upper_img_root = [Path(p) for p in img_roots]
    lower_json_root, upper_json_root = [Path(p) for p in json_roots]

    def add_samples(img_root, jaw, json_root):
        if not img_root.exists():
            print(f"  Warning: image root not found: {img_root}")
            return 0
        if not json_root.exists():
            print(f"  Warning: JSON root not found: {json_root}")
            return 0
        count = 0
        for png in sorted(img_root.glob("*_top.png")):
            case_id, cur_jaw = parse_case_and_jaw(png)
            if cur_jaw != jaw or case_id is None:
                continue
            json_path = json_root / case_id / f"{case_id}_{jaw}.json"
            if not json_path.exists():
                continue
            vertex_labels_set = load_json_labels(json_path)
            tooth_labels = create_tooth_missing_vector_16plus1_from_json(vertex_labels_set, jaw)
            samples.append({
                'png_path': str(png),
                'labels': tooth_labels,
                'jaw_type': jaw,
                'case_id': case_id
            })
            count += 1
        return count

    add_samples(lower_img_root, "lower", lower_json_root)
    add_samples(upper_img_root, "upper", upper_json_root)
    return samples


# ==================== Dynamic Loss ====================
class DynamitLoss16Plus1(nn.Module):
    """
    Combined loss for 16+1 architecture:
    - Dynamic Loss for teeth (0-15): batch-adaptive class balancing
    - Standard BCE for jaw (16): balanced binary classification
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        # Separate teeth (0-15) and jaw (16)
        teeth_logits = logits[:, :NUM_TEETH_POSITIONS]
        teeth_targets = targets[:, :NUM_TEETH_POSITIONS]
        jaw_logits = logits[:, NUM_TEETH_POSITIONS]
        jaw_targets = targets[:, NUM_TEETH_POSITIONS]
        
        # ===== Dynamic Loss for Teeth =====
        S_pos = (teeth_targets == 1).sum().float()
        S_neg = (teeth_targets == 0).sum().float()
        
        pos_coeff_val = min(1.0, (S_neg / S_pos).item()) if S_pos > 0 else 0.0
        neg_coeff_val = min(1.0, (S_pos / S_neg).item()) if S_neg > 0 else 0.0
        
        pos_coeff = torch.tensor(pos_coeff_val, device=self.device)
        neg_coeff = torch.tensor(neg_coeff_val, device=self.device)
        
        teeth_weights = torch.where(teeth_targets == 1, pos_coeff, neg_coeff)
        teeth_loss = F.binary_cross_entropy_with_logits(
            teeth_logits, teeth_targets, weight=teeth_weights
        )
        
        # ===== Standard BCE for Jaw =====
        jaw_loss = self.bce(jaw_logits, jaw_targets)
        
        # Combined loss (weighted)
        total_loss = 0.8 * teeth_loss + 0.2 * jaw_loss
        
        return total_loss


# ==================== Dataset Class ====================
class AugmentedToothDataset16Plus1(Dataset):
    """Dataset for augmented data with 16+1 output format"""
    def __init__(self, csv_paths, render_root, transform=None):
        self.samples = []
        self.transform = transform
        if isinstance(render_root, (list, tuple, set)):
            self.render_roots = [Path(r) for r in render_root]
        else:
            self.render_roots = [Path(render_root)]
        
        for csv_path in csv_paths:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            
            for idx, row in df.iterrows():
                obj_filename = row['filename']
                obj_path = Path(obj_filename)
                case_dir = None
                if obj_path.parts and obj_path.parts[0].lower() in ('upper', 'lower'):
                    jaw_type = obj_path.parts[0].lower()
                    if len(obj_path.parts) > 1:
                        case_dir = obj_path.parts[1]
                elif 'lower' in obj_filename.lower():
                    jaw_type = 'lower'
                elif 'upper' in obj_filename.lower():
                    jaw_type = 'upper'
                else:
                    continue
                
                case_id = row.get('new_id', Path(obj_filename).stem)
                png_filename = f"{case_id}_top.png"
                png_path = None
                for root in self.render_roots:
                    if case_dir:
                        candidate = root / jaw_type / case_dir / png_filename
                        if candidate.exists():
                            png_path = candidate
                            break
                    candidate = root / jaw_type / png_filename
                    if candidate.exists():
                        png_path = candidate
                        break
                
                if png_path is None:
                    alt_names = [
                        f"{case_id}.png",
                        f"{Path(obj_filename).stem}_top.png",
                        f"{Path(obj_filename).stem}.png"
                    ]
                    found = False
                    for alt_name in alt_names:
                        for root in self.render_roots:
                            if case_dir:
                                alt_path = root / jaw_type / case_dir / alt_name
                                if alt_path.exists():
                                    png_path = alt_path
                                    found = True
                                    break
                            alt_path = root / jaw_type / alt_name
                            if alt_path.exists():
                                png_path = alt_path
                                found = True
                                break
                        if found:
                            break
                    
                    if not found:
                        continue
                
                tooth_labels = create_tooth_missing_vector_16plus1_from_csv(row, jaw_type)
                
                self.samples.append({
                    'png_path': str(png_path),
                    'labels': tooth_labels,
                    'jaw_type': jaw_type,
                    'case_id': case_id
                })
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            img = Image.open(sample['png_path']).convert('RGB')
        except Exception as e:
            print(f"  Error loading image {sample['png_path']}: {e}")
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
        
        labels = torch.from_numpy(sample['labels']).float()
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, labels, sample['jaw_type']


# ==================== Model ====================
class ResNetMultiLabel16Plus1(nn.Module):
    def __init__(self, backbone="resnet18", num_outputs=17, dropout_rate=0.5):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights="IMAGENET1K_V2")
        else:
            net = resnet18(weights="IMAGENET1K_V1")
        
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


# ==================== Metrics ====================
def _expand_teeth_thresholds(teeth_thresholds):
    if isinstance(teeth_thresholds, (int, float, np.floating)):
        return np.full(NUM_TEETH_POSITIONS, float(teeth_thresholds), dtype=np.float32)
    thresholds = np.array(teeth_thresholds, dtype=np.float32)
    if thresholds.shape[0] != NUM_TEETH_POSITIONS:
        raise ValueError("teeth_thresholds must be a scalar or length NUM_TEETH_POSITIONS")
    return thresholds


def _select_best_threshold(y_true, y_prob, threshold_grid):
    best_t = 0.5
    best_f1 = -1.0
    for t in threshold_grid:
        pred = (y_prob > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, pred, average='binary', zero_division=0, pos_label=1
        )
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def tune_teeth_thresholds(pred, target, threshold_grid):
    pred_np = pred[:, :NUM_TEETH_POSITIONS].cpu().numpy()
    target_np = target[:, :NUM_TEETH_POSITIONS].cpu().numpy().astype(int)
    thresholds = np.full(NUM_TEETH_POSITIONS, 0.5, dtype=np.float32)
    for pos in range(NUM_TEETH_POSITIONS):
        thresholds[pos] = _select_best_threshold(
            target_np[:, pos], pred_np[:, pos], threshold_grid
        )
    return thresholds


def resolve_teeth_thresholds(pred, target):
    if THRESHOLD_STRATEGY == "fixed":
        return FIXED_TEETH_THRESHOLD
    if THRESHOLD_STRATEGY == "per_position":
        return tune_teeth_thresholds(pred, target, THRESHOLD_GRID)
    raise ValueError(f"Unknown THRESHOLD_STRATEGY: {THRESHOLD_STRATEGY}")


def _safe_average_precision(y_true, y_score):
    if np.all(y_true == 0) or np.all(y_true == 1):
        return None
    return average_precision_score(y_true, y_score)


def compute_pr_auc(teeth_probs, teeth_targets):
    flat_targets = teeth_targets.flatten()
    flat_probs = teeth_probs.flatten()
    micro_ap = _safe_average_precision(flat_targets, flat_probs)
    micro_ap = float(micro_ap) if micro_ap is not None else 0.0

    macro_aps = []
    for pos in range(NUM_TEETH_POSITIONS):
        ap = _safe_average_precision(teeth_targets[:, pos], teeth_probs[:, pos])
        if ap is not None:
            macro_aps.append(ap)
    macro_ap = float(np.mean(macro_aps)) if macro_aps else 0.0
    return micro_ap, macro_ap


def calculate_metrics_16plus1(pred, target, teeth_thresholds=FIXED_TEETH_THRESHOLD, jaw_threshold=JAW_THRESHOLD):
    """Calculate metrics for 16+1 output"""
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy().astype(int)
    
    teeth_thresholds = _expand_teeth_thresholds(teeth_thresholds)
    teeth_pred = (pred_np[:, :NUM_TEETH_POSITIONS] > teeth_thresholds).astype(int)
    teeth_target = target_np[:, :NUM_TEETH_POSITIONS]
    jaw_pred = (pred_np[:, NUM_TEETH_POSITIONS] > jaw_threshold).astype(int)
    jaw_target = target_np[:, NUM_TEETH_POSITIONS]
    
    # Teeth metrics (missing class)
    teeth_pred_flat = teeth_pred.flatten()
    teeth_target_flat = teeth_target.flatten()
    
    missing_precision, missing_recall, missing_f1, _ = precision_recall_fscore_support(
        teeth_target_flat, teeth_pred_flat, average='binary', zero_division=0, pos_label=1
    )
    teeth_accuracy = accuracy_score(teeth_target_flat, teeth_pred_flat)
    teeth_balanced_accuracy = balanced_accuracy_score(teeth_target_flat, teeth_pred_flat)
    pr_auc_micro, pr_auc_macro = compute_pr_auc(pred_np[:, :NUM_TEETH_POSITIONS], teeth_target)
    
    # Jaw metrics
    jaw_precision, jaw_recall, jaw_f1, _ = precision_recall_fscore_support(
        jaw_target, jaw_pred, average='binary', zero_division=0, pos_label=1
    )
    jaw_accuracy = accuracy_score(jaw_target, jaw_pred)
    
    return {
        'missing_precision': missing_precision,
        'missing_recall': missing_recall,
        'missing_f1': missing_f1,
        'missing_accuracy': teeth_accuracy,
        'missing_balanced_accuracy': teeth_balanced_accuracy,
        'missing_pr_auc_micro': pr_auc_micro,
        'missing_pr_auc_macro': pr_auc_macro,
        'jaw_precision': jaw_precision,
        'jaw_recall': jaw_recall,
        'jaw_f1': jaw_f1,
        'jaw_accuracy': jaw_accuracy
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
            'macro_accuracy': 0.0,
            'macro_positions': 0
        }
    return {
        'macro_precision': float(np.mean([metrics_dict[k]['precision'] for k in keys])),
        'macro_recall': float(np.mean([metrics_dict[k]['recall'] for k in keys])),
        'macro_f1': float(np.mean([metrics_dict[k]['f1'] for k in keys])),
        'macro_accuracy': float(np.mean([metrics_dict[k]['accuracy'] for k in keys])),
        'macro_positions': len(keys)
    }


def calculate_per_position_metrics(
    pred, target, jaw_types, teeth_thresholds=FIXED_TEETH_THRESHOLD, jaw_threshold=JAW_THRESHOLD
):
    """Calculate per-position metrics"""
    pred_np = pred.cpu().numpy()
    target = target.cpu().numpy().astype(int)
    
    teeth_thresholds = _expand_teeth_thresholds(teeth_thresholds)
    teeth_pred = (pred_np[:, :NUM_TEETH_POSITIONS] > teeth_thresholds).astype(int)
    teeth_target = target[:, :NUM_TEETH_POSITIONS]
    jaw_pred = (pred_np[:, NUM_TEETH_POSITIONS] > jaw_threshold).astype(int)
    jaw_target = target[:, NUM_TEETH_POSITIONS]
    
    upper_mask = np.array([jaw == "upper" for jaw in jaw_types])
    lower_mask = np.array([jaw == "lower" for jaw in jaw_types])
    
    per_position_metrics = OrderedDict()
    
    for pos in range(NUM_TEETH_POSITIONS):
        if upper_mask.sum() > 0:
            upper_pred = teeth_pred[upper_mask, pos]
            upper_target = teeth_target[upper_mask, pos]
            fdi_upper = POSITION_TO_FDI_UPPER[pos]
            
            if len(upper_target) > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    upper_target, upper_pred, average='binary', zero_division=0, pos_label=1
                )
                accuracy = accuracy_score(upper_target, upper_pred)
                support = int(np.sum(upper_target == 1))
                
                per_position_metrics[f"Upper_{fdi_upper}"] = {
                    'position': pos,
                    'fdi': fdi_upper,
                    'jaw': 'upper',
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'accuracy': float(accuracy),
                    'support': support
                }
        
        if lower_mask.sum() > 0:
            lower_pred = teeth_pred[lower_mask, pos]
            lower_target = teeth_target[lower_mask, pos]
            fdi_lower = POSITION_TO_FDI_LOWER[pos]
            
            if len(lower_target) > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    lower_target, lower_pred, average='binary', zero_division=0, pos_label=1
                )
                accuracy = accuracy_score(lower_target, lower_pred)
                support = int(np.sum(lower_target == 1))
                
                per_position_metrics[f"Lower_{fdi_lower}"] = {
                    'position': pos,
                    'fdi': fdi_lower,
                    'jaw': 'lower',
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'accuracy': float(accuracy),
                    'support': support
                }
    
    upper_macro = macro_from_positions(
        per_position_metrics, support_min=MACRO_SUPPORT_MIN, jaw_filter='upper'
    )
    lower_macro = macro_from_positions(
        per_position_metrics, support_min=MACRO_SUPPORT_MIN, jaw_filter='lower'
    )
    macros_present = [m for m in (upper_macro, lower_macro) if m['macro_positions'] > 0]
    if macros_present:
        macro_metrics = {
            'macro_precision': float(np.mean([m['macro_precision'] for m in macros_present])),
            'macro_recall': float(np.mean([m['macro_recall'] for m in macros_present])),
            'macro_f1': float(np.mean([m['macro_f1'] for m in macros_present])),
            'macro_accuracy': float(np.mean([m['macro_accuracy'] for m in macros_present])),
            'macro_support_min': MACRO_SUPPORT_MIN,
            'macro_positions_per_jaw': int(np.mean([m['macro_positions'] for m in macros_present])),
            'macro_jaws_included': len(macros_present)
        }
    else:
        macro_metrics = {
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'macro_accuracy': 0.0,
            'macro_support_min': MACRO_SUPPORT_MIN,
            'macro_positions_per_jaw': 0,
            'macro_jaws_included': 0
        }
    
    per_jaw_macro = {
        'upper': upper_macro,
        'lower': lower_macro
    }
    
    return per_position_metrics, macro_metrics, per_jaw_macro


# ==================== Training ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_jaws = [], [], []
    
    for imgs, labels, jaw_types in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits.detach()))
        all_labels.append(labels)
        all_jaws.extend(jaw_types)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = calculate_metrics_16plus1(all_preds, all_labels)
    
    return {
        'loss': total_loss / len(dataloader),
        'missing_f1': metrics['missing_f1'],
        'missing_precision': metrics['missing_precision'],
        'missing_recall': metrics['missing_recall'],
        'missing_accuracy': metrics['missing_accuracy'],
        'missing_balanced_accuracy': metrics['missing_balanced_accuracy'],
        'missing_pr_auc_micro': metrics['missing_pr_auc_micro'],
        'missing_pr_auc_macro': metrics['missing_pr_auc_macro'],
        'jaw_accuracy': metrics['jaw_accuracy'],
        'jaw_f1': metrics['jaw_f1']
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_jaws = [], [], []
    
    with torch.no_grad():
        for imgs, labels, jaw_types in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits))
            all_labels.append(labels)
            all_jaws.extend(jaw_types)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    teeth_thresholds = resolve_teeth_thresholds(all_preds, all_labels)
    metrics = calculate_metrics_16plus1(
        all_preds, all_labels, teeth_thresholds=teeth_thresholds, jaw_threshold=JAW_THRESHOLD
    )
    _, macro_metrics, per_jaw_macro = calculate_per_position_metrics(
        all_preds, all_labels, all_jaws, teeth_thresholds=teeth_thresholds, jaw_threshold=JAW_THRESHOLD
    )
    
    return {
        'loss': total_loss / len(dataloader),
        'missing_f1': metrics['missing_f1'],
        'missing_precision': metrics['missing_precision'],
        'missing_recall': metrics['missing_recall'],
        'missing_accuracy': metrics['missing_accuracy'],
        'missing_balanced_accuracy': metrics['missing_balanced_accuracy'],
        'missing_pr_auc_micro': metrics['missing_pr_auc_micro'],
        'missing_pr_auc_macro': metrics['missing_pr_auc_macro'],
        'jaw_accuracy': metrics['jaw_accuracy'],
        'jaw_f1': metrics['jaw_f1'],
        'macro_f1': macro_metrics['macro_f1'],
        'macro_metrics': macro_metrics,
        'per_jaw_macro': per_jaw_macro,
        'teeth_thresholds': _expand_teeth_thresholds(teeth_thresholds).tolist(),
        'jaw_threshold': JAW_THRESHOLD
    }, all_preds, all_labels, all_jaws


# ==================== Plotting ====================
def plot_training_curves(history, save_dir, filename):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ResNet18 2D Training - 16+1 Neurons', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Teeth F1
    axes[0, 1].plot(epochs, history['train_missing_f1'], 'b-', label='Train F1')
    axes[0, 1].plot(epochs, history['val_missing_f1'], 'r-', label='Val F1')
    axes[0, 1].set_title('Missing F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Teeth Accuracy
    axes[0, 2].plot(epochs, history['train_missing_acc'], 'b-', label='Train Acc')
    axes[0, 2].plot(epochs, history['val_missing_acc'], 'r-', label='Val Acc')
    axes[0, 2].set_title('Missing Accuracy')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Jaw Accuracy
    axes[1, 0].plot(epochs, history['train_jaw_acc'], 'b-', label='Train Jaw Acc')
    axes[1, 0].plot(epochs, history['val_jaw_acc'], 'r-', label='Val Jaw Acc')
    axes[1, 0].set_title('Jaw Classification Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Teeth Precision/Recall
    axes[1, 1].plot(epochs, history['train_missing_precision'], 'b--', label='Train Precision')
    axes[1, 1].plot(epochs, history['val_missing_precision'], 'r--', label='Val Precision')
    axes[1, 1].plot(epochs, history['train_missing_recall'], 'b:', label='Train Recall')
    axes[1, 1].plot(epochs, history['val_missing_recall'], 'r:', label='Val Recall')
    axes[1, 1].set_title('Missing Precision and Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Combined Metrics
    axes[1, 2].plot(epochs, history['val_macro_f1'], 'm-', label='Macro F1', linewidth=2)
    axes[1, 2].plot(epochs, history['val_missing_f1'], 'r--', label='Missing F1', linewidth=2)
    axes[1, 2].plot(epochs, history['val_pr_auc_macro'], 'c-', label='PR-AUC Macro', linewidth=2)
    axes[1, 2].plot(epochs, history['val_pr_auc_micro'], 'k--', label='PR-AUC Micro', linewidth=2)
    axes[1, 2].plot(epochs, history['val_jaw_acc'], 'g-', label='Jaw Acc', linewidth=2)
    axes[1, 2].set_title('Validation: Macro F1 vs PR-AUC')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n Training plots saved to: {save_path}")


# ==================== Custom Collate ====================
def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    jaw_types = [item[2] for item in batch]
    return imgs, labels, jaw_types


# ==================== MAIN ====================
def main():
    global available_gpus, device
    set_seed(SEED)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(" "*10 + "2D TRAINING (16+1 Neurons - ResNet18 + Dynamic Loss)")
    print("="*80)
    
    print("\n[0/5] Detecting free GPUs...")
    available_gpus = get_free_gpus()
    device = torch.device(f"cuda:{available_gpus[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Primary device: {device}")

    # Load datasets
    csv_paths = []
    if USE_AUGMENTED and Path(AUGMENTED_CSV).exists():
        csv_paths.append(AUGMENTED_CSV)
        print(f" Loading augmented CSV: {AUGMENTED_CSV}")
    elif USE_AUGMENTED:
        print(f"  Augmented CSV not found: {AUGMENTED_CSV}")
    
    if USE_RANDOM and Path(RANDOM_CSV).exists():
        csv_paths.append(RANDOM_CSV)
        print(f" Loading random CSV: {RANDOM_CSV}")
    elif USE_RANDOM:
        print(f"  Random CSV not found: {RANDOM_CSV}")
    
    if not csv_paths and not USE_ORIGINAL:
        print(" ERROR: No datasets enabled!")
        return

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n[1/5] Building dataset...")
    full_dataset = AugmentedToothDataset16Plus1(
        csv_paths=csv_paths,
        render_root=[RENDER_ROOT, RENDER_ROOT_TEST],
        transform=None
    )
    
    # Add original samples if enabled
    original_samples = []
    if USE_ORIGINAL:
        print(" Loading original dataset...")
        original_samples = build_original_samples(
            img_roots=[IMG_ROOT_LOWER, IMG_ROOT_UPPER],
            json_roots=[JSON_ROOT_LOWER, JSON_ROOT_UPPER]
        )
        if original_samples:
            full_dataset.samples.extend(original_samples)
            print(f" Added {len(original_samples)} original samples")
    
    print(f"[Info] Total loaded: {len(full_dataset.samples)} 2D samples")
    
    if len(full_dataset) == 0:
        print(" ERROR: Dataset is empty!")
        return

    print("\n[1.5/5] Performing train/val split...")
    case_to_indices = {}
    for idx, sample in enumerate(full_dataset.samples):
        case_id = sample['case_id']
        case_to_indices.setdefault(case_id, []).append(idx)
    
    case_ids = sorted(case_to_indices.keys())
    train_cases, val_cases = train_test_split(
        case_ids, test_size=0.2, random_state=SEED
    )
    
    train_indices = [i for cid in train_cases for i in case_to_indices[cid]]
    val_indices = [i for cid in val_cases for i in case_to_indices[cid]]
    
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            sample = self.dataset.samples[self.indices[idx]]
            try:
                img = Image.open(sample['png_path']).convert('RGB')
            except:
                img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
            
            labels = torch.from_numpy(sample['labels']).float()
            if self.transform is not None:
                img = self.transform(img)
            return img, labels, sample['jaw_type']
    
    train_dataset = TransformedSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformedSubset(full_dataset, val_indices, val_transform)
    
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )

    print("\n[2/5] Initializing 2D model (16+1 outputs)...")
    model = ResNetMultiLabel16Plus1(
        backbone=BACKBONE,
        num_outputs=NUM_OUTPUTS,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    if len(available_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)

    criterion = DynamitLoss16Plus1(device)
    print(" Using Dynamic Loss for teeth (0-15) + BCE for jaw (16)")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_missing_f1': [], 'val_missing_f1': [],
        'train_missing_acc': [], 'val_missing_acc': [],
        'train_missing_precision': [], 'val_missing_precision': [],
        'train_missing_recall': [], 'val_missing_recall': [],
        'train_jaw_acc': [], 'val_jaw_acc': [],
        'train_pr_auc_micro': [], 'val_pr_auc_micro': [],
        'train_pr_auc_macro': [], 'val_pr_auc_macro': [],
        'val_macro_f1': []
    }

    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs...")
    
    best_f1 = 0.0
    best_val_preds = None
    best_val_labels = None
    best_val_jaws = None
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_preds, val_labels, val_jaws = validate(model, val_loader, criterion, device)

        if EARLY_STOPPING_METRIC == "val_pr_auc_macro":
            val_score = val_metrics['missing_pr_auc_macro']
        else:
            val_score = val_metrics['macro_f1']

        scheduler.step(val_score)

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_missing_f1'].append(train_metrics['missing_f1'])
        history['val_missing_f1'].append(val_metrics['missing_f1'])
        history['train_missing_acc'].append(train_metrics['missing_accuracy'])
        history['val_missing_acc'].append(val_metrics['missing_accuracy'])
        history['train_missing_precision'].append(train_metrics['missing_precision'])
        history['val_missing_precision'].append(val_metrics['missing_precision'])
        history['train_missing_recall'].append(train_metrics['missing_recall'])
        history['val_missing_recall'].append(val_metrics['missing_recall'])
        history['train_jaw_acc'].append(train_metrics['jaw_accuracy'])
        history['val_jaw_acc'].append(val_metrics['jaw_accuracy'])
        history['train_pr_auc_micro'].append(train_metrics['missing_pr_auc_micro'])
        history['val_pr_auc_micro'].append(val_metrics['missing_pr_auc_micro'])
        history['train_pr_auc_macro'].append(train_metrics['missing_pr_auc_macro'])
        history['val_pr_auc_macro'].append(val_metrics['missing_pr_auc_macro'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
              f"Val PR-AUC(Macro): {val_metrics['missing_pr_auc_macro']:.4f} | "
              f"Val Jaw Acc: {val_metrics['jaw_accuracy']:.4f}")

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict()
        }, Path(OUTPUT_DIR) / LAST_MODEL_FILENAME)
        
        if val_score > best_f1 + MIN_DELTA:
            best_f1 = val_score
            best_val_preds = val_preds
            best_val_labels = val_labels
            best_val_jaws = val_jaws
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict()
            }, Path(OUTPUT_DIR) / BEST_MODEL_FILENAME)
            print(f"        → Best model saved ({EARLY_STOPPING_METRIC}: {val_score:.4f}) to {BEST_MODEL_FILENAME}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break

    print("\n" + "="*80)
    print(f"[4/5] Training complete!")
    print(f"Best validation {EARLY_STOPPING_METRIC}: {best_f1:.4f}")
    print("="*80)

    print("\n[4.5/5] Calculating per-position metrics on best model...")
    best_teeth_thresholds = resolve_teeth_thresholds(best_val_preds, best_val_labels)
    per_position_metrics, macro_metrics, per_jaw_macro = calculate_per_position_metrics(
        best_val_preds,
        best_val_labels,
        best_val_jaws,
        teeth_thresholds=best_teeth_thresholds,
        jaw_threshold=JAW_THRESHOLD
    )
    overall_metrics = calculate_metrics_16plus1(
        best_val_preds, best_val_labels, teeth_thresholds=best_teeth_thresholds, jaw_threshold=JAW_THRESHOLD
    )

    print("\n MACRO-AVERAGED METRICS (per jaw, 16 positions):")
    print("-"*80)
    print(f"  Macro Precision: {macro_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {macro_metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {macro_metrics['macro_f1']:.4f}")
    print(f"  Macro Accuracy:  {macro_metrics['macro_accuracy']:.4f}")
    print(f"  Support Filter:  >= {macro_metrics['macro_support_min']} positives "
          f"({macro_metrics['macro_positions_per_jaw']}/16 positions, "
          f"{macro_metrics['macro_jaws_included']} jaws)")
    
    print("\n JAW CLASSIFICATION METRICS:")
    print("-"*80)
    print(f"  Jaw Accuracy:  {overall_metrics['jaw_accuracy']:.4f}")
    print(f"  Jaw Precision: {overall_metrics['jaw_precision']:.4f}")
    print(f"  Jaw Recall:    {overall_metrics['jaw_recall']:.4f}")
    print(f"  Jaw F1:        {overall_metrics['jaw_f1']:.4f}")

    print("\n PER-POSITION METRICS (FDI Notation, separated by jaw):")
    print("-"*95)
    print(f"{'Jaw':<8} {'FDI':<8} {'Pos':<5} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-"*95)
    for key, metrics in per_position_metrics.items():
        print(f"{metrics['jaw']:<8} {metrics['fdi']:<8} {metrics['position']:<5} "
              f"{metrics['precision']:>10.4f}   {metrics['recall']:>10.4f}   "
              f"{metrics['f1']:>10.4f}   {metrics['accuracy']:>10.4f}   {metrics['support']:>8}")

    # Save metrics
    metrics_file = Path(OUTPUT_DIR) / METRICS_FILENAME
    with open(metrics_file, 'w') as f:
        json.dump({
            'overall_metrics': {k: float(v) for k, v in overall_metrics.items()},
            'macro_metrics': macro_metrics,
            'per_jaw_macro': per_jaw_macro,
            'per_position_metrics': per_position_metrics,
            'best_teeth_thresholds': _expand_teeth_thresholds(best_teeth_thresholds).tolist(),
            'training_config': {
                'loss_function': 'DynamitLoss16Plus1',
                'num_teeth_positions': NUM_TEETH_POSITIONS,
                'num_outputs': NUM_OUTPUTS,
                'use_original': USE_ORIGINAL,
                'use_augmented': USE_AUGMENTED,
                'use_random': USE_RANDOM,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'dropout_rate': DROPOUT_RATE,
                'batch_size': BATCH_SIZE,
                'macro_support_min': MACRO_SUPPORT_MIN,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                'min_delta': MIN_DELTA,
                'early_stopping_metric': EARLY_STOPPING_METRIC,
                'threshold_strategy': THRESHOLD_STRATEGY,
                'fixed_teeth_threshold': FIXED_TEETH_THRESHOLD,
                'jaw_threshold': JAW_THRESHOLD
            }
        }, f, indent=2)
    print(f"\n Metrics saved to: {metrics_file}")

    print(f"\n[5/5] Generating training plots...")
    plot_training_curves(history, PLOT_DIR, PLOT_FILENAME)
    print("\n All done!")


if __name__ == "__main__":
    main()
