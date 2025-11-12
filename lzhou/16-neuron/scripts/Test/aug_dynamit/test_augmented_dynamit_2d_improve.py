import json
import random
from pathlib import Path
from collections import OrderedDict
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from torchvision import transforms, models


# =================================================================================
# CONFIGURATION (2D)
# =================================================================================
TEST_IMG_DIR = Path("/home/user/lzhou/week9/Clean_Test_Data")
TEST_LABELS_CSV = Path("/home/user/lzhou/week10/label_flipped.csv")
MODEL_PATH = Path("/home/user/lzhou/week11/output/Train2D/aug_dynamit_17out/dynamit_loss_full_dataset_best_17out.pth")
OUTPUT_DIR = Path("/home/user/lzhou/week11/output/Test2D/aug_dynamit_17out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 320
# NUM_TEETH = 32
NUM_OUTPUTS = 17
NUM_TEETH_PER_JAW = 16
NUM_SAMPLE_PREDICTIONS = 5
BATCH_SIZE = 16
NUM_WORKERS = 0

# need both 32-dim (for CSV) and 16-dim (for model) maps ---
# 32-dim map for reading CSV
VALID_FDI_LABELS_32 = sorted(
    [
        18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
        38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48,
    ]
)
FDI_TO_INDEX_32 = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS_32)}
INDEX_TO_FDI_32 = {i: fdi_label for fdi_label, i in FDI_TO_INDEX_32.items()}

# 16-dim maps for 17-output model
UPPER_FDI_TO_IDX16 = {
    18: 0, 17: 1, 16: 2, 15: 3, 14: 4, 13: 5, 12: 6, 11: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15
}
LOWER_FDI_TO_IDX16 = {
    48: 0, 47: 1, 46: 2, 45: 3, 44: 4, 43: 5, 42: 6, 41: 7,
    31: 8, 32: 9, 33: 10, 34: 11, 35: 12, 36: 13, 37: 14, 38: 15
}
# Reverse maps for reporting
IDX16_TO_UPPER_FDI = {v: k for k, v in UPPER_FDI_TO_IDX16.items()}
IDX16_TO_LOWER_FDI = {v: k for k, v in LOWER_FDI_TO_IDX16.items()}


# =================================================================================
# DEVICE SELECTION
# =================================================================================
def get_free_gpus(threshold_mb: int = 1000, max_gpus: int = 2):
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        gpus = []
        for line in lines:
            idx_str, mem_str = line.split(", ")
            if int(mem_str) < threshold_mb:
                gpus.append(int(idx_str))
        if len(gpus) > max_gpus:
            gpus = gpus[:max_gpus]
        if not gpus:
            return [0]
        return gpus
    except Exception:
        return [0]

def filter_visible_gpus(candidate_ids):
    if not torch.cuda.is_available():
        return []
    visible = torch.cuda.device_count()
    if visible == 0:
        return []
    filtered = [idx for idx in candidate_ids if 0 <= idx < visible]
    if not filtered:
        filtered = [0]
    return filtered

def pick_device():
    detected = get_free_gpus()
    usable = filter_visible_gpus(detected)
    if torch.cuda.is_available() and usable:
        return torch.device(f"cuda:{usable[0]}")
    return torch.device("cpu")

DEVICE = pick_device()


# =================================================================================
# MODEL DEFINITION (match training)
# =================================================================================
class ToothClassificationModel(nn.Module):
    """ResNet encoder + MLP head, identical to training."""
    def __init__(self, num_outputs: int = 17, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet50":
            net = models.resnet50(weights=None)
        else:
            net = models.resnet18(weights=None)
        feat_dim = net.fc.in_features
        self.feature_extractor = nn.Sequential(*list(net.children())[:-1])
        self.fc1 = nn.Linear(feat_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_outputs)

    def forward(self, x):
        features = self.feature_extractor(x).flatten(1)
        x = self.dropout1(F.relu(self.bn1(self.fc1(features))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


# =================================================================================
# DATASET / TRANSFORMS
# =================================================================================
def build_test_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

# Tooth2DTestDataset for 17-output model
class Tooth2DTestDataset(Dataset):
    """
    Dataset that pairs *_top.png images with 17-dim label vectors.
    The labels_dict is pre-populated with 17-dim vectors by load_test_labels.
    """
    def __init__(self, image_paths, labels_dict):
        self.image_paths = image_paths
        self.labels_dict = labels_dict
        self.transform = build_test_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # use stem (e.g., '..._lower_top') as key
        case_id = img_path.stem 
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        tensor = self.transform(image)
        # labels_dict now directly contains 17-dim labels
        labels_17 = self.labels_dict[case_id]
        return tensor, torch.from_numpy(labels_17).float(), case_id


# =================================================================================
# LABEL LOADING / SAMPLE PREPARATION
# =================================================================================

# load_test_labels for 17-output model
def load_test_labels(csv_path: Path):
    df = pd.read_csv(csv_path)
    id_columns = []
    for candidate in ("filename", "case_id", "new_id"):
        if candidate in df.columns:
            id_columns.append(candidate)
    if not id_columns:
        raise ValueError("CSV must contain one of ['filename', 'case_id', 'new_id'] columns to identify samples.")

    def build_id_variants(raw_id: str):
        if not isinstance(raw_id, str): return set()
        base = raw_id.strip()
        if not base or base.lower() == "nan": return set()
        variants = {base}
        compact = base.replace(" ", "_")
        variants.add(compact)
        expanded = set()
        for candidate in variants:
            if candidate.endswith("_top"):
                expanded.add(candidate)
                expanded.add(candidate[:-4])
            else:
                expanded.add(candidate)
                expanded.add(f"{candidate}_top")
        return expanded

    labels_dict = {} # save [key] -> 17-dim-label
    missing_cols = [str(fdi) for fdi in VALID_FDI_LABELS_32 if str(fdi) not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing tooth label columns: {missing_cols}")

    for idx, row in df.iterrows():
        # make 17-dim label vector for this row
        id_variants = set()
        primary_id_key = "" # first non-empty ID found
        for col in id_columns:
            raw_id = row[col]
            variants = build_id_variants(str(raw_id))
            id_variants.update(variants)
            if not primary_id_key and str(raw_id).strip():
                primary_id_key = str(raw_id).strip().lower()
        
        if not id_variants:
            continue

        # determine jaw from primary_id_key
        jaw = None
        mapping = None
        jaw_label = 0.0
        # use a clear key for jaw checking
        key_for_jaw_check = primary_id_key.replace(" ", "_")
        
        if "_lower" in key_for_jaw_check:
            jaw = "lower"
            mapping = LOWER_FDI_TO_IDX16
            jaw_label = 1.0
        elif "_upper" in key_for_jaw_check:
            jaw = "upper"
            mapping = UPPER_FDI_TO_IDX16
            jaw_label = 0.0
        else:
            # print(f"Warning: Could not determine jaw for ID {primary_id_key}. Skipping row.")
            continue # skip rows without clear jaw info

        # initialize 17-dim label vector
        label_vec_17 = np.zeros(NUM_OUTPUTS, dtype=np.float32)

        # fill teeth (1=missing, 0=present)
        for fdi_label in mapping.keys(): # only 16 teeth for this jaw
            val = row[str(fdi_label)]
            if pd.notna(val) and int(val) == 1: # CSV 1 = missing
                idx_16 = mapping[fdi_label]
                label_vec_17[idx_16] = 1.0

        # fill jaw label
        label_vec_17[NUM_TEETH_PER_JAW] = jaw_label
        
        # assign to all variants
        for key in id_variants:
            labels_dict[key] = label_vec_17.copy()
            
    return labels_dict


def discover_image_paths(img_root: Path):
    if not img_root.exists():
        raise FileNotFoundError(f"Image directory not found: {img_root}")
    pngs = [p for p in img_root.rglob("*.png") if p.name.lower().endswith("_top.png")]
    return sorted(pngs)


def filter_samples_with_labels(image_paths, labels_dict):
    samples = []
    missing = []
    for path in image_paths:
        case_id = path.stem # e.g., '..._lower_top'
        if case_id in labels_dict:
            samples.append(path)
        else:
            missing.append(case_id)
    if missing:
        print(f"[WARN] {len(missing)} PNGs skipped (no labels found). Example: {missing[:3]}")
    return samples


# =================================================================================
# INFERENCE / METRICS
# =================================================================================
def run_inference(model, dataloader, device):
    model.eval()
    preds, targets, ids = [], [], []
    with torch.no_grad():
        for images, labels, case_ids in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
            targets.append(labels.numpy())
            ids.extend(case_ids)
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), ids


# Helper function for teeth metrics
def calculate_classification_metrics_teeth(pred_teeth, target_teeth):
    pred = (pred_teeth > 0.5).astype(int)
    target = target_teeth.astype(int)
    pred_flat, target_flat = pred.flatten(), target.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(target_flat, pred_flat, average="binary", zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy_score(target_flat, pred_flat)}

# Helper function for jaw metrics
def calculate_jaw_accuracy(pred_jaw, target_jaw):
    pred = (pred_jaw > 0.5).astype(int)
    target = target_jaw.astype(int)
    return {'jaw_accuracy': accuracy_score(target, pred)}

# calculate_metrics for 17-output model
def calculate_metrics(preds, targets):
    """
    Calculates metrics separately for teeth (16) and jaw (1).
    preds/targets shape: [N, 17]
    """
    if len(preds) == 0: return {}

    # Split data
    probs_teeth = preds[:, :NUM_TEETH_PER_JAW]
    probs_jaw = preds[:, NUM_TEETH_PER_JAW]
    
    targets_teeth = targets[:, :NUM_TEETH_PER_JAW]
    targets_jaw = targets[:, NUM_TEETH_PER_JAW]

    # Calculate metrics
    teeth_metrics = calculate_classification_metrics_teeth(probs_teeth, targets_teeth)
    jaw_metrics = calculate_jaw_accuracy(probs_jaw, targets_jaw)

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
# REPORTING / VISUALIZATION
# =================================================================================

def print_metrics_summary(metrics):
    print("\n" + "=" * 80)
    print(" " * 28 + "TESTING METRICS SUMMARY")
    print("=" * 80)
    
    teeth_metrics = metrics['teeth_metrics']
    print("\n OVERALL TEETH METRICS (Micro-Avg, Positive: Missing Tooth):")
    print(
        f"  - Precision: {teeth_metrics['precision']:.4f}\n"
        f"  - Recall:    {teeth_metrics['recall']:.4f}\n"
        f"  - F1 Score:  {teeth_metrics['f1']:.4f}\n"
        f"  - Accuracy:  {teeth_metrics['accuracy']:.4f}"
    )
    
    jaw_metrics = metrics['jaw_metrics']
    print("\n OVERALL JAW METRICS (0=Upper, 1=Lower):")
    print(f"  - Accuracy:  {jaw_metrics['jaw_accuracy']:.4f}")
    
    print("=" * 80)

def print_sample_predictions(ids, preds, targets, num_samples):
    print("\n" + "=" * 80)
    print(" " * 28 + "SAMPLE PREDICTIONS")
    print("=" * 80)
    if len(ids) == 0:
        print("No samples to display.")
        return
    num_samples = min(num_samples, len(ids))
    sample_indices = random.sample(range(len(ids)), num_samples)
    
    for i in sample_indices:
        case_id = ids[i]
        target_labels_17 = targets[i]
        pred_probs_17 = preds[i]

        # JAW PREDICTION
        true_jaw_label = int(target_labels_17[NUM_TEETH_PER_JAW])
        pred_jaw_label = int(pred_probs_17[NUM_TEETH_PER_JAW] > 0.5)
        true_jaw_str = 'Lower' if true_jaw_label == 1 else 'Upper'
        pred_jaw_str = 'Lower' if pred_jaw_label == 1 else 'Upper'
        
        print(f"\n Case ID: {case_id}\n" + "-" * 50)
        print(f"  Ground Truth Jaw: {true_jaw_str}")
        print(f"  Prediction Jaw:   {pred_jaw_str} (Prob: {pred_probs_17[NUM_TEETH_PER_JAW]:.3f})")
        print("-" * 50)

        # TEETH PREDICTION
        true_reverse_map = IDX16_TO_LOWER_FDI if true_jaw_str == 'Lower' else IDX16_TO_UPPER_FDI
        pred_reverse_map = IDX16_TO_LOWER_FDI if pred_jaw_str == 'Lower' else IDX16_TO_UPPER_FDI
        
        target_labels_16 = target_labels_17[:NUM_TEETH_PER_JAW]
        pred_labels_16 = (pred_probs_17[:NUM_TEETH_PER_JAW] > 0.5).astype(int)

        truth_missing = {true_reverse_map[j] for j, label in enumerate(target_labels_16) if label == 1}
        pred_missing = {pred_reverse_map[j] for j, label in enumerate(pred_labels_16) if label == 1}

        correctly_found = sorted(list(truth_missing.intersection(pred_missing)))
        missed = sorted(list(truth_missing.difference(pred_missing)))
        false_alarms = sorted(list(pred_missing.difference(truth_missing)))

        print(f"  Ground Truth (Missing): {sorted(list(truth_missing)) if truth_missing else 'None'}")
        print(f"  Prediction (Missing):   {sorted(list(pred_missing)) if pred_missing else 'None'}")
        print("-" * 50)
        print(f"   Correctly Found (TP): {correctly_found or 'None'}")
        print(f"   Missed Teeth (FN):    {missed or 'None'}")
        print(f"   False Alarms (FP):    {false_alarms or 'None'}")
    print("\n" + "=" * 80)

def generate_test_plots(metrics, save_dir: Path):
    teeth_metrics = metrics['teeth_metrics']
    jaw_metrics = metrics['jaw_metrics']
    
    plt.style.use("seaborn-v0_8-whitegrid")

    # removed per-tooth metrics plot for 17-output model
    # print(f" Per-tooth metrics plot skipped (not applicable for 17-output model)")

    # teeth confusion matrix
    flat_targets_teeth = teeth_metrics['flat_targets']
    flat_preds_teeth = (teeth_metrics['flat_probs'] > 0.5).astype(int)
    cm_teeth = confusion_matrix(flat_targets_teeth, flat_preds_teeth)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_teeth,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Present (0)", "Pred Missing (1)"],
        yticklabels=["Actual Present (0)", "Actual Missing (1)"],
    )
    plt.title("Overall Confusion Matrix (Teeth)", fontsize=16, fontweight="bold")
    plt.ylabel("Ground Truth"); plt.xlabel("Prediction")
    plot_path = save_dir / "confusion_matrix_teeth.png"
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f" Teeth confusion matrix plot saved to {plot_path}")

    # Jaw confusion matrix
    targets_jaw = jaw_metrics['targets']
    preds_jaw = (jaw_metrics['probs'] > 0.5).astype(int)
    cm_jaw = confusion_matrix(targets_jaw, preds_jaw)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_jaw, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Predicted Upper (0)', 'Predicted Lower (1)'],
                yticklabels=['Actual Upper (0)', 'Actual Lower (1)'])
    plt.title('Jaw Classification Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Ground Truth'); plt.xlabel('Prediction')
    plot_path = save_dir / "confusion_matrix_jaw.png"
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f" Jaw confusion matrix plot saved to {plot_path}")

    # Precision-Recall curve for teeth
    precision, recall, _ = precision_recall_curve(flat_targets_teeth, teeth_metrics['flat_probs'])
    avg_precision = average_precision_score(flat_targets_teeth, teeth_metrics['flat_probs'])
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR Curve (Avg Precision = {avg_precision:.2f})")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve (Teeth)", fontsize=16, fontweight="bold")
    plt.legend(loc="lower left"); plt.grid(True, alpha=0.5)
    plot_path = save_dir / "precision_recall_curve_teeth.png"
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f" Precision-Recall curve saved to {plot_path}")


# =================================================================================
# RESULTS SAVING
# =================================================================================
# export_results for 17-output model
def export_results(ids, preds, targets, save_dir: Path, metrics):
    results_df = pd.DataFrame({"case_id": ids})
    
    # save jaw results
    results_df['true_jaw'] = targets[:, NUM_TEETH_PER_JAW].astype(int)
    results_df['pred_jaw'] = (preds[:, NUM_TEETH_PER_JAW] > 0.5).astype(int)
    results_df['prob_jaw'] = preds[:, NUM_TEETH_PER_JAW]

    # save results for 16 teeth (index 0-15)
    for i in range(NUM_TEETH_PER_JAW):
        results_df[f'true_tooth_{i}'] = targets[:, i].astype(int)
        results_df[f'pred_tooth_{i}'] = (preds[:, i] > 0.5).astype(int)
        results_df[f'prob_tooth_{i}'] = preds[:, i]
        
    csv_path = save_dir / "test_predictions_dynamit_17out.csv"
    results_df.to_csv(csv_path, index=False)

    # save serializable metrics
    json_path = save_dir / "test_metrics_dynamit_17out.json"
    serializable_metrics = metrics.copy()
    serializable_metrics['teeth_metrics'].pop('flat_probs', None)
    serializable_metrics['teeth_metrics'].pop('flat_targets', None)
    serializable_metrics['jaw_metrics'].pop('probs', None)
    serializable_metrics['jaw_metrics'].pop('targets', None)
    with open(json_path, "w") as f:
        json.dump(serializable_metrics, f, indent=4)
        
    print(f" Saved predictions to {csv_path}")
    print(f" Saved metrics to {json_path}")


# =================================================================================
# MAIN
# =================================================================================
def main():
    print(f"Using device: {DEVICE}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    # Load model
    model = ToothClassificationModel(num_outputs=NUM_OUTPUTS).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if state_dict and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f" Model loaded from {MODEL_PATH}")

    # Prepare data
    labels_dict = load_test_labels(TEST_LABELS_CSV)
    print(f" Loaded labels for {len(labels_dict)} IDs from {TEST_LABELS_CSV} (converted to 17-dim).")

    image_paths = discover_image_paths(TEST_IMG_DIR)
    image_paths = filter_samples_with_labels(image_paths, labels_dict)
    print(f" Prepared {len(image_paths)} PNG samples for testing.")
    if len(image_paths) == 0:
        print(" No matching samples found. Exiting.")
        return

    dataset = Tooth2DTestDataset(image_paths, labels_dict)
    pin_memory = DEVICE.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # Run inference
    preds, targets, ids = run_inference(model, dataloader, DEVICE)
    print(f" Inference complete on {len(ids)} samples.")

    metrics = calculate_metrics(preds, targets)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, num_samples=NUM_SAMPLE_PREDICTIONS)

    print("\n" + "=" * 80)
    print(" " * 28 + "GENERATING PLOTS")
    print("=" * 80)
    # Generate plots for the 17-output model
    generate_test_plots(metrics, OUTPUT_DIR)

    export_results(ids, preds, targets, OUTPUT_DIR, metrics)
    print("\n Full results and metrics saved successfully to", OUTPUT_DIR)


if __name__ == "__main__":
    main()