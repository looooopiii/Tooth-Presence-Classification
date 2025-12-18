import json
import random
from pathlib import Path
from collections import OrderedDict
import re
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
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

from torchvision.models import resnet18, resnet50
from torchvision import transforms

# =================================================================================
# CONFIGURATION (Augmented Dynamit)
# =================================================================================
TEST_IMG_DIR = Path("/home/user/lzhou/week13-32/output/Render_TestAll")
TEST_LABELS_CSV = Path("/home/user/lzhou/week10/label_flipped.csv")
MODEL_PATH = Path("/home/user/lzhou/week13-32/output/Train2D/aug_dynamit/dynamit_loss_full_dataset_best.pth")
OUTPUT_DIR = Path("/home/user/lzhou/week13-32/output/Test2D/aug_dynamit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 320
NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 5

USE_PCA_ALIGN_2D = False

# [STRATEGY: MAX]
MULTIVIEW_STRATEGY = "max_per_tooth"

# [TOP-K]
MAX_PER_TOOTH_TOPK = 2

# Placeholder, will use Auto-Search
DEFAULT_THRESHOLD = 0.95

# --- FDI Notation Mapping ---
VALID_FDI_LABELS = sorted(
    [
        18, 17, 16, 15, 14, 13, 12, 11,
        21, 22, 23, 24, 25, 26, 27, 28,
        38, 37, 36, 35, 34, 33, 32, 31,
        41, 42, 43, 44, 45, 46, 47, 48,
    ]
)
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}

UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
UPPER_IDX = [FDI_TO_INDEX[f] for f in UPPER_FDI]
LOWER_IDX = [FDI_TO_INDEX[f] for f in LOWER_FDI]


# =================================================================================
# DEVICE SELECTION
# =================================================================================
def get_free_gpus(threshold_mb: int = 1000, max_gpus: int = 2):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        gpus = []
        for line in lines:
            idx_str, mem_str = [x.strip() for x in line.split(",")]
            if int(mem_str) < threshold_mb:
                gpus.append(int(idx_str))
            if len(gpus) >= max_gpus:
                break
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
        print(f"[GPU] Using cuda:{usable[0]}, candidates: {usable}")
        return torch.device(f"cuda:{usable[0]}")
    print("[GPU] Using CPU")
    return torch.device("cpu")


DEVICE = pick_device()


# =================================================================================
# MODEL DEFINITION (Dynamit Specific Structure)
# =================================================================================
class ToothClassificationModel(nn.Module):
    """
    ResNet encoder + 3-layer MLP head.
    Matches the structure in 'aug_dynamit/dynamit_loss_full_dataset_best.pth'.
    """
    def __init__(self, num_teeth: int = 32, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights=None)
        else:
            net = resnet18(weights=None)
        
        feat_dim = net.fc.in_features
        # Extract features (remove original fc)
        self.feature_extractor = nn.Sequential(*list(net.children())[:-1])
        
        # MLP Head
        self.fc1 = nn.Linear(feat_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, num_teeth)

    def forward(self, x):
        features = self.feature_extractor(x).flatten(1)
        x = self.dropout1(F.relu(self.bn1(self.fc1(features))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        logits = self.fc3(x)
        return logits


# =================================================================================
# TRANSFORMS & PCA
# =================================================================================
def build_test_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def pca_align_image_2d(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    ys, xs = np.where(arr > 10)
    if len(xs) < 20:
        return img
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    w, v = np.linalg.eigh(cov)
    idx_max = np.argmax(w)
    pc = v[:, idx_max]
    angle = np.degrees(np.arctan2(pc[1], pc[0]))
    img_rot = img.rotate(-angle, resample=Image.BILINEAR,
                         expand=True, fillcolor=(0, 0, 0))
    return img_rot


def load_image_tensor(img_path: Path, transform, use_pca_align=True):
    img = Image.open(img_path).convert("RGB")
    if use_pca_align:
        img = pca_align_image_2d(img)
    return transform(img).unsqueeze(0)


# =================================================================================
# DATA LOADING / GROUPING
# =================================================================================
def normalize_png_stem_to_newid(stem: str) -> str:
    s = stem.replace("-", "_").strip()
    s = re.sub(r"\s+", " ", s)
    if " " in s:
        left, right = s.rsplit(" ", 1)
    else:
        left, right = s, ""
    jaw = right.strip().lower()
    jaw_key = ""
    m = re.search(r"(upper|lower)jawscan(\d*)", jaw)
    if m:
        jaw_key = m.group(1) + m.group(2)
    else:
        if "upperjawscan" in jaw:
            jaw_key = "upper"
        elif "lowerjawscan" in jaw:
            jaw_key = "lower"
        else:
            lower_s = s.lower()
            if lower_s.endswith("_upper"):
                jaw_key = "upper"
                left = s[:-6]
            elif lower_s.endswith("_lower"):
                jaw_key = "lower"
                left = s[:-6]
    left = left.strip()
    left = left.replace("-", "_")
    left = left.replace(" ", "_")
    new_id = f"{left}_{jaw_key}" if jaw_key else left
    return new_id.lower()


def parse_case_id_and_angle(img_path: Path):
    raw_stem = img_path.stem
    stem = raw_stem.replace("-", "_")
    m = re.search(r"_rot(\d+)$", stem)
    angle_deg = int(m.group(1)) if m else 0
    base_stem = re.sub(r"_rot\d+$", "", stem)
    if base_stem.endswith("_top"):
        base_stem = base_stem[:-4]
    case_id_norm = normalize_png_stem_to_newid(base_stem)
    low_name = stem.lower()
    if "upper" in low_name:
        jaw_type = "upper"
    elif "lower" in low_name:
        jaw_type = "lower"
    else:
        jaw_type = "unknown"
    return case_id_norm, angle_deg, jaw_type


def group_images_by_case(img_dir: Path):
    groups = {}
    for p in img_dir.rglob("*.png"):
        case_id_norm, angle_deg, jaw_type = parse_case_id_and_angle(p)
        groups.setdefault(case_id_norm, []).append((p, angle_deg, jaw_type))
    return groups


def load_test_labels(csv_path: Path):
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    if "new_id" not in cols_lower:
        raise ValueError("CSV must contain a 'new_id' column.")
    id_col = cols_lower["new_id"]
    df[id_col] = df[id_col].astype(str).str.strip().str.lower().str.replace("-", "_")
    df.columns = [str(c) if str(c).isdigit() else c for c in df.columns]
    labels_dict = {}
    for _, row in df.iterrows():
        case_key = str(row[id_col]).strip().lower()
        label_vector = np.zeros(NUM_TEETH, dtype=np.float32)
        for tooth_fdi in VALID_FDI_LABELS:
            key = str(tooth_fdi)
            if key in df.columns and pd.notna(row[key]) and int(row[key]) == 1:
                label_vector[FDI_TO_INDEX[tooth_fdi]] = 1.0
        labels_dict[case_key] = label_vector
    return labels_dict


# =================================================================================
# MULTI-VIEW INFERENCE
# =================================================================================
def compute_jaw_confidence_from_probs(probs_tensor: torch.Tensor, jaw_type: str) -> float:
    if probs_tensor.ndim == 2:
        probs = probs_tensor[0]
    else:
        probs = probs_tensor
    if jaw_type == "upper":
        jaw_probs = probs[UPPER_IDX]
    elif jaw_type == "lower":
        jaw_probs = probs[LOWER_IDX]
    else:
        jaw_probs = probs
    conf = torch.mean(torch.abs(jaw_probs - 0.5)).item()
    return conf


def test_model_multiview(model, grouped_imgs, labels_dict, device, transform):
    model.eval()
    csv_ids = set(labels_dict.keys())
    img_ids = set(grouped_imgs.keys())
    common_ids = sorted(csv_ids & img_ids)
    
    print(f"# Intersection (test): {len(common_ids)}")
    if len(common_ids) == 0:
        return None, None, None, None, None

    all_best_preds = []
    all_targets = []
    all_ids = []
    all_best_angles = []
    all_jaw_confs = []

    for case_id in common_ids:
        gt_labels = labels_dict[case_id]
        views = grouped_imgs[case_id]
        if len(views) == 0:
            continue

        angle_list = []
        jaw_conf_list = []
        probs_list = []

        for img_path, angle_deg, jaw_type in views:
            x = load_image_tensor(img_path, transform, use_pca_align=USE_PCA_ALIGN_2D).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)

            angle_list.append(angle_deg)
            jaw_conf_list.append(compute_jaw_confidence_from_probs(probs, jaw_type))
            probs_list.append(probs.cpu().numpy()[0])

        probs_arr = np.stack(probs_list, axis=0)
        jaw_conf_arr = np.array(jaw_conf_list)

        # --- MAX PER TOOTH STRATEGY ---
        if MULTIVIEW_STRATEGY == "max_per_tooth":
            K = min(MAX_PER_TOOTH_TOPK, len(views))
            topk_idx = np.argsort(jaw_conf_arr)[-K:]
            selected_probs = probs_arr[topk_idx]
            
            final_probs = selected_probs.max(axis=0)

            local_jaw_conf = jaw_conf_arr[topk_idx]
            best_local_idx = int(topk_idx[np.argmax(local_jaw_conf)])
            final_angle = angle_list[best_local_idx]
            final_conf = jaw_conf_arr[best_local_idx]
            
        elif MULTIVIEW_STRATEGY == "avg_all":
            final_probs = probs_arr.mean(axis=0)
            best_idx = int(np.argmax(jaw_conf_arr))
            final_angle = angle_list[best_idx]
            final_conf = jaw_conf_arr[best_idx]
        elif MULTIVIEW_STRATEGY == "jaw_conf":
            best_idx = int(np.argmax(jaw_conf_arr))
            final_probs = probs_arr[best_idx]
            final_angle = angle_list[best_idx]
            final_conf = jaw_conf_arr[best_idx]
        else:
            raise ValueError(f"Unknown Strategy: {MULTIVIEW_STRATEGY}")

        all_best_preds.append(final_probs)
        all_targets.append(gt_labels)
        all_ids.append(case_id)
        all_best_angles.append(final_angle)
        all_jaw_confs.append(final_conf)

    return (
        np.array(all_best_preds),
        np.array(all_targets),
        all_ids,
        all_best_angles,
        all_jaw_confs,
    )


# =================================================================================
# AUTO THRESHOLD SEARCH
# =================================================================================
def find_optimal_threshold(preds, targets):
    """
    Search range 0.80 -> 0.995 to find max F1.
    """
    thresholds = np.arange(0.80, 0.995, 0.01)
    best_th = 0.95
    best_f1 = 0.0
    
    print("\n" + "="*50)
    print(" AUTO-THRESHOLD SEARCH (Maximizing F1)")
    print("="*50)
    print(f"{'Threshold':<10} {'Micro F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 45)
    
    flat_targets = targets.flatten().astype(int)
    
    for th in thresholds:
        flat_preds = (preds > th).astype(int).flatten()
        p, r, f1, _ = precision_recall_fscore_support(flat_targets, flat_preds, average='binary', zero_division=0)
        print(f"{th:<10.2f} {f1:<10.4f} {p:<10.4f} {r:<10.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    print("-" * 45)
    print(f"Best Threshold Found: {best_th:.3f} (F1: {best_f1:.4f})")
    print("="*50 + "\n")
    return best_th


# =================================================================================
# METRICS & PLOTTING
# =================================================================================
def calculate_metrics(preds, targets, threshold):
    preds_bin = (preds > threshold).astype(int)
    targets_bin = targets.astype(int)
    flat_preds, flat_targets = preds_bin.flatten(), targets_bin.flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_targets, flat_preds, average="binary", zero_division=0
    )
    acc = accuracy_score(flat_targets, flat_preds)

    per_tooth = OrderedDict()
    for idx in range(NUM_TEETH):
        fdi_label = INDEX_TO_FDI[idx]
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(
            targets_bin[:, idx], preds_bin[:, idx],
            average="binary", zero_division=0
        )
        acc_t = accuracy_score(targets_bin[:, idx], preds_bin[:, idx])
        per_tooth[fdi_label] = {
            "precision": float(p_t),
            "recall": float(r_t),
            "f1": float(f1_t),
            "accuracy": float(acc_t),
            "support": int(targets_bin[:, idx].sum()),
        }

    macro_precision = np.mean([m["precision"] for m in per_tooth.values()])
    macro_recall = np.mean([m["recall"] for m in per_tooth.values()])
    macro_f1 = np.mean([m["f1"] for m in per_tooth.values()])
    macro_acc = np.mean([m["accuracy"] for m in per_tooth.values()])

    return {
        "overall_micro": {
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "accuracy": float(acc),
        },
        "overall_macro": {
            "macro_precision": float(macro_precision), "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1), "macro_accuracy": float(macro_acc),
        },
        "per_tooth": per_tooth,
    }


def print_metrics_summary(metrics):
    micro = metrics["overall_micro"]
    print("\n OVERALL (MICRO-AVERAGE) METRICS:")
    print(f"  - Precision: {micro['precision']:.4f}\n"
          f"  - Recall:    {micro['recall']:.4f}\n"
          f"  - F1 Score:  {micro['f1']:.4f}\n"
          f"  - Accuracy:  {micro['accuracy']:.4f}")
    
    print("\n PER-TOOTH METRICS:")
    print("-" * 80)
    print(f"{'FDI':<6} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Acc':<10} {'Supp':<8}")
    print("-" * 80)
    for fdi, m in metrics["per_tooth"].items():
        print(f"{fdi:<6} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['accuracy']:>10.4f} {m['support']:>8}")
    print("=" * 80)


def print_sample_predictions(ids, preds, targets, num_samples, threshold):
    print(f"\n SAMPLE PREDICTIONS (Threshold={threshold:.3f})")
    print("=" * 80)
    if len(ids) == 0: return
    num_samples = min(num_samples, len(ids))
    indices = random.sample(range(len(ids)), num_samples)
    for i in indices:
        case_id = ids[i]
        t_vec = targets[i]
        p_vec = (preds[i] > threshold).astype(int)
        
        truth = {INDEX_TO_FDI[j] for j, v in enumerate(t_vec) if v==1}
        pred  = {INDEX_TO_FDI[j] for j, v in enumerate(p_vec) if v==1}
        
        tp = sorted(list(truth & pred))
        fn = sorted(list(truth - pred))
        fp = sorted(list(pred - truth))
        
        print(f"\n Case: {case_id}")
        print(f"  Truth: {sorted(list(truth)) if truth else 'None'}")
        print(f"  Pred:  {sorted(list(pred)) if pred else 'None'}")
        print(f"   TP: {tp}, FN: {fn}, FP: {fp}")
    print("\n" + "=" * 80)


def generate_test_plots(metrics, preds, targets, save_dir: Path, threshold):
    per_tooth = metrics["per_tooth"]
    labels = [str(k) for k in per_tooth.keys()]
    prec = [m["precision"] for m in per_tooth.values()]
    rec  = [m["recall"] for m in per_tooth.values()]
    f1   = [m["f1"] for m in per_tooth.values()]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, prec, width, label="Precision", color="royalblue")
    ax.bar(x, rec, width, label="Recall", color="limegreen")
    rects = ax.bar(x + width, f1, width, label="F1", color="tomato")
    ax.set_title(f"Per-Tooth Performance (Thresh={threshold:.3f})", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.bar_label(rects, padding=3, fmt="%.2f", fontsize=8)
    plt.savefig(save_dir / "per_tooth_metrics.png", dpi=300)
    plt.close()

    flat_t = targets.flatten()
    flat_p = (preds > threshold).astype(int).flatten()
    cm = confusion_matrix(flat_t, flat_p)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Thresh={threshold:.3f})")
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    p_curve, r_curve, _ = precision_recall_curve(flat_t, preds.flatten())
    plt.figure(figsize=(10,7))
    plt.plot(r_curve, p_curve, color='darkorange', lw=2)
    plt.title("PR Curve")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.savefig(save_dir / "precision_recall_curve.png", dpi=300)
    plt.close()


def export_results(ids, preds, targets, best_angles, jaw_confs, save_dir: Path, metrics, threshold):
    suffix = "max"
    df = pd.DataFrame({"case_id": ids})
    df["best_angle"] = best_angles
    df["jaw_conf"] = jaw_confs
    for fdi in VALID_FDI_LABELS:
        idx = FDI_TO_INDEX[fdi]
        df[f"true_{fdi}"] = targets[:, idx]
        df[f"pred_{fdi}"] = (preds[:, idx] > threshold).astype(int)
        df[f"prob_{fdi}"] = preds[:, idx]
        
    csv_path = save_dir / f"test_predictions_aug_dynamit_{suffix}_th{threshold:.3f}.csv"
    df.to_csv(csv_path, index=False)
    
    json_path = save_dir / f"test_metrics_aug_dynamit_{suffix}_th{threshold:.3f}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f" Saved to {csv_path}")


# =================================================================================
# MAIN
# =================================================================================
def main():
    print(f"Using device: {DEVICE}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    # Load Model (Dynamit Structure)
    model = ToothClassificationModel(num_teeth=NUM_TEETH).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if state_dict and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f" Model loaded: {MODEL_PATH}")

    # Load Data
    labels_dict = load_test_labels(TEST_LABELS_CSV)
    grouped_imgs = group_images_by_case(TEST_IMG_DIR)
    print(f" Loaded {len(labels_dict)} labels, found images for {len(grouped_imgs)} cases.")

    transform = build_test_transform()

    # Inference (Multi-View)
    preds, targets, ids, best_angles, jaw_confs = test_model_multiview(
        model, grouped_imgs, labels_dict, DEVICE, transform
    )
    if preds is None: return

    print(f" Inference done on {len(ids)} cases.")

    # Auto Threshold
    best_th = find_optimal_threshold(preds, targets)

    # Metrics & Report
    metrics = calculate_metrics(preds, targets, threshold=best_th)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, num_samples=NUM_SAMPLE_PREDICTIONS, threshold=best_th)
    
    # Save
    generate_test_plots(metrics, preds, targets, OUTPUT_DIR, threshold=best_th)
    export_results(ids, preds, targets, best_angles, jaw_confs, OUTPUT_DIR, metrics, threshold=best_th)
    
    print(f"\n Done. Results in {OUTPUT_DIR}")
    print(f" Strategy: {MULTIVIEW_STRATEGY}, Threshold: {best_th:.3f}")

if __name__ == "__main__":
    main()