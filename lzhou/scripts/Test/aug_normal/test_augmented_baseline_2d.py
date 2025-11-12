import json
import random
from pathlib import Path
from collections import OrderedDict

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


# =================================================================================
# CONFIGURATION (2D)
# =================================================================================
TEST_IMG_DIR = Path("/home/user/lzhou/week9/Clean_Test_Data")
TEST_LABELS_CSV = Path("/home/user/lzhou/week10/label_flipped.csv")
MODEL_PATH = Path("/home/user/lzhou/week10/output/Train2D/aug_normal/bce_loss_full_dataset_best.pth")
OUTPUT_DIR = Path("/home/user/lzhou/week10/output/Test2D/aug_normal")
# ------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 320
NUM_TEETH = 32
NUM_SAMPLE_PREDICTIONS = 5
BATCH_SIZE = 16
NUM_WORKERS = 0

# --- FDI Notation Mapping (must match training) ---
VALID_FDI_LABELS = sorted(
    [
        18,
        17,
        16,
        15,
        14,
        13,
        12,
        11,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        38,
        37,
        36,
        35,
        34,
        33,
        32,
        31,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
    ]
)
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi_label for fdi_label, i in FDI_TO_INDEX.items()}


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
from torchvision.models import resnet18, resnet50


class ToothClassificationModel(nn.Module):
    """ResNet encoder + MLP head, identical to training."""

    def __init__(self, num_teeth: int = 32, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights=None)
        else:
            net = resnet18(weights=None)
        feat_dim = net.fc.in_features
        self.feature_extractor = nn.Sequential(*list(net.children())[:-1])
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
        return self.fc3(x)


# =================================================================================
# DATASET / TRANSFORMS
# =================================================================================
from torchvision import transforms


def build_test_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class Tooth2DTestDataset(Dataset):
    """
    Dataset that pairs *_top.png images with 32-dim label vectors (1=missing, 0=present).
    """

    def __init__(self, image_paths, labels_dict):
        self.image_paths = image_paths
        self.labels_dict = labels_dict
        self.transform = build_test_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        case_id = img_path.stem
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        tensor = self.transform(image)
        labels = self.labels_dict[case_id]
        return tensor, torch.from_numpy(labels).float(), case_id


# =================================================================================
# LABEL LOADING / SAMPLE PREPARATION
# =================================================================================
def load_test_labels(csv_path: Path):
    df = pd.read_csv(csv_path)
    id_columns = []
    for candidate in ("filename", "case_id", "new_id"):
        if candidate in df.columns:
            id_columns.append(candidate)
    if not id_columns:
        raise ValueError("CSV must contain one of ['filename', 'case_id', 'new_id'] columns to identify samples.")

    def build_id_variants(raw_id: str):
        """Return a set of aliases for a given identifier to maximise matching chance."""
        if not isinstance(raw_id, str):
            return set()
        base = raw_id.strip()
        if not base:
            return set()
        if base.lower() == "nan":
            return set()
        variants = {base}
        # Preserve original spacing/hyphenation but also offer a compact form.
        compact = base.replace(" ", "_")
        variants.add(compact)
        # Ensure we account for aliases with or without the _top suffix.
        expanded = set()
        for candidate in variants:
            if candidate.endswith("_top"):
                expanded.add(candidate)
                expanded.add(candidate[:-4])
            else:
                expanded.add(candidate)
                expanded.add(f"{candidate}_top")
        return expanded

    labels_dict = {}
    missing_cols = [str(fdi) for fdi in VALID_FDI_LABELS if str(fdi) not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing tooth label columns: {missing_cols}")

    for idx, row in df.iterrows():
        label_vec = np.zeros(NUM_TEETH, dtype=np.float32)
        for fdi in VALID_FDI_LABELS:
            val = row[str(fdi)]
            if pd.notna(val) and int(val) == 1:
                label_vec[FDI_TO_INDEX[fdi]] = 1.0
        id_variants = set()
        for col in id_columns:
            raw_id = row[col]
            variants = build_id_variants(str(raw_id))
            id_variants.update(variants)
        for key in id_variants:
            labels_dict[key] = label_vec.copy()
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
        case_id = path.stem
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
            probs = torch.sigmoid(logits).cpu().numpy() # 模型输出logits, sigmoid转为概率
            preds.append(probs)
            targets.append(labels.numpy())
            ids.extend(case_ids)
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), ids


def calculate_metrics(preds, targets):
    preds_bin = (preds > 0.5).astype(int)
    targets_bin = targets.astype(int)
    flat_preds, flat_targets = preds_bin.flatten(), targets_bin.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(flat_targets, flat_preds, average="binary", zero_division=0)
    acc = accuracy_score(flat_targets, flat_preds)

    per_tooth = OrderedDict()
    for idx in range(NUM_TEETH):
        fdi_label = INDEX_TO_FDI[idx]
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(targets_bin[:, idx], preds_bin[:, idx], average="binary", zero_division=0)
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

    metrics = {
        "overall_micro": {"precision": float(precision), "recall": float(recall), "f1": float(f1), "accuracy": float(acc)},
        "overall_macro": {
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "macro_accuracy": float(macro_acc),
        },
        "per_tooth": per_tooth,
    }
    return metrics


# =================================================================================
# REPORTING / VISUALIZATION
# =================================================================================
def print_metrics_summary(metrics):
    print("\n" + "=" * 80)
    print(" " * 28 + "TESTING METRICS SUMMARY")
    print("=" * 80)
    micro = metrics["overall_micro"]
    print("\n OVERALL (MICRO-AVERAGE) METRICS (Positive class: Missing Tooth):")
    print(
        f"  - Precision: {micro['precision']:.4f}\n"
        f"  - Recall:    {micro['recall']:.4f}\n"
        f"  - F1 Score:  {micro['f1']:.4f}\n"
        f"  - Accuracy:  {micro['accuracy']:.4f}"
    )
    macro = metrics["overall_macro"]
    print("\n OVERALL (MACRO-AVERAGE) METRICS:")
    print(
        f"  - Macro Precision: {macro['macro_precision']:.4f}\n"
        f"  - Macro Recall:    {macro['macro_recall']:.4f}\n"
        f"  - Macro F1 Score:  {macro['macro_f1']:.4f}\n"
        f"  - Macro Accuracy:  {macro['macro_accuracy']:.4f}"
    )
    print("\n PER-TOOTH METRICS (FDI Notation):")
    print("-" * 80)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    for fdi_label, m in metrics["per_tooth"].items():
        print(
            f"Tooth {fdi_label:<5} "
            f"{m['precision']:>10.4f}   {m['recall']:>10.4f}   "
            f"{m['f1']:>10.4f}   {m['accuracy']:>10.4f}   {m['support']:>8}"
        )
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
        target_labels = targets[i]
        pred_labels = (preds[i] > 0.5).astype(int)
        truth_missing = {INDEX_TO_FDI[j] for j, label in enumerate(target_labels) if label == 1}
        pred_missing = {INDEX_TO_FDI[j] for j, label in enumerate(pred_labels) if label == 1}
        correctly_found = sorted(list(truth_missing.intersection(pred_missing)))
        missed = sorted(list(truth_missing.difference(pred_missing)))
        false_alarms = sorted(list(pred_missing.difference(truth_missing)))

        print(f"\n Case ID: {case_id}\n" + "-" * 50)
        print(f"  Ground Truth (Missing): {sorted(list(truth_missing)) if truth_missing else 'None'}")
        print(f"  Prediction (Missing):   {sorted(list(pred_missing)) if pred_missing else 'None'}")
        print("-" * 50)
        print(f"   Correctly Found (TP): {correctly_found or 'None'}")
        print(f"   Missed Teeth (FN):    {missed or 'None'}")
        print(f"   False Alarms (FP):    {false_alarms or 'None'}")
    print("\n" + "=" * 80)


def generate_test_plots(metrics, preds, targets, save_dir: Path):
    per_tooth_metrics = metrics["per_tooth"]
    fdi_labels = [str(label) for label in per_tooth_metrics.keys()]
    precision_scores = [m["precision"] for m in per_tooth_metrics.values()]
    recall_scores = [m["recall"] for m in per_tooth_metrics.values()]
    f1_scores = [m["f1"] for m in per_tooth_metrics.values()]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(fdi_labels))
    width = 0.25
    ax.bar(x - width, precision_scores, width, label="Precision", color="royalblue")
    ax.bar(x, recall_scores, width, label="Recall", color="limegreen")
    rects3 = ax.bar(x + width, f1_scores, width, label="F1 Score", color="tomato")
    ax.set_ylabel("Scores", fontsize=14)
    ax.set_title("Per-Tooth Performance (Positive Class: Missing)", fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(fdi_labels, rotation=45, ha="right")
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.bar_label(rects3, padding=3, fmt="%.2f", fontsize=8)
    fig.tight_layout()
    plot_path = save_dir / "per_tooth_metrics.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Per-tooth metrics plot saved to {plot_path}")

    flat_targets = targets.flatten()
    flat_preds = (preds > 0.5).astype(int).flatten()
    cm = confusion_matrix(flat_targets, flat_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Present (0)", "Pred Missing (1)"],
        yticklabels=["Actual Present (0)", "Actual Missing (1)"],
    )
    plt.title("Overall Confusion Matrix (All Teeth)", fontsize=16, fontweight="bold")
    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plot_path = save_dir / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Confusion matrix plot saved to {plot_path}")

    precision, recall, _ = precision_recall_curve(flat_targets, preds.flatten())
    avg_precision = average_precision_score(flat_targets, preds.flatten())
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR Curve (Avg Precision = {avg_precision:.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall Curve", fontsize=16, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.5)
    plot_path = save_dir / "precision_recall_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Precision-Recall curve saved to {plot_path}")


# =================================================================================
# RESULTS SAVING
# =================================================================================
def export_results(ids, preds, targets, save_dir: Path, metrics):
    results_df = pd.DataFrame({"case_id": ids})
    for fdi_label in VALID_FDI_LABELS:
        idx = FDI_TO_INDEX[fdi_label]
        results_df[f"true_{fdi_label}"] = targets[:, idx]
        results_df[f"pred_{fdi_label}"] = (preds[:, idx] > 0.5).astype(int)
        results_df[f"prob_{fdi_label}"] = preds[:, idx]
    
    # --- Save as CSV and JSON ---
    csv_path = save_dir / "test_predictions_bce.csv"
    results_df.to_csv(csv_path, index=False)
    json_path = save_dir / "test_metrics_bce.json"
    # ----------------------
    
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f" Saved predictions to {csv_path}")
    print(f" Saved metrics to {json_path}")


# =================================================================================
# MAIN
# =================================================================================
def main():
    print(f"Using device: {DEVICE}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    model = ToothClassificationModel(num_teeth=NUM_TEETH).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if state_dict and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f" Model loaded from {MODEL_PATH}")

    labels_dict = load_test_labels(TEST_LABELS_CSV)
    print(f" Loaded labels for {len(labels_dict)} IDs from {TEST_LABELS_CSV}")

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

    preds, targets, ids = run_inference(model, dataloader, DEVICE)
    print(f" Inference complete on {len(ids)} samples.")

    metrics = calculate_metrics(preds, targets)
    print_metrics_summary(metrics)
    print_sample_predictions(ids, preds, targets, num_samples=NUM_SAMPLE_PREDICTIONS)

    print("\n" + "=" * 80)
    print(" " * 28 + "GENERATING PLOTS")
    print("=" * 80)
    generate_test_plots(metrics, preds, targets, OUTPUT_DIR)

    export_results(ids, preds, targets, OUTPUT_DIR, metrics)
    print("\n Full results and metrics saved successfully to", OUTPUT_DIR)


if __name__ == "__main__":
    main()