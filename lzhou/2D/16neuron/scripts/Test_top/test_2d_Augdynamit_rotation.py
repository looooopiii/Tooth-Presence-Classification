import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from torchvision import transforms
except Exception as e:
    raise RuntimeError("This script requires torchvision. Please install torchvision.") from e


# -------------------------
# FDI ordering (jaw-specific 16)
# -------------------------
UPPER_FDI = [18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28]
LOWER_FDI = [48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38]

# -------------------------
# Default paths
# -------------------------
DEFAULT_IMG_DIR = "/home/user/lzhou/week15/render_output/test_top"
DEFAULT_CSV_PATH = "/home/user/lzhou/week10/label_flipped.csv"
DEFAULT_CHECKPOINT = "/home/user/lzhou/week16-17/output/Train_rotation/Augmented_16plus1teeth_dynamit/augmented_dynamit_best_2d_16plus1teeth.pth"
DEFAULT_OUT_DIR = "/home/user/lzhou/week16-17/output/Test_top/augmented_16plus1teeth_dynamit_auto_best"
DEFAULT_CACHE_IMAGES = True

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


# -------------------------
# Utilities
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="2D test with 24-angle selection"
    )
    p.add_argument("--img_dir", type=str, default=DEFAULT_IMG_DIR,
                   help="Directory containing rendered PNGs (24 per case).")
    p.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH,
                   help="CSV with labels (FDI columns + jaw or inferable).")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                   help="Output directory for results and selected images.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Model checkpoint path (.pth).")

    # Model loading: simplest assumption is checkpoint contains full model via torch.save(model)
    # If not, user can provide python file + class name to construct model and load state_dict.
    p.add_argument("--model_py", type=str,
                   default="/home/user/lzhou/week16-17/scripts/New_test_Aug/score_one_image_16plus1.py",
                   help="Optional: python file containing model class (e.g., /path/model_def.py).")
    p.add_argument("--model_class", type=str, default="ResNetMultiLabel16Plus1",
                   help="Optional: class name to instantiate (e.g., MyNet). Requires --model_py.")
    p.add_argument("--state_key", type=str, default="model_state_dict",
                   help="Optional: key in checkpoint dict for state_dict (e.g., 'model_state_dict'). If None, auto-detect.")

    # Image preprocessing
    p.add_argument("--img_size", type=int, default=256, help="Resize to img_size x img_size before feeding model.")
    p.add_argument("--mean", type=float, nargs=3, default=[0.485,0.456,0.406], help="Normalize mean (3 values).")
    p.add_argument("--std", type=float, nargs=3, default=[0.229,0.224,0.225], help="Normalize std (3 values).")

    # Case grouping from filename
    p.add_argument("--case_regex", type=str,
                   default=r"^(?P<case>.+?)_top_(?:rot-?\d+|rx-?\d+_ry-?\d+_rz-?\d+)\.png$",
                   help="Regex with named group (?P<case>...) to extract case_id from filename.")

    # Jaw source
    p.add_argument("--id_col", type=str, default="new_id", help="Case id column name in CSV.")
    p.add_argument("--jaw_col", type=str, default=None,
                   help="Optional jaw column name in CSV. Accepts values: upper/lower/0/1 or True/False. "
                        "If not provided, tries 'is_lower' then infers.")
    p.add_argument("--present_is_one", action="store_true",
                   help="If set: CSV uses present=1 missing=0, will convert to missing=1 present=0.")

    # Selection + saving
    p.add_argument("--topk", type=int, default=1, help="Save top-k images per case (default 1).")
    p.add_argument("--save_mode", type=str, default="copy", choices=["copy","symlink"],
                   help="How to save selected images.")
    p.add_argument("--min_views", type=int, default=1,
                   help="Minimum number of views required for a case; otherwise case skipped.")

    # Device
    p.add_argument("--device", type=str, default="cuda:1", help="cuda or cpu")
    p.add_argument("--batch_size", type=int, default=1, help="Kept for simplicity; script runs per-image.")
    p.add_argument("--cache_images", action="store_true", help="Cache images in memory to reduce disk I/O.")
    p.add_argument("--no_cache_images", dest="cache_images", action="store_false",
                   help="Disable image caching.")
    p.set_defaults(cache_images=DEFAULT_CACHE_IMAGES)
    return p.parse_args()


def to_case_id_from_filename(fname: str, case_re: re.Pattern) -> Optional[str]:
    name = fname.name if isinstance(fname, Path) else fname
    m = case_re.match(name)
    if not m:
        return None
    return m.group("case")


def normalize_png_stem_to_newid(stem: str) -> str:
    """Cleans up filenames to match CSV IDs."""
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


def infer_is_lower_from_id(case_id: Optional[str]) -> Optional[int]:
    if not case_id:
        return None
    s = str(case_id).strip().lower()
    if "lower" in s:
        return 1
    if "upper" in s:
        return 0
    return None


def normalize_jaw_value(v):
    """Return is_lower in {0,1} or None if cannot parse."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ["lower", "l", "mandible", "mand", "1", "true"]:
            return 1
        if s in ["upper", "u", "maxilla", "0", "false"]:
            return 0
        # try numeric string
        try:
            x = int(float(s))
            return 1 if x == 1 else 0
        except Exception:
            return None
    if isinstance(v, (int, bool)):
        return 1 if int(v) == 1 else 0
    if isinstance(v, float):
        return 1 if int(v) == 1 else 0
    return None


def infer_is_lower_like_3d(row: pd.Series) -> int:
    """
    if any upper FDI columns exist AND are not NaN -> upper, else lower.
    """
    for t in UPPER_FDI:
        c = str(t)
        if c in row.index and not pd.isna(row[c]):
            return 0
    return 1


def get_gt_for_case(
    row: pd.Series,
    present_is_one: bool,
    jaw_col: Optional[str],
    case_id: Optional[str] = None,
    raw_id: Optional[str] = None,
) -> Tuple[List[int], int]:
    # 1) jaw
    is_lower = infer_is_lower_from_id(case_id)
    if is_lower is None:
        is_lower = infer_is_lower_from_id(raw_id)
    if is_lower is None and jaw_col is not None and jaw_col in row.index:
        is_lower = normalize_jaw_value(row[jaw_col])
    if is_lower is None and "is_lower" in row.index:
        is_lower = normalize_jaw_value(row["is_lower"])
    if is_lower is None:
        is_lower = infer_is_lower_like_3d(row)

    # 2) teeth16 (missing=1 present=0)
    fdis = LOWER_FDI if is_lower == 1 else UPPER_FDI
    gt = []
    for t in fdis:
        c = str(t)
        if c not in row.index:
            raise KeyError(f"CSV missing required FDI column: '{c}'")
        v = row[c]
        if pd.isna(v):
            # If NaN occurs for in-jaw tooth columns, treat as present(0) by default.
            # If your CSV uses NaN to mean "unknown", change this policy.
            x = 0
        else:
            x = int(float(v))
        if present_is_one:
            # present=1 missing=0 -> convert to missing=1 present=0
            x = 1 - x
        # now x should be missing (1) / present (0)
        gt.append(x)

    return gt, int(is_lower)


def build_transform(img_size: int, mean: List[float], std: List[float]):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_model(checkpoint_path: str, device: torch.device,
               model_py: Optional[str], model_class: Optional[str], state_key: Optional[str]):
    """
    Flexible loader:
    - If model_py+model_class provided: import class, instantiate, load state_dict from checkpoint dict.
    - Else: try torch.load(checkpoint) as a full model.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if model_py and model_class:
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_model_module", model_py)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        cls = getattr(mod, model_class)
        model = cls()

        # detect state_dict
        if isinstance(ckpt, dict):
            key = state_key
            if key is None:
                # common keys
                for k in ["model_state_dict", "state_dict", "model"]:
                    if k in ckpt and isinstance(ckpt[k], dict):
                        key = k
                        break
            if key is None:
                # maybe checkpoint itself is state_dict
                if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    sd = ckpt
                else:
                    raise ValueError("Cannot find state_dict in checkpoint dict. Provide --state_key.")
            else:
                sd = ckpt[key]
        else:
            raise ValueError("Checkpoint is not a dict; cannot load state_dict. Remove --model_py/--model_class to load full model.")
        model.load_state_dict(sd, strict=True)

    else:
        # assume checkpoint is a full model
        if isinstance(ckpt, torch.nn.Module):
            model = ckpt
        else:
            raise ValueError(
                "Checkpoint is not a torch.nn.Module. "
                "Provide --model_py and --model_class to construct model and load state_dict."
            )

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def score_view_like_3d(model, x: torch.Tensor) -> Tuple[float, float, int, torch.Tensor]:
    """
    x: [1,C,H,W]
    Returns:
      score, jaw_prob, pred_is_lower, logits[17]
    Score uses jaw confidence only (no GT leakage).
    """
    logits = model(x).squeeze(0)  # [17]
    if logits.numel() != 17:
        raise ValueError(f"Model output dim must be 17, got {logits.numel()}")

    jaw_logit = logits[16]
    jaw_prob = torch.sigmoid(jaw_logit).item()
    pred_is_lower = 1 if jaw_prob > 0.5 else 0
    score = abs(jaw_prob - 0.5) * 2.0
    return float(score), float(jaw_prob), int(pred_is_lower), logits.detach().cpu()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_selected(src_path: Path, dst_path: Path, mode: str):
    ensure_dir(dst_path.parent)
    if mode == "symlink":
        if dst_path.exists():
            dst_path.unlink()
        os.symlink(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)


def preload_image_cache(case2imgs: Dict[str, List[Path]]) -> Dict[str, Image.Image]:
    cache: Dict[str, Image.Image] = {}
    all_paths = sorted({str(p) for paths in case2imgs.values() for p in paths})
    if len(all_paths) == 0:
        return cache
    print(f"[Cache] Preloading {len(all_paths)} images...")
    failed = 0
    for p_str in all_paths:
        try:
            with Image.open(p_str) as img:
                cache[p_str] = img.convert("RGB").copy()
        except Exception as e:
            failed += 1
            print(f"[Cache] Failed to load {p_str}: {e}")
    print(f"[Cache] Loaded {len(cache)} images (failed {failed}).")
    return cache


def load_image(path: Path, img_cache: Optional[Dict[str, Image.Image]]) -> Image.Image:
    if img_cache is not None:
        cached = img_cache.get(str(path))
        if cached is not None:
            return cached.copy()
    with Image.open(path) as img:
        return img.convert("RGB")

# -------------------------
# Metrics
# -------------------------
def bin_metrics_from_counts(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
    acc  = (tp + tn) / (tp + fp + fn + tn) if (tp+fp+fn+tn) > 0 else 0.0
    tnr  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (rec + tnr)
    return prec, rec, f1, acc, bal_acc


def compute_jaw_aware_tooth_report(rows: List[dict]) -> pd.DataFrame:
    """
    rows contains per-case:
      gt_is_lower, pred_is_lower, gt_teeth16 (len16), pred_teeth16 (len16)
    For upper cases, tooth indices correspond to UPPER_FDI ordering; for lower cases correspond to LOWER_FDI.
    We create a 32-tooth report jaw-aware.
    """
    # Collect per-FDI truth/pred
    per_tooth = {t: {"truth": [], "pred": []} for t in (UPPER_FDI + LOWER_FDI)}

    for r in rows:
        is_lower = r["gt_is_lower"]
        gt16 = r["gt_teeth16"]
        pr16 = r["pred_teeth16"]
        fdis = LOWER_FDI if is_lower == 1 else UPPER_FDI
        for i, t in enumerate(fdis):
            per_tooth[t]["truth"].append(int(gt16[i]))
            per_tooth[t]["pred"].append(int(pr16[i]))

    records = []
    for t, d in per_tooth.items():
        y = d["truth"]
        p = d["pred"]
        if len(y) == 0:
            continue
        tp = sum((yy == 1 and pp == 1) for yy, pp in zip(y, p))
        fp = sum((yy == 0 and pp == 1) for yy, pp in zip(y, p))
        fn = sum((yy == 1 and pp == 0) for yy, pp in zip(y, p))
        tn = sum((yy == 0 and pp == 0) for yy, pp in zip(y, p))
        prec, rec, f1, acc, bal_acc = bin_metrics_from_counts(tp, fp, fn, tn)
        support = sum(y)  # missing=1 support= #missing
        jaw = "lower" if t in LOWER_FDI else "upper"
        records.append({
            "tooth_fdi": t,
            "jaw": jaw,
            "support_missing": support,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_samples": len(y),
        })

    df = pd.DataFrame(records).sort_values(["jaw", "tooth_fdi"]).reset_index(drop=True)
    return df


def print_per_tooth_metrics(df_tooth: pd.DataFrame):
    print("PER-TOOTH METRICS (JAW-AWARE, Support > 0)")
    print("------------------------------------------------------------------------------------------")
    print("FDI    Prec       Recall     F1         Acc        Supp   TP    FP    FN    TN")
    print("------------------------------------------------------------------------------------------")

    d = df_tooth[df_tooth["support_missing"] > 0]
    if len(d) == 0:
        print("No teeth with support > 0.")
        return

    for jaw_label, jaw_name in [("upper", "UPPER JAW"), ("lower", "LOWER JAW")]:
        j = d[d["jaw"] == jaw_label]
        if len(j) == 0:
            continue
        print(f"\n{jaw_name}:")
        for _, r in j.iterrows():
            print(
                f"{int(r['tooth_fdi']):<6}"
                f"{r['precision']:<11.4f}"
                f"{r['recall']:<11.4f}"
                f"{r['f1']:<11.4f}"
                f"{r['accuracy']:<11.4f}"
                f"{int(r['support_missing']):<7}"
                f"{int(r['tp']):<6}"
                f"{int(r['fp']):<6}"
                f"{int(r['fn']):<6}"
                f"{int(r['tn']):<6}"
            )


def compute_teeth_confusion(eval_rows: List[dict]) -> Tuple[Dict[str, int], List[List[int]]]:
    y_true, y_pred = [], []
    for r in eval_rows:
        y_true.extend(list(map(int, r["gt_teeth16"])))
        y_pred.extend(list(map(int, r["pred_teeth16"])))
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    cm = [[tn, fp], [fn, tp]]
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, cm


def save_confusion_matrix_plot(cm: List[List[int]], out_dir: Path):
    if plt is None:
        print("Warning: matplotlib not available; skipping confusion matrix plot.")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Present (0)", "Missing (1)"])
    ax.set_yticklabels(["Present (0)", "Missing (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (16+1 Architecture)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_16plus1.png", dpi=150)
    plt.close(fig)


def save_f1_per_jaw_plot(df_tooth: pd.DataFrame, out_dir: Path):
    if plt is None:
        print("Warning: matplotlib not available; skipping F1 per-jaw plot.")
        return
    upper_f1s = []
    for fdi in UPPER_FDI:
        row = df_tooth[(df_tooth["jaw"] == "upper") & (df_tooth["tooth_fdi"] == fdi)]
        upper_f1s.append(float(row["f1"].iloc[0]) if len(row) > 0 else 0.0)
    lower_f1s = []
    for fdi in LOWER_FDI:
        row = df_tooth[(df_tooth["jaw"] == "lower") & (df_tooth["tooth_fdi"] == fdi)]
        lower_f1s.append(float(row["f1"].iloc[0]) if len(row) > 0 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(len(UPPER_FDI)), upper_f1s, color="steelblue")
    axes[0].set_xticks(range(len(UPPER_FDI)))
    axes[0].set_xticklabels([str(f) for f in UPPER_FDI], rotation=45)
    axes[0].set_title("F1 Score - Upper Jaw")
    axes[0].set_ylim(0, 1.05)
    mean_upper = float(np.mean(upper_f1s)) if np is not None else (
        sum(upper_f1s) / len(upper_f1s) if upper_f1s else 0.0
    )
    axes[0].axhline(y=mean_upper, color="red", linestyle="--",
                   label=f"Mean: {mean_upper:.3f}")
    axes[0].legend()

    axes[1].bar(range(len(LOWER_FDI)), lower_f1s, color="seagreen")
    axes[1].set_xticks(range(len(LOWER_FDI)))
    axes[1].set_xticklabels([str(f) for f in LOWER_FDI], rotation=45)
    axes[1].set_title("F1 Score - Lower Jaw")
    axes[1].set_ylim(0, 1.05)
    mean_lower = float(np.mean(lower_f1s)) if np is not None else (
        sum(lower_f1s) / len(lower_f1s) if lower_f1s else 0.0
    )
    axes[1].axhline(y=mean_lower, color="red", linestyle="--",
                   label=f"Mean: {mean_lower:.3f}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "f1_score_per_jaw_16plus1.png", dpi=150)
    plt.close(fig)


def save_results_json(
    out_dir: Path,
    eval_rows: List[dict],
    df_tooth: pd.DataFrame,
    macro_prec: float,
    macro_rec: float,
    macro_f1: float,
    macro_acc: float,
    macro_bal_acc: float,
    micro_prec: float,
    micro_rec: float,
    micro_f1: float,
    micro_acc: float,
    micro_bal_acc: float,
    jaw_prec: float,
    jaw_rec: float,
    jaw_f1: float,
    jaw_acc: float,
    jaw_bal_acc: float,
    skipped: int,
):
    upper = [r for r in eval_rows if r["gt_is_lower"] == 0]
    lower = [r for r in eval_rows if r["gt_is_lower"] == 1]
    upper_acc = sum(r["gt_is_lower"] == r["pred_is_lower"] for r in upper) / len(upper) if upper else 0.0
    lower_acc = sum(r["gt_is_lower"] == r["pred_is_lower"] for r in lower) / len(lower) if lower else 0.0

    cm_counts, _ = compute_teeth_confusion(eval_rows)
    results = {
        "architecture": "16+1 (16 positions + 1 jaw classifier)",
        "cases": {
            "evaluated": len(eval_rows),
            "skipped": skipped,
        },
        "metrics": {
            "overall_macro": {
                "macro_precision": float(macro_prec),
                "macro_recall": float(macro_rec),
                "macro_f1": float(macro_f1),
                "macro_accuracy": float(macro_acc),
                "macro_balanced_accuracy": float(macro_bal_acc),
            },
            "overall_micro": {
                "micro_precision": float(micro_prec),
                "micro_recall": float(micro_rec),
                "micro_f1": float(micro_f1),
                "micro_accuracy": float(micro_acc),
                "micro_balanced_accuracy": float(micro_bal_acc),
            },
            "jaw_classification": {
                "jaw_accuracy": float(jaw_acc),
                "jaw_precision": float(jaw_prec),
                "jaw_recall": float(jaw_rec),
                "jaw_f1": float(jaw_f1),
                "jaw_balanced_accuracy": float(jaw_bal_acc),
            },
            "per_jaw": {
                "upper_jaw_accuracy": float(upper_acc),
                "upper_jaw_samples": int(len(upper)),
                "lower_jaw_accuracy": float(lower_acc),
                "lower_jaw_samples": int(len(lower)),
            },
            "confusion_matrix_teeth": cm_counts,
        },
        "per_tooth": df_tooth.to_dict(orient="records"),
    }
    json_dir = out_dir / "json"
    ensure_dir(json_dir)
    with open(json_dir / "test_results_16plus1.json", "w") as f:
        json.dump(results, f, indent=2)


def compute_overall_teeth_micro(rows: List[dict]) -> Tuple[float,float,float,float,float]:
    """Flatten all 16-teeth predictions for their true jaw and compute micro P/R/F1/Acc."""
    y_all, p_all = [], []
    for r in rows:
        y_all.extend(list(map(int, r["gt_teeth16"])))
        p_all.extend(list(map(int, r["pred_teeth16"])))
    tp = sum((yy == 1 and pp == 1) for yy, pp in zip(y_all, p_all))
    fp = sum((yy == 0 and pp == 1) for yy, pp in zip(y_all, p_all))
    fn = sum((yy == 1 and pp == 0) for yy, pp in zip(y_all, p_all))
    tn = sum((yy == 0 and pp == 0) for yy, pp in zip(y_all, p_all))
    return bin_metrics_from_counts(tp, fp, fn, tn)


def macro_over_teeth_with_support(df_tooth: pd.DataFrame) -> Tuple[float,float,float,float,float]:
    """Macro average over teeth with support_missing > 0"""
    d = df_tooth[df_tooth["support_missing"] > 0]
    if len(d) == 0:
        return 0.0,0.0,0.0,0.0,0.0
    return (
        float(d["precision"].mean()),
        float(d["recall"].mean()),
        float(d["f1"].mean()),
        float(d["accuracy"].mean()),
        float(d["balanced_accuracy"].mean()),
    )


# -------------------------
# Main
# -------------------------
def main():
    global available_gpus, device
    args = parse_args()
    img_dir = Path(args.img_dir).resolve()
    csv_path = Path(args.csv_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    # load CSV
    df = pd.read_csv(csv_path)
    if args.id_col not in df.columns:
        raise KeyError(f"CSV missing id column '{args.id_col}'. Available columns: {list(df.columns)[:20]} ...")

    # normalize id to string
    df[args.id_col] = df[args.id_col].astype(str).str.strip()
    df["_norm_id"] = df[args.id_col].apply(normalize_png_stem_to_newid)
    if df["_norm_id"].duplicated().any():
        dup_ids = df.loc[df["_norm_id"].duplicated(), "_norm_id"].unique().tolist()
        print(f"[Warn] Duplicated normalized IDs in CSV (showing up to 5): {dup_ids[:5]}")

    # index by case id
    df_case = df.drop_duplicates(subset=["_norm_id"]).set_index("_norm_id")

    # group images by case_id
    case_re = re.compile(args.case_regex)
    case2imgs: Dict[str, List[Path]] = {}
    total_imgs = 0
    regex_unmatched = 0
    csv_missing_imgs = 0
    matched_imgs = 0
    for p in img_dir.rglob("*.png"):
        total_imgs += 1
        cid_raw = to_case_id_from_filename(p, case_re)
        if cid_raw is None:
            regex_unmatched += 1
            continue
        cid_norm = normalize_png_stem_to_newid(cid_raw)
        if cid_norm in df_case.index:
            case2imgs.setdefault(cid_norm, []).append(p)
            matched_imgs += 1
        else:
            csv_missing_imgs += 1

    # keep only cases with labels + stable order
    case2imgs = {cid: sorted(paths) for cid, paths in case2imgs.items() if cid in df_case.index}
    if len(case2imgs) == 0:
        raise RuntimeError("No cases matched between images and CSV. Check --case_regex and --id_col.")

    csv_missing_cases = sorted(set(df_case.index) - set(case2imgs.keys()))
    print(
        "[Align] images={total} | regex_unmatched={rx} | img_missing_csv={img_csv} | "
        "matched_imgs={matched} | matched_cases={cases} | csv_missing_cases={csv_missing}".format(
            total=total_imgs,
            rx=regex_unmatched,
            img_csv=csv_missing_imgs,
            matched=matched_imgs,
            cases=len(case2imgs),
            csv_missing=len(csv_missing_cases),
        )
    )

    img_cache = preload_image_cache(case2imgs) if args.cache_images else None

    if args.device == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        if args.device.startswith("cuda:"):
            try:
                req_index = int(args.device.split(":", 1)[1])
            except Exception:
                req_index = 0
            visible_count = torch.cuda.device_count()
            if visible_count == 0:
                device = torch.device("cpu")
            elif req_index >= visible_count:
                print(
                    f"Warning: Requested {args.device} but only {visible_count} visible GPU(s); "
                    "falling back to cuda:0"
                )
                device = torch.device("cuda:0")
            else:
                device = torch.device(f"cuda:{req_index}")
        else:
            available_gpus = get_free_gpus()
            device = torch.device(f"cuda:{available_gpus[0]}")
    else:
        device = torch.device("cpu")
    print(f"Primary device: {device}")

    # load model
    model = load_model(
        checkpoint_path=str(args.checkpoint),
        device=device,
        model_py=args.model_py,
        model_class=args.model_class,
        state_key=args.state_key
    )

    transform = build_transform(args.img_size, args.mean, args.std)

    best1_dir = out_dir / "best1"
    ensure_dir(best1_dir)

    per_case_rows = []
    eval_rows = []

    skipped = 0
    for cid, paths in case2imgs.items():
        if len(paths) < args.min_views:
            skipped += 1
            continue

        row = df_case.loc[cid]
        raw_id = row[args.id_col] if args.id_col in row.index else None
        gt16, gt_is_lower = get_gt_for_case(
            row,
            present_is_one=args.present_is_one,
            jaw_col=args.jaw_col,
            case_id=cid,
            raw_id=raw_id,
        )

        # evaluate all views and keep topk
        scored = []
        for p in paths:
            img = load_image(p, img_cache)
            x = transform(img).unsqueeze(0).to(device)
            score, jaw_prob, pred_is_lower, logits = score_view_like_3d(model, x)
            pred16 = (logits[:16] > 0).int().tolist()
            scored.append({
                "path": p,
                "score": score,
                "jaw_prob": jaw_prob,
                "pred_is_lower": pred_is_lower,
                "logits": logits,          # cpu tensor[17]
                "pred16": pred16,
            })

        scored.sort(key=lambda d: d["score"], reverse=True)
        topk = scored[:max(1, args.topk)]

        # best1
        best1 = topk[0]
        best1_dst = best1_dir / f"{cid}__{best1['path'].name}"
        save_selected(best1["path"], best1_dst, mode=args.save_mode)

        # final prediction uses best1
        final_logits = best1["logits"]
        pred_is_lower_final = 1 if best1["jaw_prob"] > 0.5 else 0
        pred16_final = (final_logits[:16] > 0).int().tolist()

        per_case_rows.append({
            "case_id": cid,
            "case_id_raw": raw_id if raw_id is not None else cid,
            "gt_is_lower": gt_is_lower,
            "pred_is_lower": pred_is_lower_final,
            "best_path": str(best1["path"]),
            "best_saved": str(best1_dst),
            "best_score": best1["score"],
            "best_jaw_prob": best1["jaw_prob"],
        })

        eval_rows.append({
            "case_id": cid,
            "gt_is_lower": gt_is_lower,
            "pred_is_lower": pred_is_lower_final,
            "gt_teeth16": gt16,
            "pred_teeth16": pred16_final
        })

    # Save per-case CSV
    df_cases = pd.DataFrame(per_case_rows)
    df_cases.to_csv(out_dir / "per_case_best_views.csv", index=False)

    # Metrics
    if len(eval_rows) == 0:
        raise RuntimeError("No evaluable cases. Check your CSV matching and image grouping.")

    # Jaw audit output
    print("DEBUG MODE: Full Audit for Jaw (lower=1)")
    print(f"Found {len(per_case_rows)} cases.")
    print("\n" + "=" * 85)
    print(f"{'CASE ID':<35} | {'TRUTH':<5} | {'PRED':<5} | {'PROB':<8} | {'RESULT':<10}")
    print("=" * 85)

    tp = fp = fn = tn = 0
    for r in per_case_rows:
        truth = int(r["gt_is_lower"])
        pred = int(r["pred_is_lower"])
        prob = float(r["best_jaw_prob"])
        if truth == 1 and pred == 1:
            status = "TP"
            tp += 1
        elif truth == 0 and pred == 0:
            status = "TN"
            tn += 1
        elif truth == 1 and pred == 0:
            status = "FN"
            fn += 1
        else:
            status = "FP"
            fp += 1
        case_id_disp = r.get("case_id_raw") or r["case_id"]
        print(f"{case_id_disp:<35} | {truth:<5} | {pred:<5} | {prob:.4f}   | {status:<10}")

    print("=" * 85)
    jaw_prec, jaw_rec, jaw_f1, jaw_acc, jaw_bal_acc = bin_metrics_from_counts(tp, fp, fn, tn)
    print(f"SUMMARY: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"Recall for Jaw (lower=1): {jaw_rec:.4f}")
    print(f"Balanced Accuracy for Jaw: {jaw_bal_acc:.4f}")

    micro_prec, micro_rec, micro_f1, micro_acc, micro_bal_acc = compute_overall_teeth_micro(eval_rows)

    df_tooth = compute_jaw_aware_tooth_report(eval_rows)
    df_tooth.to_csv(out_dir / "per_tooth_report.csv", index=False)
    print_per_tooth_metrics(df_tooth)

    macro_prec, macro_rec, macro_f1, macro_acc, macro_bal_acc = macro_over_teeth_with_support(df_tooth)

    cm_counts, cm = compute_teeth_confusion(eval_rows)
    print(
        "Confusion Matrix (Teeth Missing/Present): "
        f"TN={cm_counts['tn']} FP={cm_counts['fp']} FN={cm_counts['fn']} TP={cm_counts['tp']}"
    )
    print(f"[[TN, FP], [FN, TP]] = {cm}")
    save_confusion_matrix_plot(cm, out_dir)
    save_f1_per_jaw_plot(df_tooth, out_dir)
    save_results_json(
        out_dir=out_dir,
        eval_rows=eval_rows,
        df_tooth=df_tooth,
        macro_prec=macro_prec,
        macro_rec=macro_rec,
        macro_f1=macro_f1,
        macro_acc=macro_acc,
        macro_bal_acc=macro_bal_acc,
        micro_prec=micro_prec,
        micro_rec=micro_rec,
        micro_f1=micro_f1,
        micro_acc=micro_acc,
        micro_bal_acc=micro_bal_acc,
        jaw_prec=jaw_prec,
        jaw_rec=jaw_rec,
        jaw_f1=jaw_f1,
        jaw_acc=jaw_acc,
        jaw_bal_acc=jaw_bal_acc,
        skipped=skipped,
    )

    # Print summary
    print(f"Overall Macro F1:             {macro_f1:.4f}")
    print(f"Macro Precision:              {macro_prec:.4f}")
    print(f"Macro Recall:                 {macro_rec:.4f}")
    print(f"Macro Accuracy:               {macro_acc:.4f}")
    print(f"Tooth Balanced Accuracy:      {macro_bal_acc:.4f}")
    print("------------------------------------------------------------------------------------------")
    print(f"Jaw Classification Accuracy:  {jaw_acc:.4f}")


if __name__ == "__main__":
    main()
