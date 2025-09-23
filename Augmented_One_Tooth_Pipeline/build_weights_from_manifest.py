import pandas as pd
import numpy as np
import json
from pathlib import Path
import torch
from typing import Dict, Set, Tuple
import re

# === CONFIG ===
# Base directory that contains the original dataset JSONs:
#   <BASE_DIR>/<jaw>/<CASE>/<CASE>_<jaw>.json
BASE_DIR = Path("/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split")

def _infer_jaw(row: pd.Series) -> str:
    """Infer jaw from 'jaw' column if present; otherwise from obj_path or case_id pattern."""
    jaw = (str(row.get("jaw", "")) or "").lower()
    if jaw in ("upper", "lower"):
        return jaw
    obj_path = str(row.get("obj_path", "")).lower()
    name = f"{row.get('case_id','')}".lower()
    if "_upper" in obj_path or name.endswith("_upper"):
        return "upper"
    if "_lower" in obj_path or name.endswith("_lower"):
        return "lower"
    # default fallback
    return "lower"

def _json_path_for(case_id: str, jaw: str) -> Path:
    """Compose the JSON path for a given case and jaw."""
    return BASE_DIR / jaw / case_id / f"{case_id}_{jaw}.json"

def _present_teeth_from_json(json_path: Path) -> Set[str]:
    """
    Parse JSON and return a set of tooth labels present in that case, as strings like 'tooth_36'.
    Assumes JSON has a 'labels' array with integers; 0 means background/gingiva.
    """
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return set()
    labels = data.get("labels", []) or []
    present = set()
    for lab in labels:
        if isinstance(lab, int) and lab != 0:
            present.add(f"tooth_{lab}")
    return present

def _parse_removed_field(val) -> Set[str]:
    """
    Parse the 'removed_teeth' cell into a set of valid tooth tokens like 'tooth_36'.
    Filters out empty strings, NaN, and malformed tokens.
    """
    if val is None:
        return set()
    if not isinstance(val, str):
        val = str(val)
    parts = [p.strip() for p in val.split(",")]
    valid = set()
    for p in parts:
        # accept tokens strictly matching 'tooth_<digits>'
        if re.fullmatch(r"tooth_\d+", p):
            valid.add(p)
    return valid

def build_vocab_and_stats(manifest_path: Path) -> Tuple[Dict[str, int], pd.DataFrame, torch.Tensor]:
    """
    Build (vocab, stats_df, pos_weight) using REAL presence from per-case JSON.
    - Vocab = union of all teeth that are present in any case (plus any that appear in removed_teeth).
    - Counting rule:
        For each sample row:
          * removed = set(removed_teeth)
          * present = teeth truly present for (case_id, jaw) from JSON
          For each tooth in global vocab:
              if tooth in present:
                  if tooth in removed -> pos += 1
                  else                -> neg += 1
              else:
                  (skip; do not count as negative)
    """
    df = pd.read_csv(manifest_path)
    if "removed_teeth" not in df.columns:
        raise ValueError(f"'removed_teeth' column not found in {manifest_path}")

    # Cache presence per (case_id, jaw)
    presence_cache: Dict[Tuple[str, str], Set[str]] = {}

    # First pass: gather global sets
    all_present: Set[str] = set()
    any_removed: Set[str] = set()

    for _, row in df.iterrows():
        case_id = str(row.get("case_id"))
        jaw = _infer_jaw(row)
        key = (case_id, jaw)
        if key not in presence_cache:
            jp = _json_path_for(case_id, jaw)
            presence_cache[key] = _present_teeth_from_json(jp)
        all_present |= presence_cache[key]

        removed = _parse_removed_field(row.get("removed_teeth", ""))
        any_removed |= removed

    # Build vocab: include all really present teeth + any that were ever removed (safety)
    all_teeth = sorted(t for t in (all_present | any_removed) if re.fullmatch(r"tooth_\d+", str(t)))
    vocab = {t: i for i, t in enumerate(all_teeth)}

    if not vocab:
        stats = pd.DataFrame(columns=["tooth","pos","neg","total","pos_rate","pos_weight"])
        return vocab, stats, torch.tensor([], dtype=torch.float32)

    # Second pass: count pos/neg with presence gating
    pos_counts = {t: 0 for t in vocab}
    neg_counts = {t: 0 for t in vocab}

    for _, row in df.iterrows():
        case_id = str(row.get("case_id"))
        jaw = _infer_jaw(row)
        present = presence_cache[(case_id, jaw)]
        removed = _parse_removed_field(row.get("removed_teeth", ""))

        for t in vocab:
            if t in present:
                if t in removed:
                    pos_counts[t] += 1
                else:
                    neg_counts[t] += 1
            # if not present -> do nothing

    # Build stats dataframe
    rows = []
    for t in sorted(vocab):
        pos = pos_counts[t]
        neg = neg_counts[t]
        tot = pos + neg
        pos_rate = (pos / tot) if tot > 0 else 0.0
        pw = (neg / pos) if pos > 0 else np.nan
        rows.append({"tooth": t, "pos": pos, "neg": neg, "total": tot, "pos_rate": pos_rate, "pos_weight": pw})

    stats = pd.DataFrame(rows).sort_values("tooth").reset_index(drop=True)
    stats["pos_weight"] = stats["pos_weight"].fillna(1.0)

    pos_weight = torch.tensor(stats["pos_weight"].values, dtype=torch.float32)
    return vocab, stats, pos_weight

def main():
    # === paths ===
    all_manifest = Path("/home/user/lzhou/week8/data_augmentation/Aug_data/manifest_all.csv")
    out_dir = Path("/home/user/lzhou/week8/data_augmentation/Aug_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not all_manifest.exists():
        print(f"[WARN] Missing manifest: {all_manifest}")
        return

    vocab, stats, pos_w = build_vocab_and_stats(all_manifest)

    # save artifacts
    stats_path = out_dir / "class_stats_all.csv"
    vocab_path = out_dir / "vocab_all.json"
    pw_path    = out_dir / "pos_weight_all.npy"

    stats.to_csv(stats_path, index=False)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    np.save(pw_path, pos_w.numpy())

    print("\n[ALL]")
    print(f"  Teeth classes: {len(vocab)}")
    print(f"  Stats CSV : {stats_path}")
    print(f"  Vocab JSON: {vocab_path}")
    print(f"  PosW NPY  : {pw_path}")
    print(f"  pos_weight (preview): {pos_w.tolist()[:8]}{' ...' if len(pos_w)>8 else ''}")
    print("  BASE_DIR for presence:", BASE_DIR)

    print("\nDone.")

if __name__ == "__main__":
    main()