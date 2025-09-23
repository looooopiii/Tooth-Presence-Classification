import argparse, re
from pathlib import Path
import pandas as pd

# Helper imports and notes for column picking and fuzzy matching

TOOTH_RE = re.compile(r"^tooth_(\d{2})$")

def detect_tooth_cols(df):
    cols = [c for c in df.columns if TOOTH_RE.match(str(c))]
    if not cols:
        raise ValueError("cannot find tooth columns (e.g. tooth_11/tooth_48). Please check column names.")
    # rank by tooth number
    cols = sorted(cols, key=lambda c: int(TOOTH_RE.match(c).group(1)))
    return cols

def pick_col(df, candidates, required=True, user_choice=None, purpose=""):
    """
    Try to pick a column from a list of candidates.
    If user_choice is provided and exists, use it.
    Else try exact match (case-sensitive then case-insensitive),
    then fuzzy match on common patterns.
    """
    if user_choice:
        if user_choice in df.columns:
            return user_choice
        # case-insensitive hit
        for c in df.columns:
            if c.lower() == user_choice.lower():
                return c
        raise ValueError(f"Specified column '{user_choice}' for {purpose or 'unknown'} not found in CSV. Available: {list(df.columns)}")

    # exact match
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive exact
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    # fuzzy: common patterns
    patt_map = {
        "case": [r"^case[_\s]*id$", r"^case$", r"^scan[_\s]*id$", r"^study[_\s]*id$", r".*case.*", r".*id$"],
        "jaw":  [r"^jaw$", r"^arch$", r".*jaw.*", r".*arch.*"],
    }
    import re
    # decide which pattern set to use
    key = "jaw" if any(k.lower() == "jaw" for k in candidates) else "case"
    patterns = patt_map[key]
    for p in patterns:
        rx = re.compile(p, re.IGNORECASE)
        hits = [c for c in df.columns if rx.match(c)]
        if hits:
            return hits[0]

    if required:
        raise ValueError(f"cannot find column for {purpose or candidates}. Candidates={candidates}. Available columns={list(df.columns)}")
    return None

def build_image_path(image_root: Path, case_id: str, jaw: str, view: str):
    # If case_id already contains "UpperJawScan" / "LowerJawScan" (optionally with numeric suffix),
    # we should NOT append another jaw token. Example filenames:
    #   {image_root}/{case_id}_top.png
    # where case_id looks like "12345_2022-09-14 LowerJawScan" or "12345_2022-09-14 UpperJawScan001".
    cid_lower = case_id.lower()
    import re as _re
    if _re.search(r"\b(lowerjawscan\d*|upperjawscan\d*)\b", cid_lower):
        return image_root / f"{case_id}_{view}.png"
    # Otherwise, we synthesize the jaw token.
    jaw_str = "LowerJawScan" if jaw == "lower" else "UpperJawScan"
    return image_root / f"{case_id} {jaw_str}_{view}.png"

def main():
    ap = argparse.ArgumentParser(description="Convert wide table (per-tooth 0/1) to manifest_ios.csv")
    ap.add_argument("--csv", type=Path, required=True,
                    help="Input: cleaned_labels_missing_is_1.csv")
    ap.add_argument("--image_root", type=Path, required=True,
                    help="Image root directory for rendering, e.g. /local/scratch/datasets/IntraOralScans/data/renders_top")
    ap.add_argument("--out_csv", type=Path, required=True,
                    help="Output: manifest_ios.csv")
    ap.add_argument("--view", type=str, default="top", help="View name (default: top)")
    ap.add_argument("--case_col", type=str, default=None, help="Column name for case_id if not standard")
    ap.add_argument("--jaw_col", type=str, default=None, help="Column name for jaw if not standard")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # case_id / jaw with overrides and fuzzy matching
    c_case = pick_col(df, ["case_id","CaseID","case","Case","File"], user_choice=args.case_col, purpose="case_id")
    c_jaw  = pick_col(df, ["jaw","Jaw","arch","Arch"], required=False, user_choice=args.jaw_col, purpose="jaw")

    tooth_cols = detect_tooth_cols(df)

    rows = []
    missing_paths = 0
    for _, r in df.iterrows():
        case_id = str(r[c_case]).strip()

        if c_jaw:
            jaw_raw = str(r[c_jaw]).strip().lower()
            if jaw_raw in ("u","upper"): jaw = "upper"
            elif jaw_raw in ("l","lower"): jaw = "lower"
            else: jaw = jaw_raw
        else:
            # if no jaw column, infer from case_id content
            low = case_id.lower()
            if "upperjawscan" in low or "upperjaw" in low or "_upper" in low or low.endswith("u"):
                jaw = "upper"
            elif "lowerjawscan" in low or "lowerjaw" in low or "_lower" in low or low.endswith("l"):
                jaw = "lower"
            else:
                jaw = "lower"  # default

        removed_list = []
        for tcol in tooth_cols:
            val = r.get(tcol)
            try:
                v = int(val)
            except Exception:
                v = 0
            if v == 1:
                removed_list.append(tcol)

        img_path = build_image_path(args.image_root, case_id, jaw, args.view)
        if not img_path.exists():
            missing_paths += 1

        rows.append({
            "case_id": case_id,
            "jaw": jaw,
            "image_path": str(img_path),
            "removed_teeth": ",".join(removed_list),
            "mode": "external"
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Merged manifest has been saved: {args.out_csv}  (total {len(out)} rows)")
    if missing_paths:
        print(f"[WARN] {missing_paths} image paths are missing, please check --image_root or naming template (e.g., {build_image_path(Path('[ROOT]'),'CASE','lower',args.view)})")
    # Quick stats: Number of samples with at least one missing tooth
    k = (out["removed_teeth"].str.len() > 0).sum()
    print(f"[Info] Samples with missing labels: {k} / {len(out)}")

if __name__ == "__main__":
    main()