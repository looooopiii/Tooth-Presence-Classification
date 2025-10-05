"""
Flip tooth labels in a wide table:
- Preserve all original headers and order.
- For tooth columns (columns with digits in their names or starting with 'tooth_'):
    * 0 -> 1
    * 1 -> 0
    * NaN stays NaN
- Non-tooth columns (e.g. File, PatientID) are copied unchanged.
"""

import argparse
import pandas as pd
from pathlib import Path

def flip_labels(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    tooth_cols = []
    for c in out_df.columns:
        key = str(c).strip().lower()
        if any(ch.isdigit() for ch in key) or key.startswith("tooth_"):
            tooth_cols.append(c)

    for c in tooth_cols:
        s = pd.to_numeric(out_df[c], errors="coerce")
        out_df[c] = s.where(s.isna(), 1 - s)
    return out_df

def main():
    parser = argparse.ArgumentParser(description="Flip 0<->1 in tooth label table, keep NaN.")
    parser.add_argument("--in_path", required=True, help="Input CSV/XLSX")
    parser.add_argument("--out_path", required=True, help="Output CSV")
    args = parser.parse_args()

    # load table (csv or excel)
    if args.in_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.in_path)
    else:
        df = pd.read_csv(args.in_path)

    out = flip_labels(df)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)
    print(f"[OK] Wrote flipped labels to {args.out_path}")

if __name__ == "__main__":
    main()