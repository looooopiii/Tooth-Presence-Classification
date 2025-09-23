import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_stats(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"tooth","pos","neg","total","pos_rate","pos_weight"}.issubset(df.columns):
        raise ValueError("CSV lost columns: tooth,pos,neg,total,pos_rate,pos_weight")
    # tooth column to int for sorting
    df["tooth_num"] = df["tooth"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("tooth_num").reset_index(drop=True)
    return df

def plot_and_save(df: pd.DataFrame, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # missing counts
    plt.figure(figsize=(14,6))
    plt.bar(df["tooth"], df["pos"])
    plt.xticks(rotation=90)
    plt.ylabel("Missing count (pos)")
    plt.title("Per-tooth Missing Counts")
    plt.tight_layout()
    out1 = out_dir / f"{prefix}_missing_counts.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # missing rate
    plt.figure(figsize=(14,6))
    plt.bar(df["tooth"], df["pos_rate"])
    plt.xticks(rotation=90)
    plt.ylabel("Missing rate (pos_rate)")
    plt.title("Per-tooth Missing Rate")
    plt.tight_layout()
    out2 = out_dir / f"{prefix}_missing_rate.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    # print top-10 most/least missing teeth
    topk = df.sort_values("pos", ascending=False).head(10)[["tooth","pos","pos_rate"]]
    rarek = df[df["pos"]>0].sort_values("pos").head(10)[["tooth","pos","pos_rate"]]
    print("\nTop-10 most missing teeth:\n", topk.to_string(index=False))
    print("\nTop-10 least missing teeth (>0):\n", rarek.to_string(index=False))

    print(f"\nSaved:\n  {out1}\n  {out2}")

def main():
    ap = argparse.ArgumentParser(description="Visualize per-tooth missing counts and rates from class_stats_all.csv")
    ap.add_argument("--csv", type=Path, required=True,
                    help="Path to class_stats_all.csv")
    ap.add_argument("--out", type=Path, default=Path("./charts"),
                    help="Output directory for charts (default: ./charts)")
    ap.add_argument("--prefix", type=str, default="class_stats",
                    help="Output file name prefix (default: class_stats)")
    args = ap.parse_args()

    df = load_stats(args.csv)
    plot_and_save(df, args.out, args.prefix)

if __name__ == "__main__":
    main()