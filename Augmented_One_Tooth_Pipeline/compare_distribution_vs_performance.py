import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # pos/neg/total/pos_rate）
    rename_map = {
        "missing_rate": "pos_rate",
        "pos_count": "pos",
        "neg_count": "neg",
        "present_cases": "total",
    }
    for k,v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})
    required = {"tooth","pos","neg","total","pos_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. required: {sorted(required)}")
    # tooth extract number for sorting
    df["tooth_num"] = df["tooth"].str.extract(r"(\d+)").astype(int)
    return df.sort_values("tooth_num").reset_index(drop=True)

def load_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tooth","F1","best_F1","best_th","support"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. required: {sorted(required)}")
    df["tooth_num"] = df["tooth"].str.extract(r"(\d+)").astype(int)
    return df.sort_values("tooth_num").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Control category distribution (class_stats_all) and evaluation results (per-tooth) visualization")
    ap.add_argument("--stats", type=Path, required=True, help="Path: class_stats_all.csv")
    ap.add_argument("--report", type=Path, required=True, help="Path: test_report_per_tooth.csv")
    ap.add_argument("--outdir", type=Path, default=Path("./charts_compare"), help="Output directory")
    ap.add_argument("--prefix", type=str, default="all", help="Output file name prefix")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.stats)
    rep   = load_report(args.report)

    merged = pd.merge(stats, rep, on=["tooth","tooth_num"], how="inner")
    merged_path = args.outdir / f"{args.prefix}_merged_stats_report.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[OK] Merged table has been saved: {merged_path}")

    # Plot 1："Number of Missing Pos" and "Number of Appearance Total" per tooth are side by side bar chart
    plt.figure(figsize=(14,6))
    width = 0.4
    xs = range(len(merged))
    plt.bar([x - width/2 for x in xs], merged["pos"], width=width, label="Missing count (pos)")
    plt.bar([x + width/2 for x in xs], merged["total"], width=width, label="Present cases (total)")
    plt.xticks(list(xs), merged["tooth"], rotation=90)
    plt.ylabel("Counts")
    plt.title("Per-tooth: Missing Count vs Present Cases")
    plt.legend()
    plt.tight_layout()
    out1 = args.outdir / f"{args.prefix}_pos_vs_total.png"
    plt.savefig(out1, dpi=220); plt.close()
    print(f"[OK] Saved: {out1}")

    # Plot 2: Dual-axis plot: Missing rate (pos_rate) on left axis (bars) vs F1 on right axis (line)
    fig, ax1 = plt.subplots(figsize=(14,6))
    ax2 = ax1.twinx()
    ax1.bar(merged["tooth"], merged["pos_rate"], alpha=0.8, label="Missing rate (pos_rate)")
    ax2.plot(merged["tooth"], merged["F1"], marker="o", linewidth=2, label="F1 (threshold=0.5)")
    ax1.set_xlabel("Tooth")
    ax1.set_ylabel("Missing rate (pos_rate)")
    ax2.set_ylabel("F1 (0~1)")
    ax1.set_title("Per-tooth: Missing Rate vs F1")
    for ax in (ax1, ax2):
        for label in ax.get_xticklabels(): label.set_rotation(90)
    # Merge legends
    lines, labels = [], []
    for ax in (ax1, ax2):
        L = ax.get_legend_handles_labels()
        lines += L[0]; labels += L[1]
    ax1.legend(lines, labels, loc="upper right")
    fig.tight_layout()
    out2 = args.outdir / f"{args.prefix}_rate_vs_f1_dualaxis.png"
    fig.savefig(out2, dpi=220); plt.close(fig)
    print(f"[OK] Saved: {out2}")

    # Plot 3: Scatter plot: X=Appearance Total, Y=F1, Point size ∝ Total, Color by Missing Rate
    plt.figure(figsize=(10,6))
    sizes = (merged["total"] / (merged["total"].max() + 1e-9)) * 400 + 10
    sc = plt.scatter(merged["total"], merged["F1"], s=sizes, c=merged["pos_rate"], cmap="viridis")
    plt.xlabel("Present cases per tooth (total)")
    plt.ylabel("F1 (threshold=0.5)")
    plt.title("F1 vs Present Cases (color = Missing rate)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Missing rate (pos_rate)")
    plt.tight_layout()
    out3 = args.outdir / f"{args.prefix}_scatter_F1_vs_total_col_rate.png"
    plt.savefig(out3, dpi=220); plt.close()
    print(f"[OK] Saved: {out3}")

    # Console quick summary: Minimum support / Lowest F1 for several teeth
    print("\n=== Minimum support for 8 teeth ===")
    print(merged.sort_values("total").head(8)[["tooth","total","pos","pos_rate","F1","best_F1","support"]])
    print("\n=== Lowest F1 for 8 teeth ===")
    print(merged.sort_values("F1").head(8)[["tooth","total","pos","pos_rate","F1","best_F1","support"]])

if __name__ == "__main__":
    main()