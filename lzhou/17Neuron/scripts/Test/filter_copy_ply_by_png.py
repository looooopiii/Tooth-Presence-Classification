import argparse
from pathlib import Path
import shutil

def collect_png_basenames(png_dir):
    """
    Collects the base names of PNG files in the specified directory, removing the '_top' suffix.
    """
    basenames = set()
    for png in Path(png_dir).rglob("*.png"):
        stem = png.stem  # e.g., '... LowerJawScan_top'
        if stem.endswith("_top"):
            stem = stem[:-4]  # remove suffix '_top'
        basenames.add(stem)
    return basenames

def copy_matching_ply(ply_dir, basenames_set, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total, matched = 0, 0
    missing = []

    for ply in Path(ply_dir).glob("*.ply"):
        total += 1
        if ply.stem in basenames_set:
            dst = out_dir / ply.name
            shutil.copy2(ply, dst)
            matched += 1
        else:
            # Only record those not in the PNG list
            missing.append(ply.stem)

    return total, matched, missing

def main():
    parser = argparse.ArgumentParser(description="Filter and copy PLY files based on PNG filenames without '_top' suffix.")
    parser.add_argument("--ply_dir", required=True, help="Folder containing source .ply files.")
    parser.add_argument("--png_dir", required=True, help="Folder containing *_top.png files.")
    parser.add_argument("--out_dir", required=True, help="Output folder to copy matched .ply files.")
    args = parser.parse_args()

    basenames = collect_png_basenames(args.png_dir)
    print(f"[info] Collected {len(basenames)} PNG basenames (without '_top') from: {args.png_dir}")

    total, matched, missing = copy_matching_ply(args.ply_dir, basenames, args.out_dir)
    print(f"[done] Scanned PLY: {total}, Copied: {matched}, Skipped: {total - matched}")
    if missing:
        print(f"[note] Example PLY stems not found in PNG list (showing up to 10): {missing[:10]}")

if __name__ == "__main__":
    main()