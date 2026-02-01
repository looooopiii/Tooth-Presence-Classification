import json
from pathlib import Path
from collections import Counter

DATASETS = {
    "AUGMENT_RANDOM": "/home/user/tbrighton/blender_outputs/augment_random_fixed",
    "AUGMENT_TEST":   "/home/user/tbrighton/blender_outputs/augment_test_fixed"
}

UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

def analyze_single_dataset(name, path_str):
    print(f"\n\n{'#'*80}")
    print(f"ANALYZING DATASET: {name}")
    print(f"Path: {path_str}")
    print(f"{'#'*80}")
    
    root_path = Path(path_str)
    if not root_path.exists():
        print(f"⚠️ Path not found: {root_path}")
        return

    upper_counts = Counter()
    lower_counts = Counter()
    total_upper = 0
    total_lower = 0
    
    # scan files
    all_json_files = list(root_path.rglob("*.json"))
    print(f"Found {len(all_json_files)} JSON files. Processing...")
    
    for json_file in all_json_files:
        filename = json_file.name.lower()
        
        # Determine Jaw
        if "lower" in filename:
            is_lower = True
        elif "upper" in filename:
            is_lower = False
        else:
            continue 

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            present_labels = set(data.get("labels", []))
            
            if is_lower:
                total_lower += 1
                for label in present_labels:
                    if label in LOWER_FDI: lower_counts[label] += 1
            else:
                total_upper += 1
                for label in present_labels:
                    if label in UPPER_FDI: upper_counts[label] += 1
                    
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    # --- REPORTING FUNCTION ---
    def print_jaw_report(jaw_name, fdi_list, counts, total):
        print(f"\n>>> {jaw_name.upper()} JAW ({total} scans)")
        print(f"{'FDI':<6} {'Missing':<10} {'Missing %':<12} {'Status'}")
        print("-" * 55)
        
        if total == 0:
            print("No scans.")
            return

        for tooth in sorted(fdi_list):
            present = counts.get(tooth, 0)
            missing = total - present
            miss_pct = (missing / total * 100)
            
            status = ""
            if miss_pct > 25: status = "✅ BALANCED"
            elif miss_pct > 10: status = "✅ OK"
            elif miss_pct > 1: status = "⚠️ LOW"
            else: status = "❌ RARE"
            
            print(f"{tooth:<6} {missing:<10} {miss_pct:>10.1f}%      {status}")

    print_jaw_report("Upper", UPPER_FDI, upper_counts, total_upper)
    print_jaw_report("Lower", LOWER_FDI, lower_counts, total_lower)


def main():
    for name, path in DATASETS.items():
        analyze_single_dataset(name, path)

if __name__ == "__main__":
    main()
