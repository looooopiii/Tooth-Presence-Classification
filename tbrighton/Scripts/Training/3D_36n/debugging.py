import json
from pathlib import Path
from collections import Counter

# ============= CONFIGURATION =============
DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]

# Separate FDI lists for cleaner reporting
UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

def analyze_tooth_frequency_jaw_wise(data_paths):
    print("Starting Jaw-Wise Tooth Frequency Analysis...")
    
    # Separate counters and totals
    upper_counts = Counter()
    lower_counts = Counter()
    
    total_upper_scans = 0
    total_lower_scans = 0

    for path_str in data_paths:
        data_path = Path(path_str)
        if not data_path.exists():
            print(f"[Warning] Path not found: {data_path}")
            continue
        
        # Determine jaw type from folder name logic
        # Assuming the path string itself contains 'lower' or 'upper'
        is_lower_path = "lower" in str(data_path).lower()
        current_jaw_type = "lower" if is_lower_path else "upper"
        
        for case_dir in data_path.iterdir():
            if case_dir.is_dir():
                case_id = case_dir.name
                json_file = case_dir / f"{case_id}_{current_jaw_type}.json"
                
                if json_file.exists():
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        unique_labels = set(data.get("labels", []))
                        
                        if current_jaw_type == "lower":
                            total_lower_scans += 1
                            for label in unique_labels:
                                if label in LOWER_FDI:
                                    lower_counts[label] += 1
                        else:
                            total_upper_scans += 1
                            for label in unique_labels:
                                if label in UPPER_FDI:
                                    upper_counts[label] += 1
                                    
                    except Exception as e:
                        print(f"[Error] Failed to read {json_file}: {e}")

    # --- REPORTING ---
    
    def print_jaw_report(title, fdi_list, counts, total_scans):
        print("\n" + "="*80)
        print(f" {title.upper()} JAW REPORT (Total Scans: {total_scans})")
        print("="*80)
        print(f"{'FDI':<6} {'Present':<10} {'Missing':<10} {'Presence %':<12} {'Missing %':<12} {'Status'}")
        print("-" * 80)
        
        for tooth in sorted(fdi_list):
            present = counts.get(tooth, 0)
            missing = total_scans - present
            
            pres_pct = (present / total_scans * 100) if total_scans > 0 else 0
            miss_pct = (missing / total_scans * 100) if total_scans > 0 else 0
            
            # Highlight missing teeth
            status = ""
            if miss_pct > 50: status = "⚠️ HIGHLY MISSING"
            elif miss_pct > 20: status = "📉 Common Missing"
            elif miss_pct == 0: status = "✅ Always Present"
            
            print(f"{tooth:<6} {present:<10} {missing:<10} {pres_pct:>10.1f}% {miss_pct:>10.1f}%   {status}")

    print_jaw_report("Upper", UPPER_FDI, upper_counts, total_upper_scans)
    print_jaw_report("Lower", LOWER_FDI, lower_counts, total_lower_scans)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print(f"Total Jaws Analyzed: {total_upper_scans + total_lower_scans}")
    print("="*80)

if __name__ == "__main__":
    analyze_tooth_frequency_jaw_wise(DATA_PATHS)
