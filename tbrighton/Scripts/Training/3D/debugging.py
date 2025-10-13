import json
from pathlib import Path
from collections import Counter

# ============= CONFIGURATION =============
# These paths are correct. The logic to read them was the issue.
DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]

# Define all 32 valid permanent teeth in FDI notation
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
# =========================================


def analyze_tooth_frequency_fdi(data_paths):
    """
    Scans dataset JSON files to count the occurrence of each tooth using FDI notation.
    This version correctly navigates the data directory structure.
    """
    print("Starting tooth frequency analysis (with correct FDI Notation and path logic)...")
    
    tooth_counts = Counter()
    all_json_files = []

    # --- CORRECTED FILE DISCOVERY LOGIC ---
    for path_str in data_paths:
        data_path = Path(path_str)
        if not data_path.exists():
            print(f"[Warning] Path not found, skipping: {data_path}")
            continue
        
        # Determine the jaw type from the base path
        jaw_type = "lower" if "lower" in str(data_path) else "upper"
        
        # Iterate through the items in the data_path (e.g., '.../lower/')
        for case_dir in data_path.iterdir():
            # The critical fix: We check IF the item is a directory (e.g., '0AAQ6BO3')
            if case_dir.is_dir():
                case_id = case_dir.name
                
                # Construct the full path to the JSON file inside this case directory
                json_file = case_dir / f"{case_id}_{jaw_type}.json"
                
                # Check if the file actually exists before adding it to our list
                if json_file.exists():
                    all_json_files.append(json_file)

    if not all_json_files:
        print("\n[Error] Still no JSON files found. Please double-check:")
        print("1. Your DATA_PATHS are correct.")
        print("2. The directory structure is exactly as described (e.g., .../lower/CASE_ID/CASE_ID_lower.json).")
        print("3. You have read permissions for these directories.")
        return

    total_scans = len(all_json_files)
    print(f"✓ Found {total_scans} total scans to analyze.")

    # Now, process each file and count the teeth
    for json_file in all_json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            unique_labels = set(data.get("labels", []))
            
            for label in unique_labels:
                if label in VALID_FDI_LABELS:
                    tooth_counts[label] += 1
        except Exception as e:
            print(f"[Error] Could not process file {json_file}: {e}")

    # --- The reporting section remains the same ---
    print("\n" + "="*60)
    print("         TOOTH FREQUENCY AND PRESENCE REPORT (FDI)")
    print("="*60)
    print(f"Total Scans Analyzed: {total_scans}")
    print("-" * 60)
    print(f"{'FDI Tooth':<12} {'Appearance Count':<20} {'Presence (%)':<15}")
    print("-" * 60)

    for tooth_id in VALID_FDI_LABELS:
        count = tooth_counts.get(tooth_id, 0)
        percentage = (count / total_scans) * 100 if total_scans > 0 else 0
        
        rarity_flag = "❗️ RARE" if 0 < count < (total_scans * 0.2) else ""
        if count == 0:
            rarity_flag = "❌ NOT FOUND"

        print(f"Tooth {tooth_id:<5} {count:<20} {percentage:>12.2f}%   {rarity_flag}")
        
    print("="*60)


if __name__ == "__main__":
    analyze_tooth_frequency_fdi(DATA_PATHS)