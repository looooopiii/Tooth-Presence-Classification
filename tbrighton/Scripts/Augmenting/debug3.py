import pandas as pd
from pathlib import Path

# ============= CONFIGURATION =============
# --- Path to the new CSV file you want to analyze ---
RANDOM_AUGMENT_CSV = "/home/user/tbrighton/blender_outputs/augment_random/train_labels_random.csv"

# --- Label convention of the CSV ---
# Your CSVs use 1 for missing and 0 for present.
VALUE_FOR_MISSING = 1
VALUE_FOR_PRESENT = 0
# =========================================

# FDI Tooth Notation
UPPER_TEETH_STR = [str(t) for t in [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]]
LOWER_TEETH_STR = [str(t) for t in [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]]
ALL_TEETH_STR = sorted(UPPER_TEETH_STR + LOWER_TEETH_STR, key=int)

def analyze_dataset(csv_path):
    """
    Performs a full analysis of the dataset: validates jaw alignment and
    calculates the frequency of missing teeth.
    """
    filepath = Path(csv_path)
    print(f"--- ANALYZING DATASET: {filepath.name} ---")

    if not filepath.exists():
        print(f"\n❌ ERROR: File not found at '{csv_path}'. Please check the path and try again.")
        return

    try:
        df = pd.read_csv(filepath)
        print(f"  Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"\n❌ ERROR: Could not load or process the CSV file: {e}")
        return

    # --- Analysis Step 1: Jaw Alignment Validation ---
    print("\n[1/2] Checking for jaw alignment errors...")
    misaligned_rows = []
    for index, row in df.iterrows():
        filename = row.get('filename', '')
        jaw_type = 'lower' if 'lower' in filename.lower() else 'upper'
        
        cols_to_check, error_jaw = (UPPER_TEETH_STR, 'upper') if jaw_type == 'lower' else (LOWER_TEETH_STR, 'lower')
        
        for tooth_col in cols_to_check:
            if tooth_col in df.columns and row[tooth_col] == VALUE_FOR_PRESENT:
                misaligned_rows.append({
                    'row': index + 1, 'scan_jaw': jaw_type, 'error_tooth': tooth_col
                })
    
    if not misaligned_rows:
        print("✅ SUCCESS: No jaw misalignments found. Data integrity is solid.")
    else:
        print(f"⚠️ WARNING: Found {len(misaligned_rows)} rows with misaligned labels!")
        for entry in misaligned_rows[:5]:
            print(f"  - Row {entry['row']}: A '{entry['scan_jaw']}' scan incorrectly has tooth {entry['error_tooth']} marked as present.")

    # --- Analysis Step 2: Missing Tooth Frequency Count ---
    print("\n[2/2] Calculating missing tooth frequency...")
    
    # Ensure all tooth columns are treated as numeric for summation
    tooth_columns = [col for col in ALL_TEETH_STR if col in df.columns]
    missing_counts = df[tooth_columns].sum(axis=0)

    print("\n--- Frequency of Missing Teeth (Total Samples: {}) ---".format(len(df)))
    print("-" * 55)
    print(f"{'Upper Right':<28}{'Upper Left':<28}")
    for i in range(8):
        r_tooth, l_tooth = UPPER_TEETH_STR[7-i], UPPER_TEETH_STR[8+i]
        r_count, l_count = missing_counts.get(r_tooth, 0), missing_counts.get(l_tooth, 0)
        print(f"  Tooth {r_tooth}: {r_count:<5} missing      |    Tooth {l_tooth}: {l_count:<5} missing")
    
    print("-" * 55)
    print(f"{'Lower Right':<28}{'Lower Left':<28}")
    for i in range(8):
        r_tooth, l_tooth = LOWER_TEETH_STR[15-i], LOWER_TEETH_STR[i]
        r_count, l_count = missing_counts.get(r_tooth, 0), missing_counts.get(l_tooth, 0)
        print(f"  Tooth {r_tooth}: {r_count:<5} missing      |    Tooth {l_tooth}: {l_count:<5} missing")
    print("-" * 55)
    
    print("\n--- ANALYSIS COMPLETE ---")

if __name__ == "__main__":
    analyze_dataset(RANDOM_AUGMENT_CSV)
