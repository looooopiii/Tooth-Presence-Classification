import pandas as pd
from pathlib import Path

# ============= CONFIGURATION =============
# --- Path to the CSV file you want to validate ---
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"

# --- Label convention of the CSV ---
# Your CSV has flipped labels, so a 'present' tooth is marked with 0.
VALUE_FOR_PRESENT = 0
# =========================================

# FDI Tooth Notation
UPPER_TEETH_STR = [str(t) for t in [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]]
LOWER_TEETH_STR = [str(t) for t in [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]]


def validate_jaw_alignment(csv_path):
    """
    Scans the entire CSV to find rows where a tooth is marked as 'present'
    in an arch where it cannot physically exist.
    """
    print(f"--- VALIDATING JAW ALIGNMENT FOR: {Path(csv_path).name} ---")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"\n[ERROR] Could not load or process the CSV file: {e}")
        return

    misaligned_rows = []

    for index, row in df.iterrows():
        filename = row.get('filename', '')
        if 'lower' in filename.lower():
            jaw_type = 'lower'
        elif 'upper' in filename.lower():
            jaw_type = 'upper'
        else:
            print(f"Warning: Could not determine jaw type for row {index+1}. Skipping.")
            continue
            
        misaligned_teeth = []
        
        # Check for misalignments in a 'lower' jaw scan
        if jaw_type == 'lower':
            for tooth_col in UPPER_TEETH_STR:
                if tooth_col in df.columns and row[tooth_col] == VALUE_FOR_PRESENT:
                    misaligned_teeth.append(tooth_col)
        
        # Check for misalignments in an 'upper' jaw scan
        elif jaw_type == 'upper':
            for tooth_col in LOWER_TEETH_STR:
                if tooth_col in df.columns and row[tooth_col] == VALUE_FOR_PRESENT:
                    misaligned_teeth.append(tooth_col)
                    
        if misaligned_teeth:
            misaligned_rows.append({
                'row_index': index + 1,
                'filename': filename,
                'scan_jaw': jaw_type,
                'misaligned_teeth_present': misaligned_teeth
            })

    # --- Print Summary Report ---
    if not misaligned_rows:
        print("\nSUCCESS: No misaligned entries found. All labels are consistent with their jaw types.")
    else:
        print(f"\nWARNING: Found {len(misaligned_rows)} rows with misaligned labels!")
        for entry in misaligned_rows:
            print("-" * 50)
            print(f"  Row Index: {entry['row_index']}")
            print(f"  Filename: {entry['filename']}")
            print(f"  Scan Jaw Type: {entry['scan_jaw']}")
            print(f"  Incorrectly Marked as 'PRESENT': {entry['misaligned_teeth_present']}")
    
    print("\n--- VALIDATION COMPLETE ---")


if __name__ == "__main__":
    validate_jaw_alignment(TEST_LABELS_CSV)

