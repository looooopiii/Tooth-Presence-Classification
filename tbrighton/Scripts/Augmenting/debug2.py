import pandas as pd
from pathlib import Path

# ============= CONFIGURATION =============
# --- Add the full paths to any CSV files you want to check ---
FILES_TO_CHECK = [
    # Path to the new augmented dataset's labels
    "/home/user/tbrighton/blender_outputs/augment_test/train_labels_augmented.csv",
    
    # Path to your original training set labels (if you have a CSV for it)
    # "/path/to/your/original_train_labels.csv", 
    
    # Path to your test set labels
    "/home/user/tbrighton/Scripts/Testing/3D/label_flipped.csv"
]

# --- Label convention of the CSVs ---
# If your CSVs use 1 for missing and 0 for present, this should be 0.
VALUE_FOR_PRESENT = 0
# =========================================

# FDI Tooth Notation
UPPER_TEETH_STR = [str(t) for t in [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]]
LOWER_TEETH_STR = [str(t) for t in [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]]


def validate_jaw_alignment(csv_path):
    """
    Scans a CSV file to find rows where a tooth is marked as 'present'
    in an arch where it cannot physically exist.
    """
    filepath = Path(csv_path)
    print(f"\n--- VALIDATING JAW ALIGNMENT IN: {filepath.name} ---")
    
    if not filepath.exists():
        print(f"❌ ERROR: File not found at '{csv_path}'. Skipping.")
        return

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ ERROR: Could not load or process the CSV file: {e}")
        return

    misaligned_rows = []

    for index, row in df.iterrows():
        filename = row.get('filename', '')
        if 'lower' in filename.lower():
            jaw_type = 'lower'
        elif 'upper' in filename.lower():
            jaw_type = 'upper'
        else:
            # Skip rows where jaw type can't be determined from filename
            continue
            
        misaligned_teeth = []
        
        # Define which columns to check for incorrect presence
        cols_to_check = []
        if jaw_type == 'lower':
            cols_to_check = UPPER_TEETH_STR
        elif jaw_type == 'upper':
            cols_to_check = LOWER_TEETH_STR
        
        # Find misalignments
        for tooth_col in cols_to_check:
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
        print(f"✅ SUCCESS: No jaw misalignments found in '{filepath.name}'.")
    else:
        print(f"⚠️ WARNING: Found {len(misaligned_rows)} rows with misaligned labels in '{filepath.name}'!")
        # Print details for the first 5 problematic rows for brevity
        for entry in misaligned_rows[:5]:
            print("-" * 60)
            print(f"  Row Index: {entry['row_index']}")
            print(f"  Filename: {entry['filename']}")
            print(f"  Scan Jaw Type: {entry['scan_jaw']}")
            print(f"  Incorrectly Marked as 'PRESENT' in opposite jaw: {entry['misaligned_teeth_present']}")
    
    print("-" * 60)


if __name__ == "__main__":
    print("Starting data integrity validation...")
    for file_path in FILES_TO_CHECK:
        validate_jaw_alignment(file_path)
    print("\nValidation complete.")

