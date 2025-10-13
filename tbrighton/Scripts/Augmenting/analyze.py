import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============= CONFIGURATION =============
# --- Path to the new CSV file you want to analyze ---
RANDOM_AUGMENT_CSV = "/home/user/tbrighton/blender_outputs/augment_random/train_labels_random.csv"

# --- Label convention of the CSV ---
VALUE_FOR_MISSING = 1
VALUE_FOR_PRESENT = 0
# =========================================

# FDI Tooth Notation
UPPER_TEETH_STR = [str(t) for t in [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]]
LOWER_TEETH_STR = [str(t) for t in [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]]
ALL_TEETH_STR = sorted(UPPER_TEETH_STR + LOWER_TEETH_STR, key=int)

def plot_tooth_frequencies(missing_counts, present_counts, labels, save_path):
    """Generates and saves a grouped bar chart for missing and present tooth counts."""
    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 9))
    rects1 = ax.bar(x - width/2, missing_counts, width, label='Missing', color='#d62728')
    rects2 = ax.bar(x + width/2, present_counts, width, label='Present', color='#2ca02c')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Frequency Count', fontsize=14)
    ax.set_title('Frequency of Missing and Present Teeth in Augmented Dataset', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n✓ Chart saved to: {save_path}")

def analyze_dataset(csv_path):
    """
    Performs a full analysis of the dataset: validates jaw alignment,
    calculates frequencies, and plots the results.
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
    print("\n[1/3] Checking for jaw alignment errors...")
    misaligned_rows = []
    for index, row in df.iterrows():
        filename = row.get('filename', '')
        jaw_type = 'lower' if 'lower' in filename.lower() else 'upper'
        cols_to_check = UPPER_TEETH_STR if jaw_type == 'lower' else LOWER_TEETH_STR
        for tooth_col in cols_to_check:
            if tooth_col in df.columns and row[tooth_col] == VALUE_FOR_PRESENT:
                misaligned_rows.append({'row': index + 1, 'scan_jaw': jaw_type, 'error_tooth': tooth_col})
    
    if not misaligned_rows:
        print("✅ SUCCESS: No jaw misalignments found. Data integrity is solid.")
    else:
        print(f"⚠️ WARNING: Found {len(misaligned_rows)} rows with misaligned labels!")
        for entry in misaligned_rows[:5]:
            print(f"  - Row {entry['row']}: A '{entry['scan_jaw']}' scan incorrectly has tooth {entry['error_tooth']} marked as present.")

    # --- Analysis Step 2: Frequency Calculation ---
    print("\n[2/3] Calculating missing and present tooth frequencies...")
    tooth_columns = [col for col in ALL_TEETH_STR if col in df.columns]
    missing_counts = df[tooth_columns].sum(axis=0)
    present_counts = len(df) - missing_counts

    print("\n--- Frequency of Missing & Present Teeth (Total: {}) ---".format(len(df)))
    print("-" * 50)
    print(f"{'FDI Tooth':<15} {'Missing Count':<20} {'Present Count'}")
    print("-" * 50)
    for tooth in ALL_TEETH_STR:
        missing = missing_counts.get(tooth, 0)
        present = present_counts.get(tooth, 0)
        print(f"  {tooth:<13} {missing:<20} {present}")
    print("-" * 50)
    
    # --- Analysis Step 3: Plotting ---
    print("\n[3/3] Generating frequency distribution plot...")
    plot_save_path = filepath.parent / "tooth_presence_absence_counts.png"
    plot_tooth_frequencies(missing_counts, present_counts, ALL_TEETH_STR, plot_save_path)
    
    print("\n--- ANALYSIS COMPLETE ---")

if __name__ == "__main__":
    analyze_dataset(RANDOM_AUGMENT_CSV)
