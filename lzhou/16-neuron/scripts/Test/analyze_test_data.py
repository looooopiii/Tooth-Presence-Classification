import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# Use a non-interactive backend, which is essential for running on servers without a GUI
matplotlib.use('Agg')

# ============= CONFIGURATION =============
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"

# Directory where the output plot will be saved
OUTPUT_DIR = "/home/user/lzhou/week10/output/Test2D/normal/plots"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- Use the exact same FDI mapping for consistency ---
VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
# =========================================


def analyze_and_plot_test_set_frequency(csv_path, save_path):
    """
    Analyzes the tooth frequency from the test set CSV and generates a bar plot.

    Args:
        csv_path (str): The file path to the CSV containing the test labels.
        save_path (Path): The path where the output plot image will be saved.
    """
    print("Starting tooth frequency analysis and plotting for the TEST SET...")
    csv_file = Path(csv_path)

    # --- 1. Load the CSV file ---
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"\n[ERROR] The file was not found at: {csv_file}")
        print("Please make sure the TEST_LABELS_CSV path is correct.")
        return

    total_scans = len(df)
    if total_scans == 0:
        print("[Warning] The CSV file is empty. No data to plot.")
        return
        
    print(f" Successfully loaded {total_scans} scans from the CSV.")

    # --- 2. Count the occurrences for each tooth ---
    tooth_counts = {}
    for tooth_id in VALID_FDI_LABELS:
        tooth_str = str(tooth_id)
        if tooth_str in df.columns:
            count = df[tooth_str].sum()
            tooth_counts[tooth_id] = int(count)
        else:
            tooth_counts[tooth_id] = 0

    # --- 3. Generate the Bar Plot ---
    # Prepare data for plotting
    fdi_labels = [str(label) for label in tooth_counts.keys()]
    counts = list(tooth_counts.values())

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10)) # Create a wide figure to fit all 32 teeth
    
    bars = ax.bar(fdi_labels, counts, color='skyblue', edgecolor='black')

    # Add titles and labels for clarity
    ax.set_title('Test Set Tooth Frequency Distribution (FDI Notation)', fontsize=20, fontweight='bold')
    ax.set_xlabel('FDI Tooth ID', fontsize=14)
    ax.set_ylabel('Number of Appearances (Count)', fontsize=14)
    
    # Rotate x-axis labels to prevent them from overlapping
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add a horizontal grid for easier reading of values
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add the exact count number on top of each bar
    ax.bar_label(bars, padding=3, fontsize=10)
    
    # Set the y-axis limit to give some space above the tallest bar
    ax.set_ylim(0, max(counts) * 1.1)

    # Ensure everything fits without being cut off
    fig.tight_layout()

    # --- 4. Save the plot to a file ---
    try:
        plt.savefig(save_path, dpi=300) # Save as a high-resolution image
        print(f"\n Frequency plot saved successfully to: {save_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save the plot: {e}")
    
    plt.close() # Close the plot to free up memory


if __name__ == "__main__":
    # Define the full path for the output image file
    output_plot_path = Path(OUTPUT_DIR) / "test_set_frequency_plot.png"
    
    # Run the analysis and plotting function
    analyze_and_plot_test_set_frequency(TEST_LABELS_CSV, save_path=output_plot_path)