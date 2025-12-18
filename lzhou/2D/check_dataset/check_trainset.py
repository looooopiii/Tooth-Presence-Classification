
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the paths to the training data
JSON_ROOT_LOWER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower"
JSON_ROOT_UPPER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"

def count_teeth(json_root):
    """Counts the presence of each tooth in the given JSON root directory based on the 'labels' field."""
    tooth_presence = {}  # Use a dictionary to store counts for any tooth id
    num_samples = 0
    
    if not os.path.isdir(json_root):
        print(f"Directory not found: {json_root}, skipping.")
        return tooth_presence, num_samples

    for case_folder in os.listdir(json_root):
        case_path = os.path.join(json_root, case_folder)
        if os.path.isdir(case_path):
            json_files = [f for f in os.listdir(case_path) if f.endswith('.json')]
            if not json_files:
                continue

            # Assuming one json per folder as per the example
            json_file_path = os.path.join(case_path, json_files[0])
            
            num_samples += 1
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                if 'labels' in data:
                    present_teeth_in_sample = set(data['labels'])
                    for tooth_id in present_teeth_in_sample:
                        if tooth_id != 0:
                            tooth_id_str = str(tooth_id)
                            tooth_presence[tooth_id_str] = tooth_presence.get(tooth_id_str, 0) + 1
    return tooth_presence, num_samples

def main():
    """Main function to count teeth, save data, and plot distributions."""
    # Create output directory if it doesn't exist
    output_dir = '/home/user/lzhou/week13-17/scripts/set_output/train_set'
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Counting ---
    upper_counts, upper_samples = count_teeth(JSON_ROOT_UPPER)
    lower_counts, lower_samples = count_teeth(JSON_ROOT_LOWER)
    total_samples = upper_samples + lower_samples
    
    all_tooth_ids = set(upper_counts.keys()) | set(lower_counts.keys())
    
    tooth_presence_counts = {
        tooth_id: upper_counts.get(tooth_id, 0) + lower_counts.get(tooth_id, 0)
        for tooth_id in all_tooth_ids
    }
    
    if not tooth_presence_counts:
        print("Could not find or access data. Generating dummy data for demonstration.")
        # FDI-like tooth numbers for dummy data
        dummy_ids = list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48))
        tooth_presence_counts = {str(i): np.random.randint(150, 250) for i in dummy_ids}
        total_samples = 300 # Assume 300 samples for dummy data
        all_tooth_ids = tooth_presence_counts.keys()

    tooth_ids_sorted = sorted(all_tooth_ids, key=int)
    
    presence = [tooth_presence_counts.get(tid, 0) for tid in tooth_ids_sorted]
    absence = [total_samples - tooth_presence_counts.get(tid, 0) for tid in tooth_ids_sorted]

    # --- Save data to CSV ---
    df = pd.DataFrame({
        'ToothID': tooth_ids_sorted,
        'Present_Count': presence,
        'Absent_Count': absence,
        'Total_Samples': total_samples
    })
    csv_path = os.path.join(output_dir, 'tooth_counts.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data table saved to {csv_path}")

    # --- Plotting ---
    
    # Plot 1: Presence
    fig1, ax1 = plt.subplots(figsize=(15, 7))
    ax1.bar(np.arange(len(tooth_ids_sorted)), presence, color='green')
    ax1.set_ylabel('Count')
    ax1.set_title('Tooth Presence in Training Set')
    ax1.set_xticks(np.arange(len(tooth_ids_sorted)))
    ax1.set_xticklabels(tooth_ids_sorted, rotation=90)
    ax1.set_xlabel('Tooth ID (FDI Notation)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    fig1.tight_layout()
    presence_plot_path = os.path.join(output_dir, 'tooth_presence.png')
    fig1.savefig(presence_plot_path)
    print(f"Presence bar chart saved to {presence_plot_path}")
    plt.close(fig1)

    # Plot 2: Absence
    fig2, ax2 = plt.subplots(figsize=(15, 7))
    ax2.bar(np.arange(len(tooth_ids_sorted)), absence, color='red')
    ax2.set_ylabel('Count')
    ax2.set_title('Tooth Absence in Training Set')
    ax2.set_xticks(np.arange(len(tooth_ids_sorted)))
    ax2.set_xticklabels(tooth_ids_sorted, rotation=90)
    ax2.set_xlabel('Tooth ID (FDI Notation)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    fig2.tight_layout()
    absence_plot_path = os.path.join(output_dir, 'tooth_absence.png')
    fig2.savefig(absence_plot_path)
    print(f"Absence bar chart saved to {absence_plot_path}")
    plt.close(fig2)

if __name__ == '__main__':
    main()
