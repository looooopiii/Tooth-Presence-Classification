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

def get_jaw_type(tooth_id):
    """
    Determine which jaw a tooth belongs to based on FDI notation.
    Returns 'upper' for teeth 11-18 and 21-28, 'lower' for teeth 31-38 and 41-48.
    """
    tooth_num = int(tooth_id)
    if 11 <= tooth_num <= 18 or 21 <= tooth_num <= 28:
        return 'upper'
    elif 31 <= tooth_num <= 38 or 41 <= tooth_num <= 48:
        return 'lower'
    else:
        return 'unknown'

def main():
    """Main function to count teeth, save data, and plot distributions - CORRECTED VERSION."""
    output_dir = '/home/user/lzhou/week15-32/output/check/train'
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Counting ---
    upper_counts, upper_samples = count_teeth(JSON_ROOT_UPPER)
    lower_counts, lower_samples = count_teeth(JSON_ROOT_LOWER)
    
    print(f"Upper jaw samples: {upper_samples}")
    print(f"Lower jaw samples: {lower_samples}")
    
    # Get all unique tooth IDs
    all_tooth_ids = set(upper_counts.keys()) | set(lower_counts.keys())
    
    if not all_tooth_ids:
        print("Could not find or access data. Generating dummy data for demonstration.")
        dummy_ids = list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48))
        upper_counts = {str(i): np.random.randint(150, 250) for i in dummy_ids if 11 <= i <= 28}
        lower_counts = {str(i): np.random.randint(150, 250) for i in dummy_ids if 31 <= i <= 48}
        all_tooth_ids = set(upper_counts.keys()) | set(lower_counts.keys())
        upper_samples = 300
        lower_samples = 300

    # Sort tooth IDs
    tooth_ids_sorted = sorted(all_tooth_ids, key=int)
    
    # CORRECTED: Calculate presence and absence based on the CORRECT jaw
    results = []
    for tooth_id in tooth_ids_sorted:
        jaw_type = get_jaw_type(tooth_id)
        
        if jaw_type == 'upper':
            # For upper jaw teeth, use upper jaw sample count
            presence_count = upper_counts.get(tooth_id, 0)
            total_samples_for_this_tooth = upper_samples
        elif jaw_type == 'lower':
            # For lower jaw teeth, use lower jaw sample count
            presence_count = lower_counts.get(tooth_id, 0)
            total_samples_for_this_tooth = lower_samples
        else:
            # Unknown tooth ID
            presence_count = upper_counts.get(tooth_id, 0) + lower_counts.get(tooth_id, 0)
            total_samples_for_this_tooth = upper_samples + lower_samples
        
        absence_count = total_samples_for_this_tooth - presence_count
        
        results.append({
            'ToothID': tooth_id,
            'Jaw': jaw_type,
            'Present_Count': presence_count,
            'Absent_Count': absence_count,
            'Total_Samples': total_samples_for_this_tooth,
            'Presence_Rate': f"{presence_count/total_samples_for_this_tooth*100:.2f}%"
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # --- Save data to CSV ---
    csv_path = os.path.join(output_dir, 'tooth_counts_corrected.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nCorrected data table saved to {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY: Teeth with High Absence Rate (>10%)")
    print("="*80)
    high_absence = df[df['Absent_Count'] / df['Total_Samples'] > 0.1].sort_values('Absent_Count', ascending=False)
    print(high_absence[['ToothID', 'Jaw', 'Present_Count', 'Absent_Count', 'Presence_Rate']].to_string(index=False))
    
    # --- Plotting ---
    presence = df['Present_Count'].tolist()
    absence = df['Absent_Count'].tolist()
    
    # Plot 1: Stacked Bar Chart showing the corrected distribution
    fig_stacked, ax_stacked = plt.subplots(figsize=(16, 8))
    x = np.arange(len(tooth_ids_sorted))
    width = 0.6
    
    bars_present = ax_stacked.bar(x, presence, width, label='Present', color='green', alpha=0.8)
    bars_absent = ax_stacked.bar(x, absence, width, bottom=presence, label='Absent', color='red', alpha=0.8)
    
    # Add jaw separation lines
    upper_end = len([t for t in tooth_ids_sorted if get_jaw_type(t) == 'upper'])
    if upper_end < len(tooth_ids_sorted):
        ax_stacked.axvline(x=upper_end - 0.5, color='blue', linestyle='--', linewidth=2, 
                          label='Upper/Lower Jaw Boundary')
    
    ax_stacked.set_ylabel('Count', fontsize=12)
    ax_stacked.set_title('Corrected Tooth Presence and Absence in Training Set\n(Separated by Jaw)', fontsize=14, fontweight='bold')
    ax_stacked.set_xticks(x)
    ax_stacked.set_xticklabels(tooth_ids_sorted, rotation=90, fontsize=10)
    ax_stacked.set_xlabel('Tooth ID (FDI Notation)', fontsize=12)
    ax_stacked.legend(fontsize=11)
    ax_stacked.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add text annotations for jaw labels
    if upper_end > 0:
        ax_stacked.text(upper_end/2 - 0.5, ax_stacked.get_ylim()[1] * 0.95, 'UPPER JAW', 
                       ha='center', fontsize=12, fontweight='bold', color='blue')
    if upper_end < len(tooth_ids_sorted):
        ax_stacked.text((upper_end + len(tooth_ids_sorted))/2 - 0.5, ax_stacked.get_ylim()[1] * 0.95, 
                       'LOWER JAW', ha='center', fontsize=12, fontweight='bold', color='blue')
    
    fig_stacked.tight_layout()
    stacked_plot_path = os.path.join(output_dir, 'corrected_stacked_distribution.png')
    fig_stacked.savefig(stacked_plot_path, dpi=300)
    print(f"Corrected stacked bar chart saved to {stacked_plot_path}")
    plt.close(fig_stacked)

    # Plot 2: Presence Rate as Percentage
    fig_rate, ax_rate = plt.subplots(figsize=(16, 8))
    presence_rate = [(p / (p + a) * 100) for p, a in zip(presence, absence)]
    
    colors = ['green' if rate > 90 else 'orange' if rate > 70 else 'red' for rate in presence_rate]
    ax_rate.bar(x, presence_rate, color=colors, alpha=0.7)
    ax_rate.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    ax_rate.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    
    ax_rate.set_ylabel('Presence Rate (%)', fontsize=12)
    ax_rate.set_title('Tooth Presence Rate in Training Set (Corrected)', fontsize=14, fontweight='bold')
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels(tooth_ids_sorted, rotation=90, fontsize=10)
    ax_rate.set_xlabel('Tooth ID (FDI Notation)', fontsize=12)
    ax_rate.set_ylim(0, 105)
    ax_rate.legend(fontsize=11)
    ax_rate.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig_rate.tight_layout()
    rate_plot_path = os.path.join(output_dir, 'corrected_presence_rate.png')
    fig_rate.savefig(rate_plot_path, dpi=300)
    print(f"Corrected presence rate chart saved to {rate_plot_path}")
    plt.close(fig_rate)
    
    # Plot 3: Comparison - Before vs After Correction (Conceptual)
    # This shows why the correction was necessary
    fig_compare, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # "Before" - incorrect calculation
    total_all = upper_samples + lower_samples
    incorrect_absence = [total_all - (upper_counts.get(tid, 0) + lower_counts.get(tid, 0)) 
                        for tid in tooth_ids_sorted]
    incorrect_presence = [upper_counts.get(tid, 0) + lower_counts.get(tid, 0) 
                         for tid in tooth_ids_sorted]
    
    ax1.bar(x, incorrect_presence, width, label='Present', color='green', alpha=0.6)
    ax1.bar(x, incorrect_absence, width, bottom=incorrect_presence, label='Absent', color='red', alpha=0.6)
    ax1.set_title('BEFORE Correction\n(Incorrect: All jaws counted for all teeth)', fontsize=12, fontweight='bold', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tooth_ids_sorted, rotation=90, fontsize=9)
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # "After" - correct calculation
    ax2.bar(x, presence, width, label='Present', color='green', alpha=0.8)
    ax2.bar(x, absence, width, bottom=presence, label='Absent', color='red', alpha=0.8)
    ax2.set_title('AFTER Correction\n(Correct: Upper teeth use upper samples, lower teeth use lower samples)', 
                 fontsize=12, fontweight='bold', color='green')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tooth_ids_sorted, rotation=90, fontsize=9)
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig_compare.suptitle('Comparison: Impact of Jaw-Specific Counting', fontsize=14, fontweight='bold')
    fig_compare.tight_layout()
    compare_plot_path = os.path.join(output_dir, 'before_after_comparison.png')
    fig_compare.savefig(compare_plot_path, dpi=300)
    print(f"Before/After comparison chart saved to {compare_plot_path}")
    plt.close(fig_compare)
    
    print(f"\n{'='*80}")
    print("Analysis complete! All corrected plots and data saved.")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
