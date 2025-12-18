
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_test_set():
    """
    Analyzes the tooth presence and absence from the test set CSV file,
    saves the statistics, and plots the distributions.
    """
    input_csv = '/home/user/lzhou/week10/label_flipped.csv'
    output_dir = '/home/user/lzhou/week13-17/scripts/set_output/test_set'

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        return

    # --- Data Processing ---
    # Identify tooth columns (assuming they are all numeric strings)
    tooth_columns = [col for col in df.columns if col.isdigit()]
    
    total_samples = len(df)
    
    if total_samples == 0:
        print("Warning: The CSV file is empty.")
        return

    results = []
    for tooth_id in tooth_columns:
        # 1 means missing, 0 means present
        absent_count = df[tooth_id].sum()
        present_count = total_samples - absent_count
        results.append({
            'ToothID': tooth_id,
            'Present_Count': present_count,
            'Absent_Count': absent_count,
        })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    results_df['Total_Samples'] = total_samples
    
    # Sort by tooth ID
    results_df['ToothID_int'] = results_df['ToothID'].astype(int)
    results_df = results_df.sort_values('ToothID_int').drop(columns=['ToothID_int'])


    # --- Save data to CSV ---
    csv_path = os.path.join(output_dir, 'tooth_counts.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Data table saved to {csv_path}")

    # Prepare data for plotting
    tooth_ids_sorted = results_df['ToothID'].tolist()
    presence = results_df['Present_Count'].tolist()
    absence = results_df['Absent_Count'].tolist()
    
    # --- Plotting ---

    # Plot 1: Stacked Bar Chart
    fig_stacked, ax_stacked = plt.subplots(figsize=(15, 7))
    x = np.arange(len(tooth_ids_sorted))
    width = 0.5
    ax_stacked.bar(x, presence, width, label='Present (0)')
    ax_stacked.bar(x, absence, width, bottom=presence, label='Absent (1)')
    ax_stacked.set_ylabel('Count')
    ax_stacked.set_title('Stacked Tooth Presence and Absence in Test Set')
    ax_stacked.set_xticks(x)
    ax_stacked.set_xticklabels(tooth_ids_sorted, rotation=90)
    ax_stacked.set_xlabel('Tooth ID (FDI Notation)')
    ax_stacked.axhline(y=total_samples, color='gray', linestyle='--', label=f'Total Samples: {total_samples}')
    ax_stacked.legend()
    fig_stacked.tight_layout()
    stacked_plot_path = os.path.join(output_dir, 'stacked_distribution.png')
    fig_stacked.savefig(stacked_plot_path)
    print(f"Stacked bar chart saved to {stacked_plot_path}")
    plt.close(fig_stacked)

    # Plot 2: Presence
    fig_presence, ax_presence = plt.subplots(figsize=(15, 7))
    ax_presence.bar(x, presence, color='green')
    ax_presence.set_ylabel('Count')
    ax_presence.set_title('Tooth Presence in Test Set')
    ax_presence.set_xticks(x)
    ax_presence.set_xticklabels(tooth_ids_sorted, rotation=90)
    ax_presence.set_xlabel('Tooth ID (FDI Notation)')
    ax_presence.grid(axis='y', linestyle='--', alpha=0.7)
    fig_presence.tight_layout()
    presence_plot_path = os.path.join(output_dir, 'tooth_presence.png')
    fig_presence.savefig(presence_plot_path)
    print(f"Presence bar chart saved to {presence_plot_path}")
    plt.close(fig_presence)

    # Plot 3: Absence
    fig_absence, ax_absence = plt.subplots(figsize=(15, 7))
    ax_absence.bar(x, absence, color='red')
    ax_absence.set_ylabel('Count')
    ax_absence.set_title('Tooth Absence in Test Set')
    ax_absence.set_xticks(x)
    ax_absence.set_xticklabels(tooth_ids_sorted, rotation=90)
    ax_absence.set_xlabel('Tooth ID (FDI Notation)')
    ax_absence.grid(axis='y', linestyle='--', alpha=0.7)
    fig_absence.tight_layout()
    absence_plot_path = os.path.join(output_dir, 'tooth_absence.png')
    fig_absence.savefig(absence_plot_path)
    print(f"Absence bar chart saved to {absence_plot_path}")
    plt.close(fig_absence)

if __name__ == '__main__':
    analyze_test_set()
