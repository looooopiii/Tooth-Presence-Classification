import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def analyze_test_set_corrected():
    """
    CORRECTED VERSION: Analyzes tooth presence/absence with jaw-specific counting.
    
    Uses filename to separate UpperJawScan and LowerJawScan, then:
    - Upper teeth (11-28) counted against UpperJawScan rows only
    - Lower teeth (31-48) counted against LowerJawScan rows only
    """
    input_csv = '/home/user/lzhou/week10/label_flipped.csv'
    output_dir = '/home/user/lzhou/week15-32/output/check/test'

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        return

    print("="*80)
    print(" TEST SET ANALYSIS - CORRECTED VERSION")
    print("="*80)
    
    # --- Separate by jaw type using filename ---
    upper_df = df[df['filename'].str.contains('UpperJaw', case=False, na=False)].copy()
    lower_df = df[df['filename'].str.contains('LowerJaw', case=False, na=False)].copy()
    
    upper_count = len(upper_df)
    lower_count = len(lower_df)
    total_count = len(df)
    
    print(f"\n Data Structure:")
    print(f"   Total rows: {total_count}")
    print(f"   UpperJawScan rows: {upper_count}")
    print(f"   LowerJawScan rows: {lower_count}")
    
    # --- Identify tooth columns ---
    tooth_columns = [col for col in df.columns if str(col).isdigit()]
    tooth_ids = sorted([int(c) for c in tooth_columns])
    
    print(f"\n Tooth Columns:")
    print(f"   Total tooth columns: {len(tooth_columns)}")
    print(f"   Tooth IDs: {tooth_ids}")
    
    # --- CORRECTED: Calculate statistics with JAW-SPECIFIC counting ---
    print(f"\n  Applying jaw-specific counting...")
    
    results = []
    for tooth_id in tooth_columns:
        tooth_num = int(tooth_id)
        jaw_type = get_jaw_type(tooth_id)
        
        if jaw_type == 'upper':
            # For upper teeth (11-28), only count UpperJawScan rows
            relevant_data = upper_df[tooth_id]
            total_samples_for_tooth = upper_count
        elif jaw_type == 'lower':
            # For lower teeth (31-48), only count LowerJawScan rows
            relevant_data = lower_df[tooth_id]
            total_samples_for_tooth = lower_count
        else:
            # Unknown tooth type (shouldn't happen with FDI notation)
            relevant_data = df[tooth_id]
            total_samples_for_tooth = total_count
        
        # 1 = missing, 0 = present
        absent_count = int(relevant_data.sum())
        present_count = total_samples_for_tooth - absent_count
        
        results.append({
            'ToothID': tooth_num,
            'Jaw': jaw_type,
            'Present_Count': present_count,
            'Absent_Count': absent_count,
            'Total_Samples': total_samples_for_tooth,
            'Present_%': f"{present_count/total_samples_for_tooth*100:.1f}%",
            'Absent_%': f"{absent_count/total_samples_for_tooth*100:.1f}%"
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ToothID')
    
    # --- Save data to CSV ---
    csv_path = os.path.join(output_dir, 'tooth_counts_corrected.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n Corrected data saved to: {csv_path}")
    
    # --- Print summary ---
    print("\n" + "="*80)
    print(" CORRECTED STATISTICS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Identify highly missing teeth
    print("\n" + "="*80)
    print(" TEETH WITH HIGH ABSENCE RATE (>20%)")
    print("="*80)
    high_absence = results_df[results_df['Absent_Count']/results_df['Total_Samples'] > 0.2]
    if len(high_absence) > 0:
        print(high_absence[['ToothID', 'Jaw', 'Absent_Count', 'Total_Samples', 'Absent_%']].to_string(index=False))
    else:
        print("No teeth with absence rate > 20%")

    # --- Prepare data for plotting ---
    tooth_ids_list = results_df['ToothID'].tolist()
    presence = results_df['Present_Count'].tolist()
    absence = results_df['Absent_Count'].tolist()
    
    # --- Plotting ---
    
    # Plot 1: Stacked Bar Chart with Jaw Separation
    fig_stacked, ax_stacked = plt.subplots(figsize=(16, 8))
    x = np.arange(len(tooth_ids_list))
    width = 0.6
    
    bars_present = ax_stacked.bar(x, presence, width, label='Present (0)', color='green', alpha=0.8)
    bars_absent = ax_stacked.bar(x, absence, width, bottom=presence, label='Absent (1)', color='red', alpha=0.8)
    
    # Add jaw separation line
    upper_end = len([t for t in tooth_ids_list if get_jaw_type(str(t)) == 'upper'])
    if upper_end > 0 and upper_end < len(tooth_ids_list):
        ax_stacked.axvline(x=upper_end - 0.5, color='blue', linestyle='--', 
                          linewidth=2, label='Upper/Lower Jaw Boundary')
    
    ax_stacked.set_ylabel('Count', fontsize=12)
    ax_stacked.set_title('CORRECTED: Tooth Presence and Absence in Test Set\n(Jaw-Specific Counting)', 
                        fontsize=14, fontweight='bold')
    ax_stacked.set_xticks(x)
    ax_stacked.set_xticklabels(tooth_ids_list, rotation=90, fontsize=10)
    ax_stacked.set_xlabel('Tooth ID (FDI Notation)', fontsize=12)
    ax_stacked.legend(fontsize=11, loc='upper right')
    ax_stacked.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add jaw labels
    if upper_end > 0:
        ax_stacked.text(upper_end/2 - 0.5, ax_stacked.get_ylim()[1] * 0.95, 
                       'UPPER JAW', ha='center', fontsize=12, fontweight='bold', 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    if upper_end < len(tooth_ids_list):
        ax_stacked.text((upper_end + len(tooth_ids_list))/2 - 0.5, 
                       ax_stacked.get_ylim()[1] * 0.95, 
                       'LOWER JAW', ha='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    fig_stacked.tight_layout()
    stacked_plot_path = os.path.join(output_dir, 'stacked_distribution_corrected.png')
    fig_stacked.savefig(stacked_plot_path, dpi=300)
    print(f"\n Stacked bar chart saved to: {stacked_plot_path}")
    plt.close(fig_stacked)

    # Plot 2: Absence Rate Percentage
    fig_rate, ax_rate = plt.subplots(figsize=(16, 8))
    absence_rate = [(a / (p + a) * 100) if (p + a) > 0 else 0 for p, a in zip(presence, absence)]
    
    colors = ['darkred' if rate > 50 else 'orange' if rate > 20 else 'green' for rate in absence_rate]
    bars = ax_rate.bar(x, absence_rate, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax_rate.axhline(y=50, color='darkred', linestyle='--', alpha=0.5, linewidth=2, label='50% threshold')
    ax_rate.axhline(y=20, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='20% threshold')
    
    # Add jaw separation
    if upper_end > 0 and upper_end < len(tooth_ids_list):
        ax_rate.axvline(x=upper_end - 0.5, color='blue', linestyle='--', linewidth=2)
    
    ax_rate.set_ylabel('Absence Rate (%)', fontsize=12)
    ax_rate.set_title('Tooth Absence Rate in Test Set (Corrected)', 
                     fontsize=14, fontweight='bold')
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels(tooth_ids_list, rotation=90, fontsize=10)
    ax_rate.set_xlabel('Tooth ID (FDI Notation)', fontsize=12)
    ax_rate.set_ylim(0, 105)
    ax_rate.legend(fontsize=11)
    ax_rate.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig_rate.tight_layout()
    rate_plot_path = os.path.join(output_dir, 'absence_rate_corrected.png')
    fig_rate.savefig(rate_plot_path, dpi=300)
    print(f" Absence rate chart saved to: {rate_plot_path}")
    plt.close(fig_rate)
    
    # Plot 3: Comparison - Before vs After (Conceptual)
    print(f"\n Generating before/after comparison...")

    # Calculate "incorrect" statistics for comparison
    incorrect_stats = []
    for tooth_id in tooth_columns:
        # Wrong way: use all 186 rows for every tooth
        absent_count_wrong = int(df[tooth_id].sum())
        present_count_wrong = total_count - absent_count_wrong
        incorrect_stats.append({
            'tooth': int(tooth_id),
            'present': present_count_wrong,
            'absent': absent_count_wrong
        })
    
    fig_compare, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left: BEFORE (incorrect)
    incorrect_df = pd.DataFrame(incorrect_stats).sort_values('tooth')
    x_wrong = np.arange(len(incorrect_df))
    ax1.bar(x_wrong, incorrect_df['present'], width, label='Present', color='green', alpha=0.6)
    ax1.bar(x_wrong, incorrect_df['absent'], width, bottom=incorrect_df['present'], 
           label='Absent', color='red', alpha=0.6)
    ax1.set_title('BEFORE Correction\n All 186 rows used for every tooth', 
                 fontsize=12, fontweight='bold', color='darkred')
    ax1.set_xticks(x_wrong)
    ax1.set_xticklabels(incorrect_df['tooth'].tolist(), rotation=90, fontsize=9)
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Tooth ID')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.axhline(y=total_count, color='gray', linestyle=':', label=f'Total: {total_count}')
    
    # Right: AFTER (correct)
    ax2.bar(x, presence, width, label='Present', color='green', alpha=0.8)
    ax2.bar(x, absence, width, bottom=presence, label='Absent', color='red', alpha=0.8)
    if upper_end > 0 and upper_end < len(tooth_ids_list):
        ax2.axvline(x=upper_end - 0.5, color='blue', linestyle='--', linewidth=2)
    ax2.set_title('AFTER Correction\n Jaw-specific counting applied', 
                 fontsize=12, fontweight='bold', color='darkgreen')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tooth_ids_list, rotation=90, fontsize=9)
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Tooth ID')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig_compare.suptitle('Impact of Jaw-Specific Correction on Test Set Statistics', 
                        fontsize=14, fontweight='bold')
    fig_compare.tight_layout()
    compare_plot_path = os.path.join(output_dir, 'before_after_comparison.png')
    fig_compare.savefig(compare_plot_path, dpi=300)
    print(f" Comparison chart saved to: {compare_plot_path}")
    plt.close(fig_compare)
    
    print(f"\n{'='*80}")
    print(" CORRECTED ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"All files saved to: {output_dir}/")
    print(f"\nKey files:")
    print(f"   {csv_path}")
    print(f"   {stacked_plot_path}")
    print(f"   {rate_plot_path}")
    print(f"   {compare_plot_path}")
    print("="*80 + "\n")

if __name__ == '__main__':
    analyze_test_set_corrected()