import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============= CONFIGURATION =============
AUGMENTED_CSV = "/home/user/lzhou/week16/Aug/augment_test/train_labels_augmented.csv"
RANDOM_CSV = "/home/user/lzhou/week16/Aug/augment_random/train_labels_random.csv"
OUTPUT_DIR = Path("/home/user/lzhou/week16-32/output/check/Aug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FDI Tooth Notation
UPPER_TEETH = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_TEETH = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
ALL_TEETH = sorted(UPPER_TEETH + LOWER_TEETH)

def get_jaw_type(tooth_id):
    """Determine jaw type from tooth ID"""
    if tooth_id in UPPER_TEETH:
        return 'upper'
    elif tooth_id in LOWER_TEETH:
        return 'lower'
    return 'unknown'

def analyze_with_jaw_correction(csv_path, dataset_name):
    """
     CORRECTED: Analyzes augmented data with jaw-specific counting
    - Upper teeth (11-28) only count against upper jaw rows
    - Lower teeth (31-48) only count against lower jaw rows
    """
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*80}")
    print(f" {dataset_name} Analysis (Corrected)")
    print(f"{'='*80}")
    
    # Separate by jaw using filename
    upper_df = df[df['filename'].str.contains('upper', case=False)].copy()
    lower_df = df[df['filename'].str.contains('lower', case=False)].copy()
    
    upper_count = len(upper_df)
    lower_count = len(lower_df)
    total_count = len(df)
    
    print(f"\n Dataset Structure:")
    print(f"   Total samples: {total_count}")
    print(f"   Upper jaw samples: {upper_count}")
    print(f"   Lower jaw samples: {lower_count}")
    
    # Tooth columns (as strings)
    tooth_cols = [str(t) for t in ALL_TEETH]
    
    #  CORRECTED: Jaw-specific counting
    results = []
    for tooth_str in tooth_cols:
        tooth_id = int(tooth_str)
        jaw = get_jaw_type(tooth_id)
        
        if jaw == 'upper':
            relevant_data = upper_df[tooth_str]
            total_samples = upper_count
        elif jaw == 'lower':
            relevant_data = lower_df[tooth_str]
            total_samples = lower_count
        else:
            relevant_data = df[tooth_str]
            total_samples = total_count
        
        # 1 = missing, 0 = present
        missing_count = int(relevant_data.sum())
        present_count = total_samples - missing_count
        
        missing_rate = missing_count / total_samples * 100 if total_samples > 0 else 0
        present_rate = present_count / total_samples * 100 if total_samples > 0 else 0
        
        results.append({
            'Tooth': tooth_id,
            'Jaw': jaw,
            'Missing': missing_count,
            'Present': present_count,
            'Total': total_samples,
            'Missing_%': missing_rate,
            'Present_%': present_rate
        })
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print(f"\n Per-Tooth Statistics:")
    print(results_df.to_string(index=False))
    
    # Balance analysis
    print(f"\n  Balance Analysis:")
    print(f"   Average missing rate: {results_df['Missing_%'].mean():.1f}%")
    print(f"   Std deviation: {results_df['Missing_%'].std():.1f}%")
    print(f"   Min missing rate: {results_df['Missing_%'].min():.1f}% (Tooth {results_df.loc[results_df['Missing_%'].idxmin(), 'Tooth']})")
    print(f"   Max missing rate: {results_df['Missing_%'].max():.1f}% (Tooth {results_df.loc[results_df['Missing_%'].idxmax(), 'Tooth']})")
    
    # Imbalance classification
    highly_imbalanced = results_df[
        (results_df['Missing_%'] < 20) | (results_df['Missing_%'] > 80)
    ]
    moderately_imbalanced = results_df[
        ((results_df['Missing_%'] >= 20) & (results_df['Missing_%'] < 30)) |
        ((results_df['Missing_%'] > 70) & (results_df['Missing_%'] <= 80))
    ]
    balanced = results_df[
        (results_df['Missing_%'] >= 30) & (results_df['Missing_%'] <= 70)
    ]
    
    print(f"\n   Highly imbalanced (<20% or >80%): {len(highly_imbalanced)} teeth")
    print(f"   Moderately imbalanced (20-30% or 70-80%): {len(moderately_imbalanced)} teeth")
    print(f"   Balanced (30-70%): {len(balanced)} teeth")
    
    return results_df

def compare_datasets(df1, name1, df2, name2):
    """Compare two datasets side by side"""
    print(f"\n{'='*80}")
    print(f" Comparison: {name1} vs {name2}")
    print(f"{'='*80}")
    
    comparison = pd.DataFrame({
        'Tooth': df1['Tooth'],
        'Jaw': df1['Jaw'],
        f'{name1}_Missing%': df1['Missing_%'],
        f'{name2}_Missing%': df2['Missing_%'],
        'Difference': df2['Missing_%'] - df1['Missing_%']
    })
    
    print(f"\n Side-by-Side Comparison:")
    print(comparison.to_string(index=False))
    
    print(f"\n Key Insights:")
    improved = comparison[comparison['Difference'] > 0]
    worsened = comparison[comparison['Difference'] < 0]
    
    print(f"   Teeth with MORE missing samples in {name2}: {len(improved)} ({len(improved)/len(comparison)*100:.1f}%)")
    print(f"   Teeth with FEWER missing samples in {name2}: {len(worsened)} ({len(worsened)/len(comparison)*100:.1f}%)")
    print(f"   Average change: {comparison['Difference'].mean():.1f}%")
    
    return comparison

def visualize_all(df_aug, df_rand, comparison):
    """Create comprehensive visualizations"""
    
    # Plot 1: Stacked bar chart for both datasets
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    
    for idx, (df, name, ax) in enumerate([
        (df_aug, 'Augmented (Test-Pattern Based)', axes[0]),
        (df_rand, 'Random Augmented', axes[1])
    ]):
        x = np.arange(len(df))
        width = 0.6
        
        # Color by jaw
        colors_present = ['lightgreen' if jaw == 'upper' else 'lightblue' 
                         for jaw in df['Jaw']]
        colors_missing = ['darkred' if jaw == 'upper' else 'darkblue' 
                         for jaw in df['Jaw']]
        
        ax.bar(x, df['Present'], width, label='Present', color=colors_present, alpha=0.8)
        ax.bar(x, df['Missing'], width, bottom=df['Present'], 
               label='Missing', color=colors_missing, alpha=0.8)
        
        # Add jaw separation
        upper_end = len([t for t in df['Tooth'] if get_jaw_type(t) == 'upper'])
        if upper_end > 0 and upper_end < len(df):
            ax.axvline(x=upper_end - 0.5, color='purple', linestyle='--', 
                      linewidth=2, label='Jaw Boundary')
        
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f' CORRECTED: {name}\n(Jaw-Specific Counting Applied)', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Tooth'].tolist(), rotation=90, fontsize=9)
        ax.set_xlabel('Tooth ID (FDI Notation)', fontsize=11)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add jaw labels
        if upper_end > 0:
            ax.text(upper_end/2 - 0.5, ax.get_ylim()[1] * 0.95, 
                   'UPPER JAW', ha='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        if upper_end < len(df):
            ax.text((upper_end + len(df))/2 - 0.5, ax.get_ylim()[1] * 0.95, 
                   'LOWER JAW', ha='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    fig.tight_layout()
    plot1_path = OUTPUT_DIR / 'augmentation_comparison_stacked.png'
    fig.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"\n Saved: {plot1_path}")
    plt.close()
    
    # Plot 2: Missing rate percentage comparison
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    
    for idx, (df, name, ax, target) in enumerate([
        (df_aug, 'Augmented', axes[0], 50),
        (df_rand, 'Random', axes[1], 50)
    ]):
        x = np.arange(len(df))
        
        # Color by balance quality
        colors = []
        for rate in df['Missing_%']:
            if rate < 20 or rate > 80:
                colors.append('darkred')  # Highly imbalanced
            elif (rate >= 20 and rate < 30) or (rate > 70 and rate <= 80):
                colors.append('orange')   # Moderately imbalanced
            else:
                colors.append('green')    # Balanced
        
        ax.bar(x, df['Missing_%'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Reference lines
        ax.axhline(y=50, color='green', linestyle='-', linewidth=2, alpha=0.5, label='Perfect balance (50%)')
        ax.axhline(y=30, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='30% threshold')
        ax.axhline(y=70, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='70% threshold')
        ax.axhline(y=20, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='20% threshold')
        ax.axhline(y=80, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='80% threshold')
        
        # Jaw boundary
        upper_end = len([t for t in df['Tooth'] if get_jaw_type(t) == 'upper'])
        if upper_end > 0 and upper_end < len(df):
            ax.axvline(x=upper_end - 0.5, color='purple', linestyle='--', linewidth=2)
        
        ax.set_ylabel('Missing Rate (%)', fontsize=11)
        ax.set_title(f'{name} Dataset - Missing Rate per Tooth', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Tooth'].tolist(), rotation=90, fontsize=9)
        ax.set_xlabel('Tooth ID (FDI Notation)', fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    plot2_path = OUTPUT_DIR / 'missing_rate_comparison.png'
    fig.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {plot2_path}")
    plt.close()
    
    # Plot 3: Direct comparison bar chart
    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(len(comparison))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison['Augmented_Missing%'], width, 
                   label='Augmented (Test-Pattern)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, comparison['Random_Missing%'], width, 
                   label='Random Augmented', color='coral', alpha=0.8)
    
    # Reference line for perfect balance
    ax.axhline(y=50, color='green', linestyle='--', linewidth=2, alpha=0.5, 
              label='Perfect Balance (50%)')
    
    # Jaw boundary
    upper_end = len([t for t in comparison['Tooth'] if get_jaw_type(t) == 'upper'])
    if upper_end > 0 and upper_end < len(comparison):
        ax.axvline(x=upper_end - 0.5, color='purple', linestyle='--', 
                  linewidth=2, label='Jaw Boundary')
    
    ax.set_ylabel('Missing Rate (%)', fontsize=12)
    ax.set_title(' Direct Comparison: Augmented vs Random Augmented\n(Corrected Jaw-Specific Counting)', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['Tooth'].tolist(), rotation=90, fontsize=9)
    ax.set_xlabel('Tooth ID (FDI Notation)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    plot3_path = OUTPUT_DIR / 'direct_comparison.png'
    fig.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {plot3_path}")
    plt.close()
    
    # Plot 4: Histogram of missing rates
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (df, name, ax) in enumerate([
        (df_aug, 'Augmented', axes[0]),
        (df_rand, 'Random', axes[1])
    ]):
        ax.hist(df['Missing_%'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=50, color='green', linestyle='--', linewidth=2, label='Perfect Balance')
        ax.axvline(x=df['Missing_%'].mean(), color='red', linestyle='-', linewidth=2, 
                  label=f'Mean: {df["Missing_%"].mean():.1f}%')
        
        ax.set_xlabel('Missing Rate (%)', fontsize=11)
        ax.set_ylabel('Number of Teeth', fontsize=11)
        ax.set_title(f'{name} Dataset\nDistribution of Missing Rates', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    plot4_path = OUTPUT_DIR / 'missing_rate_distribution.png'
    fig.savefig(plot4_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {plot4_path}")
    plt.close()

def main():
    print("="*80)
    print(" AUGMENTATION ANALYSIS WITH CORRECTED JAW-SPECIFIC COUNTING")
    print("="*80)
    
    # Analyze both datasets
    df_aug = analyze_with_jaw_correction(AUGMENTED_CSV, "Augmented Dataset")
    df_rand = analyze_with_jaw_correction(RANDOM_CSV, "Random Augmented Dataset")
    
    # Compare
    comparison = compare_datasets(df_aug, "Augmented", df_rand, "Random")
    
    # Visualize
    print(f"\n{'='*80}")
    print(" Generating Visualizations...")
    print("="*80)
    visualize_all(df_aug, df_rand, comparison)
    
    # Save CSV reports
    csv_aug = OUTPUT_DIR / 'augmented_statistics.csv'
    csv_rand = OUTPUT_DIR / 'random_statistics.csv'
    csv_comp = OUTPUT_DIR / 'comparison.csv'
    
    df_aug.to_csv(csv_aug, index=False)
    df_rand.to_csv(csv_rand, index=False)
    comparison.to_csv(csv_comp, index=False)
    
    print(f"\n Saved CSV reports:")
    print(f"   {csv_aug}")
    print(f"   {csv_rand}")
    print(f"   {csv_comp}")
    
    # Final assessment
    print(f"\n{'='*80}")
    print(" FINAL ASSESSMENT: Did Augmentation Solve Imbalance?")
    print("="*80)
    
    for df, name in [(df_aug, "Augmented"), (df_rand, "Random")]:
        highly_imbalanced = len(df[(df['Missing_%'] < 20) | (df['Missing_%'] > 80)])
        balanced = len(df[(df['Missing_%'] >= 30) & (df['Missing_%'] <= 70)])
        
        print(f"\n {name} Dataset:")
        print(f"    Balanced teeth (30-70%): {balanced}/32 ({balanced/32*100:.1f}%)")
        print(f"     Highly imbalanced (<20% or >80%): {highly_imbalanced}/32 ({highly_imbalanced/32*100:.1f}%)")
        
        if balanced >= 24:  # 75% or more balanced
            print(f"   Verdict: GOOD - Augmentation effectively reduced imbalance!")
        elif balanced >= 16:  # 50% or more balanced
            print(f"   Verdict: MODERATE - Some improvement, but more work needed")
        else:
            print(f"   Verdict: POOR - Augmentation did not sufficiently address imbalance")
    
    print(f"\n{'='*80}")
    print(" Analysis complete! Check the output directory for detailed results.")
    print(f"   {OUTPUT_DIR}/")
    print("="*80)

if __name__ == '__main__':
    main()