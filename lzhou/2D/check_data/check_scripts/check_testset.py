import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

INPUT_CSV = "/home/user/lzhou/week10/label_flipped.csv"
RENDER_ROOT = "/home/user/lzhou/week15/render_output/test"
OUTPUT_DIR = "/home/user/lzhou/week16-32/output/check/test"

def normalize_key(value):
    if value is None or pd.isna(value):
        return None
    return str(value).strip().lower().replace("-", "_")

def normalize_png_stem_to_newid(stem):
    s = stem.replace('-', '_').strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'_rot\d+$', '', s, flags=re.IGNORECASE)
    if s.endswith('_top'):
        s = s[:-4]

    jaw_key = ''
    lower_s = s.lower()

    if 'upperjawscan' in lower_s:
        match = re.search(r'upperjawscan(\d*)', lower_s, flags=re.IGNORECASE)
        suffix = match.group(1) if match else ''
        jaw_key = f"upper{suffix}"
        s = re.sub(r'upperjawscan\d*', '', lower_s, flags=re.IGNORECASE)
    elif 'lowerjawscan' in lower_s:
        match = re.search(r'lowerjawscan(\d*)', lower_s, flags=re.IGNORECASE)
        suffix = match.group(1) if match else ''
        jaw_key = f"lower{suffix}"
        s = re.sub(r'lowerjawscan\d*', '', lower_s, flags=re.IGNORECASE)
    elif lower_s.endswith('_upper'):
        jaw_key = 'upper'
        s = s[:-6]
    elif lower_s.endswith('_lower'):
        jaw_key = 'lower'
        s = s[:-6]

    s = s.strip().replace(' ', '_').replace('-', '_')
    while '__' in s:
        s = s.replace('__', '_')
    s = s.strip('_')

    if jaw_key:
        new_id = f"{s}_{jaw_key}"
    else:
        new_id = s
    return new_id.lower()

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

def detect_jaw_from_string(value):
    if value is None or pd.isna(value):
        return None
    low = str(value).lower()
    if "upper" in low:
        return "upper"
    if "lower" in low:
        return "lower"
    return None

def parse_case_from_new_id(new_id):
    if new_id is None or pd.isna(new_id):
        return None, None
    s = str(new_id).strip()
    s_lower = s.lower()
    for tag in ("_upper", "_lower"):
        idx = s_lower.find(tag)
        if idx != -1:
            return s[:idx], tag[1:]
    return None, None

def derive_row_info(filename, new_id):
    jaw = None
    case_id = None
    stem = None

    if filename is not None and not pd.isna(filename):
        raw = str(filename).strip()
        if raw:
            raw_path = Path(raw.replace("\\", "/"))
            parts = raw_path.parts
            if parts and parts[0].lower() in ("upper", "lower"):
                jaw = parts[0].lower()
                if len(parts) > 1:
                    case_id = parts[1]
            stem = raw_path.stem if raw_path.stem else None
            if jaw is None:
                jaw = detect_jaw_from_string(raw)

    if case_id is None and new_id is not None and not pd.isna(new_id):
        case_from_new, jaw_from_new = parse_case_from_new_id(new_id)
        case_id = case_id or case_from_new
        jaw = jaw or jaw_from_new

    if stem is None and new_id is not None and not pd.isna(new_id):
        stem = str(new_id).strip()

    return case_id, jaw, stem

def ensure_top_suffix(stem):
    stem = stem.strip()
    if stem.lower().endswith("_top"):
        return stem
    return f"{stem}_top"

def build_dir_index(dir_path, cache):
    if dir_path in cache:
        return cache[dir_path]
    index = {}
    if dir_path.exists():
        for png_path in dir_path.glob("*.png"):
            index[normalize_key(png_path.stem)] = png_path
    cache[dir_path] = index
    return index

def build_png_index(render_root):
    index = {}
    render_root = Path(render_root)
    if not render_root.exists():
        print(f"Render root not found: {render_root}")
        return index

    for png_path in render_root.rglob("*.png"):
        new_id = normalize_png_stem_to_newid(png_path.stem)
        new_id_norm = normalize_key(new_id)
        if not new_id_norm:
            continue
        index.setdefault(new_id_norm, []).append(png_path)
    return index

def find_image_path(render_root, jaw, case_id, stem, cache):
    if not jaw or not stem:
        return None
    render_root = Path(render_root)
    stem_top = ensure_top_suffix(stem)
    candidates = []
    if case_id:
        candidates.append(render_root / jaw / case_id / f"{stem_top}.png")
    candidates.append(render_root / jaw / f"{stem_top}.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    stem_norm = normalize_key(stem_top)
    if case_id:
        case_dir = render_root / jaw / case_id
        index = build_dir_index(case_dir, cache)
        if stem_norm in index:
            return index[stem_norm]

    jaw_dir = render_root / jaw
    index = build_dir_index(jaw_dir, cache)
    if stem_norm in index:
        return index[stem_norm]

    return None

def collect_render_case_keys(render_root):
    case_keys = set()
    render_root = Path(render_root)
    if not render_root.exists():
        print(f"Render root not found: {render_root}")
        return case_keys

    for jaw in ("upper", "lower"):
        jaw_dir = render_root / jaw
        if not jaw_dir.exists():
            continue
        for entry in jaw_dir.iterdir():
            if entry.is_dir():
                case_id = entry.name.strip()
                case_id_norm = normalize_key(case_id)
                if case_id_norm:
                    case_keys.add(f"{case_id_norm}_{jaw}")
            elif entry.is_file() and entry.suffix.lower() == ".png":
                case_id, _ = parse_case_from_new_id(entry.stem)
                case_id_norm = normalize_key(case_id)
                if case_id_norm:
                    case_keys.add(f"{case_id_norm}_{jaw}")

    if not case_keys:
        for png_path in render_root.rglob("*.png"):
            parts = png_path.parts
            for i, part in enumerate(parts):
                part_lower = part.lower()
                if part_lower in ("upper", "lower"):
                    jaw = part_lower
                    case_id = None
                    if i + 2 < len(parts):
                        case_id = parts[i + 1]
                    if case_id is None or case_id.lower().endswith(".png"):
                        case_id, _ = parse_case_from_new_id(png_path.stem)
                    case_id_norm = normalize_key(case_id)
                    if case_id_norm:
                        case_keys.add(f"{case_id_norm}_{jaw}")
                    break
    return case_keys

def analyze_test_set_corrected():
    """
    CORRECTED VERSION: Analyzes tooth presence/absence with jaw-specific counting.
    
    Uses filename to separate upper and lower jaws, then:
    - Upper teeth (11-28) counted against upper jaw rows only
    - Lower teeth (31-48) counted against lower jaw rows only
    """
    input_csv = INPUT_CSV
    output_dir = OUTPUT_DIR
    render_root = RENDER_ROOT

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        raw_df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        return

    print("="*80)
    print(" TEST SET ANALYSIS - CORRECTED VERSION")
    print("="*80)

    png_index = build_png_index(render_root)
    render_case_keys = collect_render_case_keys(render_root)
    print(f"\n Render images (unique IDs): {len(png_index)}")
    if render_case_keys:
        print(f" Render cases (jaw-specific keys): {len(render_case_keys)}")
    
    df = raw_df.copy()
    case_ids = []
    jaw_types = []
    stems = []
    case_keys = []
    for _, row in df.iterrows():
        case_id, jaw, stem = derive_row_info(row.get('filename'), row.get('new_id'))
        case_ids.append(case_id)
        jaw_types.append(jaw)
        stems.append(stem)
        case_id_norm = normalize_key(case_id)
        if case_id_norm and jaw:
            case_keys.append(f"{case_id_norm}_{jaw}")
        else:
            case_keys.append(None)

    df['case_id'] = case_ids
    df['jaw_type'] = jaw_types
    df['image_stem'] = stems
    df['case_key'] = case_keys
    if 'new_id' in df.columns:
        df['new_id_norm'] = df['new_id'].apply(normalize_key)
    else:
        df['new_id_norm'] = None

    parsed_df = df[df['case_key'].notna()].copy()
    matched_mask = parsed_df['new_id_norm'].isin(png_index.keys())
    if render_case_keys:
        matched_mask = matched_mask | parsed_df['case_key'].isin(render_case_keys)
    matched_df = parsed_df[matched_mask].copy()

    dir_index_cache = {}
    image_paths = []
    for _, row in matched_df.iterrows():
        image_path = None
        new_id_norm = row.get('new_id_norm')
        if new_id_norm in png_index:
            image_path = png_index[new_id_norm][0]
        else:
            image_path = find_image_path(
                render_root,
                row['jaw_type'],
                row['case_id'],
                row['image_stem'],
                dir_index_cache
            )
        image_paths.append(str(image_path) if image_path else None)

    matched_df['image_path'] = image_paths
    found_df = matched_df[matched_df['image_path'].notna()].copy()
    missing_count = len(matched_df) - len(found_df)

    dedup_df = found_df.drop_duplicates(subset=['image_path']).copy()
    dropped_dup = len(found_df) - len(dedup_df)

    print(f"\n Cleaning Summary:")
    print(f"   CSV rows: {len(raw_df)}")
    print(f"   Rows with parsed case/jaw: {len(parsed_df)}")
    print(f"   Matched rows (case names): {len(matched_df)}")
    print(f"   Rows with images: {len(found_df)} (missing {missing_count})")
    print(f"   Deduped rows: {len(dedup_df)} (dropped {dropped_dup} duplicates)")

    cleaned_csv_path = os.path.join(output_dir, 'cleaned_testset.csv')
    dedup_df.to_csv(cleaned_csv_path, index=False)
    print(f"   Cleaned CSV saved to: {cleaned_csv_path}")

    if dropped_dup > 0:
        dup_samples = found_df[found_df.duplicated(subset=['image_path'], keep='first')]
        print(f"   Duplicate examples (first 5):")
        print(dup_samples[['filename', 'new_id', 'image_path']].head().to_string(index=False))

    # --- Separate by jaw type using cleaned rows ---
    upper_df = dedup_df[dedup_df['jaw_type'] == 'upper'].copy()
    lower_df = dedup_df[dedup_df['jaw_type'] == 'lower'].copy()
    
    upper_count = len(upper_df)
    lower_count = len(lower_df)
    total_count = len(dedup_df)
    
    if total_count == 0:
        print("\n No matched cases found after filtering. Check render_root and input_csv.")
        return

    print(f"\n Data Structure:")
    print(f"   Total rows: {total_count}")
    print(f"   Upper jaw rows: {upper_count}")
    print(f"   Lower jaw rows: {lower_count}")
    
    # --- Identify tooth columns ---
    tooth_columns = [col for col in dedup_df.columns if str(col).isdigit()]
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
            relevant_data = dedup_df[tooth_id]
            total_samples_for_tooth = total_count
        
        # 1 = missing, 0 = present
        absent_count = int(relevant_data.sum())
        present_count = total_samples_for_tooth - absent_count
        
        if total_samples_for_tooth > 0:
            present_pct = f"{present_count/total_samples_for_tooth*100:.1f}%"
            absent_pct = f"{absent_count/total_samples_for_tooth*100:.1f}%"
        else:
            present_pct = "0.0%"
            absent_pct = "0.0%"

        results.append({
            'ToothID': tooth_num,
            'Jaw': jaw_type,
            'Present_Count': present_count,
            'Absent_Count': absent_count,
            'Total_Samples': total_samples_for_tooth,
            'Present_%': present_pct,
            'Absent_%': absent_pct
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
    valid_total = results_df['Total_Samples'] > 0
    high_absence = results_df[valid_total & (results_df['Absent_Count']/results_df['Total_Samples'] > 0.2)]
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
        # Wrong way: use all rows for every tooth
        absent_count_wrong = int(dedup_df[tooth_id].sum())
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
    ax1.set_title(f'BEFORE Correction\n All {total_count} rows used for every tooth', 
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
    print(f"   {cleaned_csv_path}")
    print(f"   {csv_path}")
    print(f"   {stacked_plot_path}")
    print(f"   {rate_plot_path}")
    print(f"   {compare_plot_path}")
    print("="*80 + "\n")

if __name__ == '__main__':
    analyze_test_set_corrected()
