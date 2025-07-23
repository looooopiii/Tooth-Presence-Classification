"""
Tooth Missing/Present Labeling Script
Processes JSON segmentation files to create binary labels for tooth presence/absence
"""

import json
import csv
from pathlib import Path
import pandas as pd
from collections import Counter

# === CONFIG ===
BASE_DIR = Path("/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split")
IMAGE_DIR = Path("/home/user/tbrighton/blender_outputs/multi_views")
OUTPUT_CSV = Path("/home/user/tbrighton/blender-scripts/tooth_labels.csv")

# Standard tooth numbering (FDI notation)
UPPER_TEETH = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_TEETH = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
#ALL_TEETH = UPPER_TEETH + LOWER_TEETH

def load_json_labels(json_file):
    """Load and parse JSON segmentation file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f" Error loading {json_file}: {e}")
        return None

def analyze_present_teeth(labels, instances):
    """
    Analyze which teeth are present based on labels and instances
    Returns set of tooth numbers that are present
    """
    # Get unique tooth labels (excluding 0 which is gingiva)
    unique_labels = set(labels)
    unique_labels.discard(0)  # Remove gingiva label
    
    # Count instances for each tooth to ensure they have sufficient vertices
    label_counts = Counter(labels)
    
    # Filter out teeth with very few vertices (likely noise)
    MIN_VERTICES = 50  # Minimum vertices to consider a tooth present
    present_teeth = {label for label in unique_labels if label_counts[label] >= MIN_VERTICES}
    
    return present_teeth

def create_tooth_presence_vector(present_teeth, jaw_type):
    """
    Create binary vector for tooth presence
    Returns dictionary with tooth_number: 1/0 (present/missing)
    """
    if jaw_type == "upper":
        reference_teeth = UPPER_TEETH
    elif jaw_type == "lower":
        reference_teeth = LOWER_TEETH
    else:
        raise ValueError(f"Unknown jaw type: {jaw_type}")
    
    tooth_presence = {}
    for tooth_num in reference_teeth:
        tooth_presence[tooth_num] = 1 if tooth_num in present_teeth else 0
    
    return tooth_presence

def process_patient_data():
    """Process all patients and create labeling dataset"""
    results = []
    errors = []
    
    processed_count = 0
    missing_json = 0
    missing_image = 0
    
    for jaw in ['upper', 'lower']:
        jaw_folder = BASE_DIR / jaw
        if not jaw_folder.exists():
            print(f"⚠️ Jaw folder not found: {jaw_folder}")
            continue
            
        print(f"\n Processing {jaw} jaw...")
        
        for patient_folder in jaw_folder.iterdir():
            if not patient_folder.is_dir():
                continue
                
            patient_id = patient_folder.name
            
            # Check if rendered image exists
            image_file = IMAGE_DIR / f"{patient_id}_{jaw}_top.png"
            if not image_file.exists():
                missing_image += 1
                continue
            
            # Look for JSON file
            json_file = patient_folder / f"{patient_id}_{jaw}.json"
            if not json_file.exists():
                missing_json += 1
                print(f" JSON not found: {json_file}")
                continue
            
            # Load JSON data
            json_data = load_json_labels(json_file)
            if json_data is None:
                errors.append(f"Failed to load JSON: {json_file}")
                continue
            
            # Validate JSON structure
            if not all(key in json_data for key in ['labels', 'instances', 'jaw']):
                errors.append(f"Invalid JSON structure: {json_file}")
                continue
            
            # Analyze present teeth
            present_teeth = analyze_present_teeth(json_data['labels'], json_data['instances'])
            
            # Create tooth presence vector
            try:
                tooth_presence = create_tooth_presence_vector(present_teeth, jaw)
            except ValueError as e:
                errors.append(f"Error processing {patient_id}_{jaw}: {e}")
                continue
            
            # Create result record
            result = {
                'patient_id': patient_id,
                'jaw': jaw,
                'image_path': str(image_file),
                'total_teeth_present': sum(tooth_presence.values()),
                'total_teeth_missing': len(tooth_presence) - sum(tooth_presence.values()),
            }
            
            # Add individual tooth presence
            for tooth_num, present in tooth_presence.items():
                result[f'tooth_{tooth_num}'] = present
            
            # Add summary info
            result['present_teeth_list'] = sorted(list(present_teeth))
            result['vertex_count'] = len(json_data['labels'])
            
            results.append(result)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f" Processed {processed_count} patients...")
    
    print(f"\nProcessing Summary:")
    print(f"   • Successfully processed: {processed_count}")
    print(f"   • Missing JSON files: {missing_json}")
    print(f"   • Missing image files: {missing_image}")
    print(f"   • Errors encountered: {len(errors)}")
    
    return results, errors

def save_results(results, errors):
    """Save results to CSV and error log"""
    if not results:
        print(" No results to save!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    base_cols = ['patient_id', 'jaw', 'image_path', 'total_teeth_present', 'total_teeth_missing', 'vertex_count']
    tooth_cols = [col for col in df.columns if col.startswith('tooth_')]
    other_cols = [col for col in df.columns if col not in base_cols + tooth_cols]
    
    column_order = base_cols + sorted(tooth_cols) + other_cols
    df = df[column_order]
    
    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f" Saved labels to: {OUTPUT_CSV}")
    
    # Print dataset statistics
    print(f"\n Dataset Statistics:")
    print(f"   • Total samples: {len(df)}")
    print(f"   • Upper jaw samples: {len(df[df['jaw'] == 'upper'])}")
    print(f"   • Lower jaw samples: {len(df[df['jaw'] == 'lower'])}")
    
    # Tooth presence statistics
    tooth_cols = [col for col in df.columns if col.startswith('tooth_')]
    for col in sorted(tooth_cols)[:5]:  # Show first 5 teeth
        present_count = df[col].sum()
        missing_count = len(df) - present_count
        print(f"   • {col}: {present_count} present, {missing_count} missing")
    
    # Overall missing tooth statistics
    print(f"   • Avg teeth present per jaw: {df['total_teeth_present'].mean():.1f}")
    print(f"   • Avg teeth missing per jaw: {df['total_teeth_missing'].mean():.1f}")
    
    # Save errors if any
    if errors:
        error_file = OUTPUT_CSV.parent / "labeling_errors.txt"
        with open(error_file, 'w') as f:
            f.write("\n".join(errors))
        print(f" Errors logged to: {error_file}")

def analyze_tooth_distribution(csv_file):
    """Analyze and visualize tooth presence distribution"""
    df = pd.read_csv(csv_file)
    
    print(f"\n🔍 Tooth Distribution Analysis:")
    tooth_cols = [col for col in df.columns if col.startswith('tooth_')]
    
    # Calculate presence percentage for each tooth
    tooth_stats = {}
    for col in tooth_cols:
        tooth_num = int(col.split('_')[1])
        present_count = df[col].sum()
        total_count = len(df)
        percentage = (present_count / total_count) * 100
        tooth_stats[tooth_num] = {
            'present': present_count,
            'missing': total_count - present_count,
            'percentage': percentage
        }
    
    # Sort by tooth number
    for tooth_num in sorted(tooth_stats.keys()):
        stats = tooth_stats[tooth_num]
        print(f"   Tooth {tooth_num}: {stats['present']:3d} present ({stats['percentage']:5.1f}%), {stats['missing']:3d} missing")
    
    # Find most commonly missing teeth
    missing_counts = [(tooth_num, stats['missing']) for tooth_num, stats in tooth_stats.items()]
    missing_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n Most Commonly Missing Teeth:")
    for tooth_num, missing_count in missing_counts[:10]:
        percentage = (missing_count / len(df)) * 100
        print(f" Tooth {tooth_num}: {missing_count} missing ({percentage:.1f}%)")

def main():
    print("Starting tooth labeling process...")
    print(f" Dataset directory: {BASE_DIR}")
    print(f"  Images directory: {IMAGE_DIR}")
    print(f" Output CSV: {OUTPUT_CSV}")
    
    # Process all patient data
    results, errors = process_patient_data()
    
    # Save results
    save_results(results, errors)
    
    # Analyze distribution
    if OUTPUT_CSV.exists():
        analyze_tooth_distribution(OUTPUT_CSV)
    
    print(f"\n Labeling complete! Check {OUTPUT_CSV} for results.")

if __name__ == "__main__":
    main()