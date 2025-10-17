import json
import pandas as pd
import shutil
from pathlib import Path
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

# ============= CONFIGURATION =============
ORIGINAL_TRAIN_DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]
OUTPUT_DIR = Path("/home/user/lzhou/week10/output/augment_random")

FILL_BASE = True
BASE_COLOR = (0.85, 0.70, 0.70)
RANDOM_SEED = 42
COPIES_PER_SCAN = 2

# Tiered probabilities for tooth removal (updated: wisdom lowest, second molars medium, others high)
TEETH_REMOVAL_PROBS = {
    # Tier 1: Wisdom Teeth (Lowest Priority)
    18: 0.02, 28: 0.02, 38: 0.02, 48: 0.02,
    # Tier 2: Second Molars (Medium Priority)
    17: 0.12, 27: 0.12, 37: 0.12, 47: 0.12
}
# Tier 3: All other teeth get this high base probability
BASE_PROB = 0.30

# FDI Tooth Notation
UPPER_TEETH = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_TEETH = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
ALL_TEETH = sorted(UPPER_TEETH + LOWER_TEETH)
# =========================================


# =========================================
# HELPER FUNCTIONS
# =========================================
def load_all_train_samples():
    """Loads all available training samples from the specified paths."""
    samples = {'upper': [], 'lower': []}
    for data_path in ORIGINAL_TRAIN_DATA_PATHS:
        path = Path(data_path)
        if not path.exists():
            print(f"Warning: Training data path not found: {path}")
            continue
        jaw_type = "lower" if "lower" in str(path) else "upper"
        for case_dir in sorted(path.iterdir()):
            if case_dir.is_dir():
                obj, json_file = case_dir / f"{case_dir.name}_{jaw_type}.obj", case_dir / f"{case_dir.name}_{jaw_type}.json"
                if obj.exists() and json_file.exists():
                    samples[jaw_type].append({'case_id': case_dir.name, 'obj': str(obj), 'json': str(json_file), 'jaw': jaw_type})
    return samples

def load_tooth_labels_from_json(json_path):
    """Reads the JSON file and returns a dictionary mapping tooth FDI to its vertex indices."""
    with open(json_path, 'r') as f: data = json.load(f)
    tooth_to_vertices = {}
    for i, label in enumerate(data.get("labels", [])):
        if label != 0: tooth_to_vertices.setdefault(label, []).append(i + 1)
    return tooth_to_vertices

def create_updated_json_labels(original_json_path, output_json_path, removed_teeth_fdi):
    """Creates a new JSON file, setting the vertices of removed teeth to 0."""
    with open(original_json_path, 'r') as f: data = json.load(f)
    labels = data.get('labels', [])
    for i, label in enumerate(labels):
        if label in removed_teeth_fdi: labels[i] = 0
    with open(output_json_path, 'w') as f: json.dump({'labels': labels}, f)

def find_boundary_vertices(vertices_to_remove, faces):
    """Finds the vertex loop on the boundary of the area to be removed."""
    boundary_vertices = set()
    for face_line in faces:
        parts = face_line.strip().split()[1:]; face_verts = [int(p.split('/')[0]) for p in parts if '/' in p or p.isdigit()]
        if any(v in vertices_to_remove for v in face_verts) and any(v not in vertices_to_remove for v in face_verts):
            boundary_vertices.update(v for v in face_verts if v not in vertices_to_remove)
    return boundary_vertices

def remove_teeth_from_obj(input_obj, output_obj, teeth_to_remove, all_train_tooth_labels):
    """Removes specified teeth from an OBJ file and fills the resulting hole."""
    output_obj = Path(output_obj)
    if not teeth_to_remove: shutil.copyfile(input_obj, output_obj); return
    vertices_to_remove = {v for tooth in teeth_to_remove for v in all_train_tooth_labels.get(tooth, [])}
    if not vertices_to_remove: shutil.copyfile(input_obj, output_obj); return
    with open(input_obj, 'r') as f: lines = f.readlines()
    vertex_lines, face_lines, other_lines = [l for l in lines if l.startswith('v ')], [l for l in lines if l.startswith('f ')], [l for l in lines if not l.startswith(('v ', 'f '))]
    boundary_verts_old = find_boundary_vertices(vertices_to_remove, face_lines)
    with open(output_obj, 'w') as f:
        mtl_path = output_obj.with_suffix('.mtl'); f.write(f"mtllib {mtl_path.name}\n")
        with open(mtl_path, 'w') as mtl_f: mtl_f.write(f"newmtl Gingiva\nKd {BASE_COLOR[0]} {BASE_COLOR[1]} {BASE_COLOR[2]}\nKs 0.02 0.02 0.02\nNs 10.0\nillum 2\n")
        for line in other_lines: f.write(line)
        vertex_map, new_idx, all_coords = {}, 1, {}
        for old_idx, line in enumerate(vertex_lines, 1):
            all_coords[old_idx] = [float(c) for c in line.strip().split()[1:]]
            if old_idx not in vertices_to_remove: f.write(line); vertex_map[old_idx] = new_idx; new_idx += 1
        centroid_idx = None
        if boundary_verts_old:
            centroid = np.mean([all_coords.get(v, [0,0,0]) for v in boundary_verts_old], axis=0)
            f.write(f"v {centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f}\n"); centroid_idx = new_idx
        for line in face_lines:
            parts = line.strip().split(); face_verts_old = [int(p.split('/')[0]) for p in parts[1:]]
            if not any(v in vertices_to_remove for v in face_verts_old):
                new_face = 'f ' + ' '.join(p.replace(str(v_old), str(vertex_map[v_old]), 1) for p, v_old in zip(parts[1:], face_verts_old))
                f.write(new_face + '\n')
        if centroid_idx:
            boundary_new_indices = sorted([vertex_map[i] for i in boundary_verts_old if i in vertex_map])
            if boundary_new_indices:
                f.write("usemtl Gingiva\n")
                for i in range(len(boundary_new_indices)): v1, v2 = boundary_new_indices[i], boundary_new_indices[(i + 1) % len(boundary_new_indices)]; f.write(f"f {v1} {v2} {centroid_idx}\n")

# =========================================
# MAIN EXECUTION
# =========================================
def main():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[1/3] Loading original training samples..."); train_samples = load_all_train_samples()
    print(f"  Found {len(train_samples['upper'])} upper and {len(train_samples['lower'])} lower samples.")
    
    csv_label_rows = []
    
    all_original_scans = train_samples['upper'] + train_samples['lower']
    total_tasks = len(all_original_scans) * COPIES_PER_SCAN
    
    print(f"[2/3] Starting random augmentation for {total_tasks} new scans...")
    with tqdm(total=total_tasks, desc="Augmenting") as pbar:
        for train_sample in all_original_scans:
            jaw_type = train_sample['jaw']
            original_case_id, train_obj_path, train_json_path = train_sample['case_id'], train_sample['obj'], train_sample['json']
            
            all_train_tooth_labels = load_tooth_labels_from_json(train_json_path)
            teeth_present_in_train = set(all_train_tooth_labels.keys())

            for copy_num in range(1, COPIES_PER_SCAN + 1):
                pbar.set_description(f"Processing {original_case_id}")
                
                # Determine which teeth are candidates for removal based on jaw type
                candidate_teeth = UPPER_TEETH if jaw_type == 'upper' else LOWER_TEETH
                
                # Filter candidates to only those present in the current scan
                possible_to_remove = [tooth for tooth in candidate_teeth if tooth in teeth_present_in_train]
                if not possible_to_remove:
                    tqdm.write(f"  > Skipping {original_case_id} Copy {copy_num}: No teeth present to remove.")
                    pbar.update(1)
                    continue
                
                # Create weights corresponding to the available teeth
                weights = [TEETH_REMOVAL_PROBS.get(tooth, BASE_PROB) for tooth in possible_to_remove]
                
                # Decide how many teeth to remove for this copy
                num_to_remove = random.randint(2, min(5, len(possible_to_remove)))
                
                # Select the teeth to remove using weighted random sampling
                selected_for_removal = set(random.choices(possible_to_remove, weights=weights, k=num_to_remove))
                
                teeth_present_after = teeth_present_in_train - selected_for_removal
                
                output_subdir = OUTPUT_DIR / jaw_type / original_case_id
                output_subdir.mkdir(parents=True, exist_ok=True)
                new_filename_base = f"{original_case_id}_{jaw_type}_randomcopy_{copy_num:02d}"
                output_obj, output_json = output_subdir / f"{new_filename_base}.obj", output_subdir / f"{new_filename_base}.json"
                
                remove_teeth_from_obj(train_obj_path, output_obj, selected_for_removal, all_train_tooth_labels)
                create_updated_json_labels(train_json_path, output_json, selected_for_removal)
                
                # Create the final, correct label dictionary for the CSV
                final_flipped_label_dict = {}
                for tooth in ALL_TEETH:
                    is_in_correct_jaw = (jaw_type == 'upper' and tooth in UPPER_TEETH) or \
                                        (jaw_type == 'lower' and tooth in LOWER_TEETH)
                    if is_in_correct_jaw:
                        final_flipped_label_dict[str(tooth)] = 1 if tooth not in teeth_present_after else 0
                    else:
                        final_flipped_label_dict[str(tooth)] = 1
                
                csv_label_rows.append({'filename': str(output_obj.relative_to(OUTPUT_DIR)),'new_id': new_filename_base,'Date of labeling': datetime.now().strftime('%Y-%m-%d'),'filetype': 'obj',**final_flipped_label_dict})
                pbar.update(1)

    print("\n[3/3] All augmentation tasks complete. Saving master labels CSV...")
    if csv_label_rows:
        df_labels = pd.DataFrame(csv_label_rows)
        # Ensure correct column order, converting tooth numbers to strings for lookup
        cols = ['filename', 'new_id', 'Date of labeling', 'filetype'] + [str(t) for t in ALL_TEETH]
        df_labels = df_labels[cols]
        output_csv_path = OUTPUT_DIR / "train_labels_random.csv"
        df_labels.to_csv(output_csv_path, index=False)
        print(f"✓ Successfully saved master label file with {len(df_labels)} entries to {output_csv_path}")

    print("\n✓ Random augmentation process complete!")

if __name__ == "__main__":
    main()
