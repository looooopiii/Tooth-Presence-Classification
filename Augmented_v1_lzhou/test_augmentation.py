import json
import pandas as pd
import shutil
from pathlib import Path
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

# ============= CONFIGURATION =============
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"
TRAIN_DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]
OUTPUT_DIR = Path("/home/user/lzhou/week10/output/augment_test")

FILL_BASE = True
BASE_COLOR = (0.85, 0.70, 0.70)
RANDOM_SEED = 43
NUM_COPIES = 5
FLIPPED_LABELS_IN_CSV = True

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
    for data_path in TRAIN_DATA_PATHS:
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

def _order_loop(adj, start):
    """
    Given an undirected adjacency mapping {v: set(neighbors)} for a single loop,
    return an ordered list of vertices following the boundary.
    """
    loop = [start]
    prev = None
    cur = start
    while True:
        nbrs = adj[cur] - ({prev} if prev is not None else set())
        if not nbrs:
            break
        nxt = next(iter(nbrs))
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > 100000:  # safety
            break
    return loop

def extract_boundary_loops(vertices_to_remove, face_lines):
    """
    Build boundary *loops* (each loop is an ordered list of kept-vertex indices)
    around regions that will be removed. We collect edges from faces that mix
    removed and kept vertices, then split them into connected components and
    order each component along the loop.
    """
    # Parse faces into lists of vertex indices (v/vt/vn tolerant)
    faces = []
    for fl in face_lines:
        parts = fl.strip().split()[1:]
        if not parts:
            continue
        vids = []
        for p in parts:
            # token like "12/34/56" -> take vertex id "12"
            base = p.split('/')[0]
            if base.lstrip('-').isdigit():
                vids.append(int(base))
        if len(vids) >= 2:
            faces.append(vids)

    kept_kept_edges = []  # boundary candidate edges (both endpoints kept) from mixed faces
    for vids in faces:
        has_removed = any(v in vertices_to_remove for v in vids)
        has_kept = any(v not in vertices_to_remove for v in vids)
        if not (has_removed and has_kept):
            continue
        n = len(vids)
        for i in range(n):
            a, b = vids[i], vids[(i + 1) % n]
            # An edge is on the boundary if both endpoints are kept,
            # but the face itself touches removed vertices.
            if (a not in vertices_to_remove) and (b not in vertices_to_remove):
                kept_kept_edges.append((a, b))

    # Build undirected adjacency of boundary graph
    adj = {}
    for a, b in kept_kept_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # Extract connected components
    visited = set()
    loops = []
    for v in list(adj.keys()):
        if v in visited:
            continue
        # BFS/DFS to get component
        stack = [v]
        comp = set()
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            comp.add(x)
            stack.extend(list(adj.get(x, [])))
        # Order this component as a loop by walking degree-2 chain
        # pick a start (any)
        start = next(iter(comp))
        sub_adj = {k: (adj.get(k, set()) & comp) for k in comp}
        ordered = _order_loop(sub_adj, start)
        if len(ordered) >= 3:
            loops.append(ordered)
    return loops

def remove_teeth_from_obj(input_obj, output_obj, teeth_to_remove, all_train_tooth_labels):
    """Removes specified teeth from an OBJ file and fills the resulting hole."""
    output_obj = Path(output_obj)
    if not teeth_to_remove: shutil.copyfile(input_obj, output_obj); return
    vertices_to_remove = {v for tooth in teeth_to_remove for v in all_train_tooth_labels.get(tooth, [])}
    if not vertices_to_remove: shutil.copyfile(input_obj, output_obj); return
    with open(input_obj, 'r') as f: lines = f.readlines()
    vertex_lines, face_lines, other_lines = [l for l in lines if l.startswith('v ')], [l for l in lines if l.startswith('f ')], [l for l in lines if not l.startswith(('v ', 'f '))]
    boundary_loops_old = extract_boundary_loops(vertices_to_remove, face_lines)
    with open(output_obj, 'w') as f:
        mtl_path = output_obj.with_suffix('.mtl'); f.write(f"mtllib {mtl_path.name}\n")
        with open(mtl_path, 'w') as mtl_f: mtl_f.write(f"newmtl Gingiva\nKd {BASE_COLOR[0]} {BASE_COLOR[1]} {BASE_COLOR[2]}\nKs 0.02 0.02 0.02\nNs 10.0\nillum 2\n")
        for line in other_lines: f.write(line)
        vertex_map, new_idx, all_coords = {}, 1, {}
        for old_idx, line in enumerate(vertex_lines, 1):
            all_coords[old_idx] = [float(c) for c in line.strip().split()[1:]]
            if old_idx not in vertices_to_remove: f.write(line); vertex_map[old_idx] = new_idx; new_idx += 1
        # Add centroids for each boundary loop (computed in old vertex index space).
        # We'll store (centroid_idx_new, ordered_new_indices) for later triangulation.
        loop_infos = []
        for loop in boundary_loops_old:
            # robustly guard against degenerate loops
            coords = [all_coords[v] for v in loop if v in all_coords]
            if len(coords) < 3:
                continue
            centroid = np.mean(coords, axis=0)
            f.write(f"v {centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f}\n")
            centroid_idx_new = new_idx
            new_idx += 1
            # map old loop vertex indices to new ones (skip those removed)
            ordered_new = []
            for v_old in loop:
                if v_old in vertices_to_remove:
                    continue
                if v_old in vertex_map:
                    ordered_new.append(vertex_map[v_old])
            if len(ordered_new) >= 3:
                loop_infos.append((centroid_idx_new, ordered_new))
        for line in face_lines:
            parts = line.strip().split()
            face_verts_old = [int(p.split('/')[0]) for p in parts[1:]]
            if not any(v in vertices_to_remove for v in face_verts_old):
                new_face = 'f ' + ' '.join(p.replace(str(v_old), str(vertex_map[v_old]), 1) for p, v_old in zip(parts[1:], face_verts_old))
                f.write(new_face + '\n')
        # Triangulate each loop as a fan to its own centroid.
        if loop_infos:
            f.write("usemtl Gingiva\n")
            for centroid_idx_new, ordered_new in loop_infos:
                # create triangles (v_i, v_{i+1}, centroid)
                for i in range(len(ordered_new)):
                    v1 = ordered_new[i]
                    v2 = ordered_new[(i + 1) % len(ordered_new)]
                    f.write(f"f {v1} {v2} {centroid_idx_new}\n")

# =========================================
# MAIN EXECUTION
# =========================================
def main():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    if not Path(TEST_LABELS_CSV).exists(): print(f"[ERROR] CSV file not found: {TEST_LABELS_CSV}"); return

    print("[1/3] Loading all training samples..."); train_samples = load_all_train_samples()
    print(f"  Found {len(train_samples['upper'])} upper and {len(train_samples['lower'])} lower samples.")
    
    print(f"[2/3] Processing test patterns..."); df_test = pd.read_csv(TEST_LABELS_CSV)
    
    csv_label_rows = []
    
    total_tasks = len(df_test) * NUM_COPIES
    with tqdm(total=total_tasks, desc="Augmenting Data") as pbar:
        for i, test_row in df_test.iterrows():
            jaw_type = 'lower' if 'lower' in test_row['filename'].lower() else 'upper'
            
            if not train_samples[jaw_type]:
                tqdm.write(f"Warning: No training samples for jaw '{jaw_type}'. Skipping {NUM_COPIES} tasks for pattern {i+1}.")
                pbar.update(NUM_COPIES)
                continue
                
            value_for_missing = 1 if FLIPPED_LABELS_IN_CSV else 0
            target_missing_teeth = {int(col) for col in df_test.columns if col.isdigit() and str(col) in test_row and test_row[str(col)] == value_for_missing}
            
            for copy_num in range(1, NUM_COPIES + 1):
                train_sample = random.choice(train_samples[jaw_type])
                original_case_id, train_obj_path, train_json_path = train_sample['case_id'], train_sample['obj'], train_sample['json']
                all_train_tooth_labels = load_tooth_labels_from_json(train_json_path)
                teeth_present_in_train = set(all_train_tooth_labels.keys())
                teeth_to_remove = target_missing_teeth.intersection(teeth_present_in_train)
                teeth_present_after = teeth_present_in_train - teeth_to_remove
                
                output_subdir = OUTPUT_DIR / jaw_type / original_case_id
                output_subdir.mkdir(parents=True, exist_ok=True)
                new_filename_base = f"{original_case_id}_{jaw_type}_pattern_{i:04d}_copy_{copy_num:02d}"
                output_obj, output_json = output_subdir / f"{new_filename_base}.obj", output_subdir / f"{new_filename_base}.json"
                
                remove_teeth_from_obj(train_obj_path, output_obj, teeth_to_remove, all_train_tooth_labels)
                create_updated_json_labels(train_json_path, output_json, teeth_to_remove)
                
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

    print("\n[3/3] All tasks complete. Saving master labels CSV...")
    if csv_label_rows:
        df_labels = pd.DataFrame(csv_label_rows)
        cols = ['filename', 'new_id', 'Date of labeling', 'filetype'] + [str(t) for t in ALL_TEETH]
        df_labels = df_labels[cols]
        output_csv_path = OUTPUT_DIR / "train_labels_augmented.csv"
        df_labels.to_csv(output_csv_path, index=False)
        print(f"✓ Successfully saved master label file with {len(df_labels)} entries to {output_csv_path}")

    print("\n✓ Augmentation process complete!")

if __name__ == "__main__":
    main()
