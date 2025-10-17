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
# === Augmentation policy suggested by prof/supervisor ===
# Choose how many teeth to remove and which ones with probabilities (lower prob for wisdom teeth),
# aiming for a more balanced per-tooth missing/present distribution.
AUGMENT_MODE = "balanced"  # {"balanced", "match_test"}
REMOVE_K_RANGE = (2, 5)     # randomly choose K in this range
# weights per tooth id; defaults to 1.0 if not listed. Wisdom teeth get lower weights.
REMOVAL_WEIGHTS = {
    # Upper jaw
    18: 0.2, 17: 0.6, 16: 0.8, 15: 1.0, 14: 1.0, 13: 1.2, 12: 1.2, 11: 1.2,
    21: 1.2, 22: 1.2, 23: 1.2, 24: 1.0, 25: 1.0, 26: 0.8, 27: 0.6, 28: 0.2,
    # Lower jaw
    48: 0.2, 47: 0.6, 46: 0.8, 45: 1.0, 44: 1.0, 43: 1.2, 42: 1.2, 41: 1.2,
    31: 1.2, 32: 1.2, 33: 1.2, 34: 1.0, 35: 1.0, 36: 0.8, 37: 0.6, 38: 0.2,
}
# Limit fraction of wisdom teeth in a removal set (encourages non‑wisdom deletions)
MAX_WISDOM_IN_SET = 1       # at most this many wisdom teeth per augmented sample
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


# --- Balanced sampling utility ---
def _sample_teeth_balanced(candidates, k):
    """Sample k unique teeth from candidates with weights in REMOVAL_WEIGHTS.
    Enforce at most MAX_WISDOM_IN_SET wisdom teeth if possible.
    """
    if not candidates:
        return set()
    k = max(0, min(k, len(candidates)))
    # Build weight vector
    weights = np.array([REMOVAL_WEIGHTS.get(t, 1.0) for t in candidates], dtype=np.float64)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    probs = weights / weights.sum()
    # initial sample
    chosen = list(np.random.choice(candidates, size=k, replace=False, p=probs))
    # enforce wisdom constraint
    wisdom_set = {18, 28, 38, 48}
    if MAX_WISDOM_IN_SET is not None and MAX_WISDOM_IN_SET >= 0:
        wisdom_in_chosen = [t for t in chosen if t in wisdom_set]
        if len(wisdom_in_chosen) > MAX_WISDOM_IN_SET:
            # replace extras with non-wisdom teeth if available
            keep = set(chosen) - set(wisdom_in_chosen[MAX_WISDOM_IN_SET:])
            pool = [t for t in candidates if (t not in keep) and (t not in wisdom_set)]
            need = k - len(keep)
            if need > 0 and len(pool) >= need:
                add = list(np.random.choice(pool, size=need, replace=False))
                chosen = list(keep) + add
    return set(chosen)

# --- Robust boundary detection and ordering (edge kept/removed counts) ---
def _parse_face_verts(face_line: str):
    """Parse an OBJ face line 'f v[/vt[/vn]] ...' -> list of vertex ids (int)."""
    parts = face_line.strip().split()
    vids = []
    for tok in parts[1:]:
        base = tok.split('/')[0]
        if base.lstrip('-').isdigit():
            vids.append(int(base))
    return vids

def _build_boundary_components(vertices_to_remove, face_lines):
    """
    Build boundary components using edge kept/removed adjacency counting.
    Returns: list of dicts [{'ordered': [v1,...], 'closed': True/False}, ...]
    where 'ordered' is a simple walk sequence (ok for chains or cycles).
    """
    # 1) Parse faces + classify kept/removed
    faces = []
    face_types = []  # 'kept' | 'removed'
    for fl in face_lines:
        vids = _parse_face_verts(fl)
        if len(vids) < 2:
            continue
        faces.append(vids)
        is_removed = any(v in vertices_to_remove for v in vids)
        face_types.append('removed' if is_removed else 'kept')

    # 2) Accumulate for each undirected edge the number of kept/removed adjacencies
    from collections import defaultdict, deque
    edge_cnt = defaultdict(lambda: {'kept': 0, 'removed': 0})
    for vids, ftype in zip(faces, face_types):
        n = len(vids)
        for i in range(n):
            a, b = vids[i], vids[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            edge_cnt[key][ftype] += 1

    # 3) Select boundary edges: both endpoints kept, and edge touches kept & removed
    boundary_edges = []
    for (u, v), cnt in edge_cnt.items():
        if cnt['kept'] > 0 and cnt['removed'] > 0:
            if (u not in vertices_to_remove) and (v not in vertices_to_remove):
                boundary_edges.append((u, v))

    # 4) Build adjacency and split into components
    adj = {}
    for u, v in boundary_edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    components = []
    visited = set()
    for start in list(adj.keys()):
        if start in visited:
            continue
        # BFS to get connected vertices
        dq = deque([start])
        comp = set()
        while dq:
            x = dq.popleft()
            if x in visited:
                continue
            visited.add(x)
            comp.add(x)
            dq.extend(list(adj.get(x, [])))

        # degrees within comp
        deg = {k: len(adj.get(k, set()) & comp) for k in comp}
        closed = all(d == 2 for d in deg.values()) and len(comp) >= 3

        # produce a simple walk order (works for chains/cycles)
        if closed:
            # walk cycle
            cur = next(iter(comp))
            prev = None
            ordered = [cur]
            while True:
                nbrs = (adj.get(cur, set()) & comp) - ({prev} if prev else set())
                if not nbrs:
                    break
                nxt = next(iter(nbrs))
                if nxt == ordered[0]:
                    break
                ordered.append(nxt)
                prev, cur = cur, nxt
                if len(ordered) > 200000:
                    break
        else:
            # open chain: start from any endpoint (deg==1) if exists
            endpoints = [k for k, d in deg.items() if d == 1]
            cur = endpoints[0] if endpoints else next(iter(comp))
            prev = None
            ordered = [cur]
            while True:
                nbrs = (adj.get(cur, set()) & comp) - ({prev} if prev else set())
                if not nbrs:
                    break
                nxt = next(iter(nbrs))
                ordered.append(nxt)
                prev, cur = cur, nxt
                if len(ordered) > 200000:
                    break

        if len(ordered) >= 2:
            components.append({'ordered': ordered, 'closed': closed})

    return components

def _order_by_local_plane(old_ids, coord_lut):
    """
    Project boundary vertices to a best-fit plane through centroid and order by angle.
    Works for both closed cycles and open chains; for chains this produces a 'near cycle'
    that is robust for fan triangulation.
    """
    pts = [coord_lut[i] for i in old_ids if i in coord_lut]
    if len(pts) == 0:
        return []
    arr = np.asarray(pts, dtype=np.float64)
    c = arr.mean(axis=0)

    # PCA to get local plane
    X = arr - c
    cov = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(cov)
    # normal is eigenvector with smallest eigenvalue
    n_idx = int(np.argmin(eigvals))
    # pick two largest as in-plane axes
    in_idx = [i for i in [0,1,2] if i != n_idx]
    u, v = eigvecs[:, in_idx[1]], eigvecs[:, in_idx[0]]  # ensure 2 vectors

    # 2D coords and angles
    du = X @ u
    dv = X @ v
    ang = np.arctan2(dv, du)
    order = np.argsort(ang)
    ordered_old_ids = [old_ids[i] for i in order]
    # remove accidental duplicates while preserving order
    seen = set()
    unique_ids = []
    for oid in ordered_old_ids:
        if oid not in seen:
            unique_ids.append(oid)
            seen.add(oid)
    return unique_ids

def remove_teeth_from_obj(input_obj, output_obj, teeth_to_remove, all_train_tooth_labels):
    """Removes specified teeth from an OBJ file and fills the resulting hole."""
    output_obj = Path(output_obj)
    if not teeth_to_remove:
        shutil.copyfile(input_obj, output_obj)
        return
    vertices_to_remove = {v for tooth in teeth_to_remove for v in all_train_tooth_labels.get(tooth, [])}
    if not vertices_to_remove:
        shutil.copyfile(input_obj, output_obj)
        return
    with open(input_obj, 'r') as f:
        lines = f.readlines()

    # Scan original lines to capture mtllib and dominant material used by kept faces.
    mtllib_name = None
    current_mtl = None
    kept_mtl_counts = {}
    for ln in lines:
        if ln.startswith('mtllib '):
            # take the first mtllib reference
            if mtllib_name is None:
                mtllib_name = ln.split(None, 1)[1].strip()
        elif ln.startswith('usemtl '):
            parts = ln.split()
            if len(parts) >= 2:
                current_mtl = parts[1].strip()
        elif ln.startswith('f '):
            parts = ln.strip().split()
            face_verts_old_scan = [int(p.split('/')[0]) for p in parts[1:]]
            # we haven't built vertices_to_remove yet; defer this count after it's computed

    vertex_lines = [l for l in lines if l.startswith('v ')]
    face_lines = [l for l in lines if l.startswith('f ')]
    other_lines = [l for l in lines if not l.startswith(('v ', 'f '))]

    # Now that vertices_to_remove is known, recompute dominant material based on kept faces
    current_mtl = None
    kept_mtl_counts = {}
    for ln in lines:
        if ln.startswith('usemtl '):
            parts = ln.split()
            if len(parts) >= 2:
                current_mtl = parts[1].strip()
        elif ln.startswith('f '):
            parts = ln.strip().split()
            face_verts_old_scan = [int(p.split('/')[0]) for p in parts[1:]]
            if not any(v in vertices_to_remove for v in face_verts_old_scan):
                if current_mtl:
                    kept_mtl_counts[current_mtl] = kept_mtl_counts.get(current_mtl, 0) + 1
    fill_material = None
    if kept_mtl_counts:
        fill_material = max(kept_mtl_counts, key=kept_mtl_counts.get)

    boundary_components = _build_boundary_components(vertices_to_remove, face_lines)
    with open(output_obj, 'w') as f:
        # Preserve original mtllib if present; also copy the .mtl next to the new obj
        if mtllib_name:
            # write a single mtllib line
            f.write(f"mtllib {mtllib_name}\n")
            # try copying original mtl file into the output directory if it exists
            src_mtl = Path(input_obj).parent / mtllib_name
            dst_mtl = output_obj.parent / Path(mtllib_name).name
            try:
                if src_mtl.exists() and (not dst_mtl.exists()):
                    shutil.copyfile(src_mtl, dst_mtl)
            except Exception as _e:
                pass
        # write other non-geometry lines except any existing mtllib duplicates
        for line in other_lines:
            if line.startswith('mtllib '):
                continue
            f.write(line)
        vertex_map, new_idx, all_coords = {}, 1, {}
        for old_idx, line in enumerate(vertex_lines, 1):
            all_coords[old_idx] = [float(c) for c in line.strip().split()[1:]]
            if old_idx not in vertices_to_remove:
                f.write(line)
                vertex_map[old_idx] = new_idx
                new_idx += 1
        # Add centroids for each boundary component (open or closed chain/cycle), robust ordering.
        loop_infos = []
        for comp in boundary_components:
            loop_old = comp['ordered']
            if len(loop_old) < 2:
                continue
            # order by local plane angle to form a robust polygon sequence
            loop_old_ordered = _order_by_local_plane(loop_old, all_coords)
            if len(loop_old_ordered) < 2:
                continue
            coords = [all_coords[v] for v in loop_old_ordered if v in all_coords]
            centroid = np.mean(coords, axis=0)
            f.write(f"v {centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f}\n")
            centroid_idx_new = new_idx
            new_idx += 1

            ordered_new = []
            for v_old in loop_old_ordered:
                if v_old in vertices_to_remove:
                    continue
                if v_old in vertex_map:
                    ordered_new.append(vertex_map[v_old])
            if len(ordered_new) >= 2:
                loop_infos.append((centroid_idx_new, ordered_new))
        for line in face_lines:
            parts = line.strip().split()
            face_verts_old = [int(p.split('/')[0]) for p in parts[1:]]
            if not any(v in vertices_to_remove for v in face_verts_old):
                new_face = 'f ' + ' '.join(p.replace(str(v_old), str(vertex_map[v_old]), 1) for p, v_old in zip(parts[1:], face_verts_old))
                f.write(new_face + '\n')
        # Triangulate each polygon as a fan to its own centroid, robust for n==2.
        if loop_infos:
            if fill_material:
                f.write(f"usemtl {fill_material}\n")
            for centroid_idx_new, ordered_new in loop_infos:
                n = len(ordered_new)
                if n == 2:
                    v1, v2 = ordered_new[0], ordered_new[1]
                    f.write(f"f {v1} {v2} {centroid_idx_new}\n")
                else:
                    for i in range(n):
                        v1 = ordered_new[i]
                        v2 = ordered_new[(i + 1) % n]
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
                # Policy switch: choose which teeth to remove
                if AUGMENT_MODE == "match_test":
                    teeth_to_remove = target_missing_teeth.intersection(teeth_present_in_train)
                else:  # balanced policy from professor/supervisor suggestions
                    # sample K in range and select with probabilities (down‑weight wisdom teeth)
                    k_remove = random.randint(REMOVE_K_RANGE[0], REMOVE_K_RANGE[1])
                    cand = sorted(teeth_present_in_train)
                    teeth_to_remove = _sample_teeth_balanced(cand, k_remove)
                # Guard: if nothing to remove, skip this copy but update progress
                if not teeth_to_remove:
                    # if nothing selected (e.g., candidate empty), skip copy but still update progress
                    pbar.update(1)
                    continue
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
