import json
import pandas as pd
import shutil
from pathlib import Path
import numpy as np
import random
import os
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm

# ============= CONFIGURATION =============
TEST_LABELS_CSV = "/home/user/lzhou/week10/label_flipped.csv"
TRAIN_DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]
OUTPUT_DIR = Path("/home/user/lzhou/week16/Aug/augment_test")

FILL_BASE = True
BASE_COLOR = (0.85, 0.70, 0.70)
RANDOM_SEED = 43
NUM_COPIES = 5
CPU_COUNT = os.cpu_count() or 1
DEFAULT_WORKERS = max(1, CPU_COUNT - 1)
WORKERS = int(os.getenv("AUG_WORKERS", str(DEFAULT_WORKERS)))
WORKERS = max(1, WORKERS)
USE_TORCH = True
TORCH_DEVICE = os.getenv("AUG_DEVICE", "cuda")
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

def _read_obj_cached(obj_path):
    with open(obj_path, 'r') as f:
        lines = f.readlines()
    vertex_lines = [l for l in lines if l.startswith('v ')]
    face_lines = [l for l in lines if l.startswith('f ')]
    other_lines = [l for l in lines if not l.startswith(('v ', 'f '))]
    return {
        'lines': lines,
        'vertex_lines': vertex_lines,
        'face_lines': face_lines,
        'other_lines': other_lines,
    }

_TRAIN_SAMPLES = None

def _init_worker(train_samples):
    global _TRAIN_SAMPLES
    _TRAIN_SAMPLES = train_samples


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
    Returns: list of dicts [{'vertices': set([...]), 'edges': [(u,v), ...]}, ...]
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
    from collections import defaultdict
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
        stack = [start]
        comp = set()
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            comp.add(x)
            stack.extend(list(adj.get(x, [])))

        comp_edges = [(u, w) for (u, w) in boundary_edges if u in comp and w in comp]
        if comp_edges:
            components.append({'vertices': comp, 'edges': comp_edges})

    return components

def _face_normal(vids, coord_lut):
    """Compute an (unnormalized) face normal from the first 3 vertices."""
    if len(vids) < 3:
        return None
    v0 = coord_lut.get(vids[0])
    v1 = coord_lut.get(vids[1])
    v2 = coord_lut.get(vids[2])
    if v0 is None or v1 is None or v2 is None:
        return None
    a = np.asarray(v1, dtype=np.float64) - np.asarray(v0, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64) - np.asarray(v0, dtype=np.float64)
    n = np.cross(a, b)
    if np.linalg.norm(n) == 0:
        return None
    return n

def _accumulate_boundary_edge_normals(face_lines, vertices_to_remove, coord_lut, boundary_edge_set):
    edge_normals = {edge: np.zeros(3, dtype=np.float64) for edge in boundary_edge_set}
    if not boundary_edge_set:
        return edge_normals
    for fl in face_lines:
        vids = _parse_face_verts(fl)
        if len(vids) < 3:
            continue
        if any(v in vertices_to_remove for v in vids):
            continue
        n = _face_normal(vids, coord_lut)
        if n is None:
            continue
        m = len(vids)
        for i in range(m):
            a, b = vids[i], vids[(i + 1) % m]
            key = (a, b) if a < b else (b, a)
            if key in edge_normals:
                edge_normals[key] += n
    return edge_normals

def _component_reference_normal(comp_edges, edge_normals, comp_vertices, coord_lut):
    ref = np.zeros(3, dtype=np.float64)
    for u, v in comp_edges:
        key = (u, v) if u < v else (v, u)
        ref += edge_normals.get(key, 0)
    if np.linalg.norm(ref) < 1e-9:
        pts = [coord_lut[v] for v in comp_vertices if v in coord_lut]
        if len(pts) >= 3:
            arr = np.asarray(pts, dtype=np.float64)
            c = arr.mean(axis=0)
            x = arr - c
            cov = x.T @ x
            eigvals, eigvecs = np.linalg.eigh(cov)
            ref = eigvecs[:, int(np.argmin(eigvals))]
    if np.linalg.norm(ref) == 0:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return ref

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

def remove_teeth_from_obj(input_obj, output_obj, teeth_to_remove, all_train_tooth_labels, obj_cache=None):
    """Removes specified teeth from an OBJ file and fills the resulting hole."""
    output_obj = Path(output_obj)
    if not teeth_to_remove:
        shutil.copyfile(input_obj, output_obj)
        return
    vertices_to_remove = {v for tooth in teeth_to_remove for v in all_train_tooth_labels.get(tooth, [])}
    if not vertices_to_remove:
        shutil.copyfile(input_obj, output_obj)
        return
    if obj_cache is None:
        obj_cache = _read_obj_cached(input_obj)
    lines = obj_cache['lines']

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

    vertex_lines = obj_cache['vertex_lines']
    face_lines = obj_cache['face_lines']
    other_lines = obj_cache['other_lines']

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
            coords = [float(c) for c in line.strip().split()[1:4]]
            if len(coords) < 3:
                coords = coords + [0.0] * (3 - len(coords))
            all_coords[old_idx] = coords
            if old_idx not in vertices_to_remove:
                f.write(line)
                vertex_map[old_idx] = new_idx
                new_idx += 1
        boundary_edge_set = set()
        for comp in boundary_components:
            for u, v in comp['edges']:
                key = (u, v) if u < v else (v, u)
                boundary_edge_set.add(key)
        edge_normals = _accumulate_boundary_edge_normals(face_lines, vertices_to_remove, all_coords, boundary_edge_set)

        fill_faces = []
        for comp in boundary_components:
            comp_vertices = comp['vertices']
            comp_edges = comp['edges']
            if len(comp_vertices) < 2 or not comp_edges:
                continue
            coords = [all_coords[v] for v in comp_vertices if v in all_coords]
            if len(coords) < 2:
                continue
            centroid = np.mean(np.asarray(coords, dtype=np.float64), axis=0)
            f.write(f"v {centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f}\n")
            centroid_idx_new = new_idx
            new_idx += 1
            ref_normal = _component_reference_normal(comp_edges, edge_normals, comp_vertices, all_coords)
            for u, v in comp_edges:
                if u not in vertex_map or v not in vertex_map:
                    continue
                u_new = vertex_map[u]
                v_new = vertex_map[v]
                cu = np.asarray(all_coords[u], dtype=np.float64)
                cv = np.asarray(all_coords[v], dtype=np.float64)
                tri_n = np.cross(cv - cu, centroid - cu)
                if np.dot(tri_n, ref_normal) < 0:
                    u_new, v_new = v_new, u_new
                fill_faces.append((u_new, v_new, centroid_idx_new))
        for line in face_lines:
            parts = line.strip().split()
            face_verts_old = [int(p.split('/')[0]) for p in parts[1:]]
            if not any(v in vertices_to_remove for v in face_verts_old):
                new_face = 'f ' + ' '.join(p.replace(str(v_old), str(vertex_map[v_old]), 1) for p, v_old in zip(parts[1:], face_verts_old))
                f.write(new_face + '\n')
        # Cap each boundary edge to the component centroid.
        if fill_faces:
            if fill_material:
                f.write(f"usemtl {fill_material}\n")
            for v1, v2, cidx in fill_faces:
                f.write(f"f {v1} {v2} {cidx}\n")

def _process_test_row(args):
    row_idx, pattern_idx, test_row, value_for_missing = args
    jaw_type = 'lower' if 'lower' in str(test_row.get('filename', '')).lower() else 'upper'
    train_samples = _TRAIN_SAMPLES[jaw_type]
    if not train_samples:
        return []

    rng = random.Random(RANDOM_SEED + row_idx)
    np.random.seed(RANDOM_SEED + row_idx)

    target_missing_teeth = {
        int(col) for col, val in test_row.items()
        if str(col).isdigit() and val == value_for_missing
    }

    obj_cache_by_path = {}
    label_cache_by_json = {}

    def get_obj_cache(obj_path):
        if obj_path not in obj_cache_by_path:
            obj_cache_by_path[obj_path] = _read_obj_cached(obj_path)
        return obj_cache_by_path[obj_path]

    def get_label_cache(json_path):
        if json_path not in label_cache_by_json:
            label_cache_by_json[json_path] = load_tooth_labels_from_json(json_path)
        return label_cache_by_json[json_path]

    rows = []
    for copy_num in range(1, NUM_COPIES + 1):
        train_sample = rng.choice(train_samples)
        original_case_id = train_sample['case_id']
        train_obj_path = train_sample['obj']
        train_json_path = train_sample['json']

        obj_cache = get_obj_cache(train_obj_path)
        all_train_tooth_labels = get_label_cache(train_json_path)
        teeth_present_in_train = set(all_train_tooth_labels.keys())

        if AUGMENT_MODE == "match_test":
            teeth_to_remove = target_missing_teeth.intersection(teeth_present_in_train)
        else:
            k_remove = rng.randint(REMOVE_K_RANGE[0], REMOVE_K_RANGE[1])
            cand = sorted(teeth_present_in_train)
            teeth_to_remove = _sample_teeth_balanced(cand, k_remove)

        if not teeth_to_remove:
            continue

        teeth_present_after = teeth_present_in_train - teeth_to_remove

        output_subdir = OUTPUT_DIR / jaw_type / original_case_id
        output_subdir.mkdir(parents=True, exist_ok=True)
        new_filename_base = f"{original_case_id}_{jaw_type}_pattern_{pattern_idx:04d}_copy_{copy_num:02d}"
        output_obj = output_subdir / f"{new_filename_base}.obj"
        output_json = output_subdir / f"{new_filename_base}.json"

        remove_teeth_from_obj(train_obj_path, output_obj, teeth_to_remove, all_train_tooth_labels, obj_cache)
        create_updated_json_labels(train_json_path, output_json, teeth_to_remove)

        final_flipped_label_dict = {}
        for tooth in ALL_TEETH:
            is_in_correct_jaw = (jaw_type == 'upper' and tooth in UPPER_TEETH) or \
                                (jaw_type == 'lower' and tooth in LOWER_TEETH)
            if is_in_correct_jaw:
                final_flipped_label_dict[str(tooth)] = 1 if tooth not in teeth_present_after else 0
            else:
                final_flipped_label_dict[str(tooth)] = 1

        rows.append({
            'filename': str(output_obj.relative_to(OUTPUT_DIR)),
            'new_id': new_filename_base,
            'Date of labeling': datetime.now().strftime('%Y-%m-%d'),
            'filetype': 'obj',
            **final_flipped_label_dict
        })

    return rows

# =========================================
# MAIN EXECUTION
# =========================================
def main():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    if not Path(TEST_LABELS_CSV).exists(): print(f"[ERROR] CSV file not found: {TEST_LABELS_CSV}"); return

    print("[1/3] Loading all training samples..."); train_samples = load_all_train_samples()
    print(f"  Found {len(train_samples['upper'])} upper and {len(train_samples['lower'])} lower samples.")
    _init_worker(train_samples)
    
    print(f"[2/3] Processing test patterns..."); df_test = pd.read_csv(TEST_LABELS_CSV)
    
    csv_label_rows = []
    
    total_tasks = len(df_test) * NUM_COPIES
    value_for_missing = 1 if FLIPPED_LABELS_IN_CSV else 0
    tasks = []
    for row_idx, (pattern_idx, test_row) in enumerate(df_test.iterrows()):
        tasks.append((row_idx, pattern_idx, test_row.to_dict(), value_for_missing))

    with tqdm(total=total_tasks, desc="Augmenting Data") as pbar:
        if WORKERS <= 1:
            for rows in map(_process_test_row, tasks):
                csv_label_rows.extend(rows)
                pbar.update(len(rows))
        else:
            with mp.Pool(processes=WORKERS, initializer=_init_worker, initargs=(train_samples,)) as pool:
                for rows in pool.imap_unordered(_process_test_row, tasks):
                    csv_label_rows.extend(rows)
                    pbar.update(len(rows))

    print("\n[3/3] All tasks complete. Saving master labels CSV...")
    if csv_label_rows:
        df_labels = pd.DataFrame(csv_label_rows)
        cols = ['filename', 'new_id', 'Date of labeling', 'filetype'] + [str(t) for t in ALL_TEETH]
        df_labels = df_labels[cols]
        output_csv_path = OUTPUT_DIR / "train_labels_augmented.csv"
        df_labels.to_csv(output_csv_path, index=False)
        print(f" Successfully saved master label file with {len(df_labels)} entries to {output_csv_path}")

    print("\n Augmentation process complete!")

if __name__ == "__main__":
    main()
