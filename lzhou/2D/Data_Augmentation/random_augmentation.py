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
ORIGINAL_TRAIN_DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]
OUTPUT_DIR = Path("/home/user/lzhou/week16/Aug/augment_random")

FILL_BASE = True
BASE_COLOR = (0.85, 0.70, 0.70)
RANDOM_SEED = 42
COPIES_PER_SCAN = 2
DEBUG_LIMIT = None  # Set to an integer for debugging with fewer samples
CPU_COUNT = os.cpu_count() or 1
DEFAULT_WORKERS = max(1, CPU_COUNT - 1)
WORKERS = int(os.getenv("AUG_WORKERS", str(DEFAULT_WORKERS)))
WORKERS = max(1, WORKERS)

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

def _order_loop(adj, start):
    """
    Given an undirected adjacency mapping {v: set(neighbors)} for a single loop,
    return an ordered list of vertices following the boundary.
    """
    loop = [start]
    prev = None
    cur = start
    while True:
        nbrs = adj.get(cur, set()) - ({prev} if prev is not None else set())
        if not nbrs:
            break
        nxt = next(iter(nbrs))
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > 200000:  # safety guard
            break
    return loop

def extract_boundary_loops(vertices_to_remove, face_lines):
    """
    Build ordered boundary loops (each loop is a list of kept-vertex indices)
    around regions that will be removed. We collect edges from faces that mix
    removed and kept vertices, then split them into connected components and
    order each component along the loop.
    """
    # Parse faces into lists of vertex indices (tolerate v/vt/vn tokens)
    faces = []
    for fl in face_lines:
        parts = fl.strip().split()[1:]
        if not parts:
            continue
        vids = []
        for p in parts:
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

    # Extract connected components and order them as loops
    visited = set()
    loops = []
    for v in list(adj.keys()):
        if v in visited:
            continue
        # BFS to get component
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
        start = next(iter(comp))
        sub_adj = {k: (adj.get(k, set()) & comp) for k in comp}
        ordered = _order_loop(sub_adj, start)
        if len(ordered) >= 3:
            loops.append(ordered)
    return loops

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
    # 1) Parse faces and classify
    faces = []
    face_types = []  # 'kept' | 'removed'
    for fl in face_lines:
        vids = _parse_face_verts(fl)
        if len(vids) < 2:
            continue
        faces.append(vids)
        is_removed = any(v in vertices_to_remove for v in vids)
        face_types.append('removed' if is_removed else 'kept')

    # 2) Edge counts
    from collections import defaultdict
    edge_cnt = defaultdict(lambda: {'kept': 0, 'removed': 0})
    for vids, ftype in zip(faces, face_types):
        n = len(vids)
        for i in range(n):
            a, b = vids[i], vids[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            edge_cnt[key][ftype] += 1

    # 3) Boundary edges (kept vs removed on the two sides)
    boundary_edges = []
    for (u, v), cnt in edge_cnt.items():
        if cnt['kept'] > 0 and cnt['removed'] > 0:
            if (u not in vertices_to_remove) and (v not in vertices_to_remove):
                boundary_edges.append((u, v))

    # 4) Build graph
    adj = {}
    for u, v in boundary_edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    # 5) Connected components and collect edges
    components = []
    visited = set()
    nodes = set(adj.keys())
    for v in list(nodes):
        if v in visited:
            continue
        # BFS to get component vertices
        stack = [v]
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

def _centroid_of_vertices(old_ids, coord_lut):
    pts = [coord_lut[v] for v in old_ids if v in coord_lut]
    if len(pts) == 0:
        return None
    import numpy as _np
    arr = _np.asarray(pts, dtype=_np.float64)
    c = arr.mean(axis=0)
    return c

def remove_teeth_from_obj(input_obj, output_obj, teeth_to_remove, all_train_tooth_labels, obj_cache=None):
    """Removes specified teeth from an OBJ file and fills the resulting hole."""
    output_obj = Path(output_obj)
    if not teeth_to_remove: shutil.copyfile(input_obj, output_obj); return
    vertices_to_remove = {v for tooth in teeth_to_remove for v in all_train_tooth_labels.get(tooth, [])}
    if not vertices_to_remove: shutil.copyfile(input_obj, output_obj); return
    if obj_cache is None:
        obj_cache = _read_obj_cached(input_obj)
    vertex_lines = obj_cache['vertex_lines']
    face_lines = obj_cache['face_lines']
    other_lines = obj_cache['other_lines']
    # boundary components from kept/removed adjacency
    boundary_components = _build_boundary_components(vertices_to_remove, face_lines)
    with open(output_obj, 'w') as f:
        mtl_path = output_obj.with_suffix('.mtl'); f.write(f"mtllib {mtl_path.name}\n")
        with open(mtl_path, 'w') as mtl_f: mtl_f.write(f"newmtl Gingiva\nKd {BASE_COLOR[0]} {BASE_COLOR[1]} {BASE_COLOR[2]}\nKs 0.02 0.02 0.02\nNs 10.0\nillum 2\n")
        for line in other_lines: f.write(line)
        vertex_map, new_idx, all_coords = {}, 1, {}
        for old_idx, line in enumerate(vertex_lines, 1):
            coords = [float(c) for c in line.strip().split()[1:4]]
            if len(coords) < 3:
                coords = coords + [0.0] * (3 - len(coords))
            all_coords[old_idx] = coords
            if old_idx not in vertices_to_remove: f.write(line); vertex_map[old_idx] = new_idx; new_idx += 1

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
            parts = line.strip().split(); face_verts_old = [int(p.split('/')[0]) for p in parts[1:]]
            if not any(v in vertices_to_remove for v in face_verts_old):
                new_face = 'f ' + ' '.join(p.replace(str(v_old), str(vertex_map[v_old]), 1) for p, v_old in zip(parts[1:], face_verts_old))
                f.write(new_face + '\n')
        # Cap each boundary edge to the component centroid.
        if fill_faces:
            f.write("usemtl Gingiva\n")
            for v1, v2, cidx in fill_faces:
                f.write(f"f {v1} {v2} {cidx}\n")

def _process_train_sample(args):
    train_sample, sample_idx = args
    jaw_type = train_sample['jaw']
    original_case_id = train_sample['case_id']
    train_obj_path = train_sample['obj']
    train_json_path = train_sample['json']

    obj_cache = _read_obj_cached(train_obj_path)
    all_train_tooth_labels = load_tooth_labels_from_json(train_json_path)
    teeth_present_in_train = set(all_train_tooth_labels.keys())

    candidate_teeth = UPPER_TEETH if jaw_type == 'upper' else LOWER_TEETH
    possible_to_remove = [tooth for tooth in candidate_teeth if tooth in teeth_present_in_train]
    if not possible_to_remove:
        return []

    weights = [TEETH_REMOVAL_PROBS.get(tooth, BASE_PROB) for tooth in possible_to_remove]
    rng = random.Random(RANDOM_SEED + sample_idx)

    rows = []
    output_subdir = OUTPUT_DIR / jaw_type / original_case_id
    output_subdir.mkdir(parents=True, exist_ok=True)

    for copy_num in range(1, COPIES_PER_SCAN + 1):
        num_to_remove = rng.randint(2, min(5, len(possible_to_remove)))
        selected_for_removal = set(rng.choices(possible_to_remove, weights=weights, k=num_to_remove))
        if not selected_for_removal:
            continue
        teeth_present_after = teeth_present_in_train - selected_for_removal

        new_filename_base = f"{original_case_id}_{jaw_type}_randomcopy_{copy_num:02d}"
        output_obj = output_subdir / f"{new_filename_base}.obj"
        output_json = output_subdir / f"{new_filename_base}.json"

        remove_teeth_from_obj(train_obj_path, output_obj, selected_for_removal, all_train_tooth_labels, obj_cache)
        create_updated_json_labels(train_json_path, output_json, selected_for_removal)

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[1/3] Loading original training samples..."); train_samples = load_all_train_samples()
    print(f"  Found {len(train_samples['upper'])} upper and {len(train_samples['lower'])} lower samples.")
    
    csv_label_rows = []
    
    all_original_scans = train_samples['upper'] + train_samples['lower']
    if DEBUG_LIMIT is not None:
        random.shuffle(all_original_scans)
        all_original_scans = all_original_scans[:DEBUG_LIMIT]
    total_tasks = len(all_original_scans) * COPIES_PER_SCAN
    
    print(f"[2/3] Starting random augmentation for {total_tasks} new scans...")
    tasks = [(sample, idx) for idx, sample in enumerate(all_original_scans)]
    with tqdm(total=total_tasks, desc="Augmenting") as pbar:
        if WORKERS <= 1:
            for rows in map(_process_train_sample, tasks):
                csv_label_rows.extend(rows)
                pbar.update(len(rows))
        else:
            with mp.Pool(processes=WORKERS) as pool:
                for rows in pool.imap_unordered(_process_train_sample, tasks):
                    csv_label_rows.extend(rows)
                    pbar.update(len(rows))

    print("\n[3/3] All augmentation tasks complete. Saving master labels CSV...")
    if csv_label_rows:
        df_labels = pd.DataFrame(csv_label_rows)
        # Ensure correct column order, converting tooth numbers to strings for lookup
        cols = ['filename', 'new_id', 'Date of labeling', 'filetype'] + [str(t) for t in ALL_TEETH]
        df_labels = df_labels[cols]
        output_csv_path = OUTPUT_DIR / "train_labels_random.csv"
        df_labels.to_csv(output_csv_path, index=False)
        print(f" Successfully saved master label file with {len(df_labels)} entries to {output_csv_path}")

    print("\n Random augmentation process complete!")

if __name__ == "__main__":
    main()
