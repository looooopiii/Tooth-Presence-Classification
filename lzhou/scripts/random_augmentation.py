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
    Robust boundary detection:
      1) Classify each face as 'kept' (no removed verts) or 'removed' (touches any removed vert).
      2) For every edge (u,v) in faces, accumulate counts 'kept_adj', 'removed_adj'.
      3) An edge is a boundary edge if kept_adj>0 and removed_adj>0 and both endpoints are kept.
      4) Build an undirected graph over kept vertices with boundary edges.
      5) Split into connected components and order each as:
         - closed loop (all degree==2): walk cycle;
         - open chain (has deg==1 endpoints): walk from an endpoint to the other;
         - otherwise: greedy walk from an arbitrary start.
    Returns: list of dicts [{'ordered': [v1,...], 'closed': True/False}, ...]
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

    # 5) Connected components and ordering
    components = []
    visited = set()

    def order_closed_cycle(a0):
        loop = [a0]
        prev = None
        cur = a0
        while True:
            nbrs = adj.get(cur, set()) - ({prev} if prev is not None else set())
            if not nbrs:
                break
            nxt = next(iter(nbrs))
            if nxt == a0:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
            if len(loop) > 200000:
                break
        return loop

    def order_open_chain(start):
        seq = [start]
        prev = None
        cur = start
        while True:
            nbrs = adj.get(cur, set()) - ({prev} if prev is not None else set())
            if not nbrs:
                break
            nxt = next(iter(nbrs))
            seq.append(nxt)
            prev, cur = cur, nxt
            if len(seq) > 200000:
                break
        return seq

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

        # degrees within component
        deg = {k: len(adj.get(k, set()) & comp) for k in comp}
        deg_vals = list(deg.values())
        closed = all(d == 2 for d in deg_vals) and len(comp) >= 3

        if closed:
            start = next(iter(comp))
            sub_adj = {k: (adj.get(k, set()) & comp) for k in comp}
            ordered = order_closed_cycle(start)
        else:
            # find endpoint (degree==1) if exists; otherwise pick arbitrary
            endpoints = [k for k, d in deg.items() if d == 1]
            start = endpoints[0] if endpoints else next(iter(comp))
            sub_adj = {k: (adj.get(k, set()) & comp) for k in comp}
            adj_tmp = adj  # reuse
            ordered = order_open_chain(start)

        if len(ordered) >= 2:
            components.append({'ordered': ordered, 'closed': closed})

    return components

def _centroid_of_vertices(old_ids, coord_lut):
    pts = [coord_lut[v] for v in old_ids if v in coord_lut]
    if len(pts) == 0:
        return None
    import numpy as _np
    arr = _np.asarray(pts, dtype=_np.float64)
    c = arr.mean(axis=0)
    return c

def remove_teeth_from_obj(input_obj, output_obj, teeth_to_remove, all_train_tooth_labels):
    """Removes specified teeth from an OBJ file and fills the resulting hole."""
    output_obj = Path(output_obj)
    if not teeth_to_remove: shutil.copyfile(input_obj, output_obj); return
    vertices_to_remove = {v for tooth in teeth_to_remove for v in all_train_tooth_labels.get(tooth, [])}
    if not vertices_to_remove: shutil.copyfile(input_obj, output_obj); return
    with open(input_obj, 'r') as f: lines = f.readlines()
    vertex_lines, face_lines, other_lines = [l for l in lines if l.startswith('v ')], [l for l in lines if l.startswith('f ')], [l for l in lines if not l.startswith(('v ', 'f '))]
    # robust boundary components (closed loops or open chains)
    boundary_components = _build_boundary_components(vertices_to_remove, face_lines)
    with open(output_obj, 'w') as f:
        mtl_path = output_obj.with_suffix('.mtl'); f.write(f"mtllib {mtl_path.name}\n")
        with open(mtl_path, 'w') as mtl_f: mtl_f.write(f"newmtl Gingiva\nKd {BASE_COLOR[0]} {BASE_COLOR[1]} {BASE_COLOR[2]}\nKs 0.02 0.02 0.02\nNs 10.0\nillum 2\n")
        for line in other_lines: f.write(line)
        vertex_map, new_idx, all_coords = {}, 1, {}
        for old_idx, line in enumerate(vertex_lines, 1):
            all_coords[old_idx] = [float(c) for c in line.strip().split()[1:]]
            if old_idx not in vertices_to_remove: f.write(line); vertex_map[old_idx] = new_idx; new_idx += 1

        # Build fill fans for each boundary component (closed loop or open chain)
        loop_infos = []
        for comp in boundary_components:
            loop_old = comp['ordered']
            # Need at least 2 points to form triangles with a centroid
            if len(loop_old) < 2:
                continue
            centroid = _centroid_of_vertices(loop_old, all_coords)
            if centroid is None:
                continue
            f.write(f"v {centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f}\n")
            centroid_idx_new = new_idx
            new_idx += 1

            ordered_new = []
            for v_old in loop_old:
                if v_old in vertices_to_remove:
                    continue
                if v_old in vertex_map:
                    ordered_new.append(vertex_map[v_old])

            # need >=2 edges to form at least one triangle with centroid
            if len(ordered_new) >= 2:
                loop_infos.append((centroid_idx_new, ordered_new))

        for line in face_lines:
            parts = line.strip().split(); face_verts_old = [int(p.split('/')[0]) for p in parts[1:]]
            if not any(v in vertices_to_remove for v in face_verts_old):
                new_face = 'f ' + ' '.join(p.replace(str(v_old), str(vertex_map[v_old]), 1) for p, v_old in zip(parts[1:], face_verts_old))
                f.write(new_face + '\n')
        # Triangulate each component as a fan to its centroid.
        if loop_infos:
            f.write("usemtl Gingiva\n")
            for centroid_idx_new, ordered_new in loop_infos:
                n = len(ordered_new)
                if n == 2:
                    # create a degenerate cap (single triangle) to close tiny slits
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
