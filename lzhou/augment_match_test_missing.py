import os, csv, json, random, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------- Config -----------------
BASE_DIR = os.environ.get(
    "BATCH_BASE_DIR",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split"
)
OUT_ROOT = Path(os.environ.get(
    "OUT_ROOT",
    "/home/user/lzhou/week9/data_augmentation/Aug_data"
))
TEST_LABELS_CSV = os.environ.get(
    "TEST_LABELS_CSV",
    "/home/user/lzhou/week9/data_augmentation/test_labels_flipped.csv"
)  # 1=missing, 0=present, empty=unknown
AUG_MIN_VARIANTS = int(os.environ.get("AUG_MIN_VARIANTS", "2"))
AUG_MAX_VARIANTS = int(os.environ.get("AUG_MAX_VARIANTS", "5"))
TEETH_TO_REMOVE_PER_VARIANT = int(os.environ.get("TEETH_PER_VARIANT", "1"))
RNG_SEED = int(os.environ.get("RNG_SEED", "42"))
BATCH_LIMIT = int(os.environ.get("BATCH_LIMIT", "0"))  # 0=all
FILL_MODE = os.environ.get("FILL_MODE", "delete_white").strip().lower()  # 'delete_white' | 'delete' | 'white'
USE_GPU = os.environ.get("USE_GPU", "1").strip() != "0"

UPPER_FDI = set(list(range(11, 19)) + list(range(21, 29)))
LOWER_FDI = set(list(range(31, 39)) + list(range(41, 49)))
# ------------------------------------------

def _get_blender_bin():
    env = os.environ.get("BLENDER_BIN")
    return os.path.expanduser(env) if env else "blender"

def load_test_missing_patterns(csv_path):
    df = pd.read_csv(csv_path)
    tooth_cols = []
    for c in df.columns:
        s = str(c).strip().lower()
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits or s.startswith("tooth_"):
            tooth_cols.append(c)
    upper, lower = [], []
    for _, row in df[tooth_cols].iterrows():
        mu, ml = [], []
        for c in tooth_cols:
            key = str(c).strip().lower()
            digits = "".join(ch for ch in key if ch.isdigit())
            if not digits: 
                continue
            fdi = int(digits)
            v = row[c]
            miss = False
            if pd.notna(v) and str(v).strip() != "":
                try: miss = int(v) == 1
                except: miss = str(v).strip() == "1"
            if not miss: 
                continue
            if fdi in UPPER_FDI: mu.append(f"tooth_{fdi}")
            elif fdi in LOWER_FDI: ml.append(f"tooth_{fdi}")
        if mu: upper.append(mu)
        if ml: lower.append(ml)
    print(f"[TestPatterns] upper={len(upper)} lower={len(lower)} from {csv_path}")
    return {"upper": upper, "lower": lower}

def read_tooth_vertices_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    labels = data.get("labels", [])
    m = {}
    for i, lab in enumerate(labels):
        if lab and int(lab) != 0:
            name = f"tooth_{int(lab)}"
            m.setdefault(name, []).append(i + 1)  # OBJ 1-based
    return m

def build_whitefill_obj(input_obj, out_obj, selected_teeth, tooth_vertices):
    """Create a copy of input_obj, but tag faces that reference selected teeth's vertices as WHITEBASE material."""
    out_obj = Path(out_obj)
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = out_obj.with_suffix(".mtl"); mtl_name = mtl_path.name
    selected_v = set()
    for t in selected_teeth:
        selected_v.update(tooth_vertices.get(t, []))
    if not selected_v:
        shutil.copyfile(input_obj, out_obj); return True

    with open(input_obj, "r") as f:
        lines = f.readlines()
    out_lines, wrote_mtllib, current_mtl = [], False, None

    def set_mtl(name):
        nonlocal current_mtl
        if current_mtl != name:
            out_lines.append(f"usemtl {name}\n")
            current_mtl = name

    for line in lines:
        if line.startswith("mtllib "): 
            continue
        if not wrote_mtllib and line.startswith("v "):
            out_lines.append(f"mtllib {mtl_name}\n"); wrote_mtllib = True
        if line.startswith("f "):
            parts = line.strip().split()
            vids, ok = [], True
            for p in parts[1:]:
                vref = p.split("/")[0]
                try: vids.append(int(vref))
                except: ok = False; break
            if ok and vids:
                if any((vid in selected_v) for vid in vids): set_mtl("WHITEBASE")
                else: set_mtl("DEFAULT")
            out_lines.append(line if line.endswith("\n") else line+"\n")
        else:
            out_lines.append(line if line.endswith("\n") else line+"\n")

    with open(out_obj, "w") as f:
        f.writelines(out_lines)
    with open(mtl_path, "w") as f:
        f.write("# Auto MTL\n")
        f.write("newmtl WHITEBASE\nKa 0.98 0.98 0.98\nKd 0.98 0.98 0.98\nKs 0.00 0.00 0.00\nNs 1.0\nd 1.0\nillum 1\n")
        f.write("\nnewmtl DEFAULT\nKa 0.75 0.72 0.70\nKd 0.80 0.78 0.75\nKs 0.20 0.20 0.20\nNs 200.0\nd 1.0\nillum 2\n")
    return True


def modify_obj_by_labels(input_path, output_path, teeth_to_remove_names, tooth_labels):
    """Delete vertices of selected teeth (labels !=0), drop faces that reference them,
    and remap surviving vertex indices (keeps gingiva, so sockets remain)."""
    print(f"[Step] Delete-mode OBJ: {input_path} -> {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not teeth_to_remove_names:
        shutil.copyfile(input_path, output_path)
        print("[Info] No teeth selected. Copied original.")
        return True

    remove_v = set()
    for t in teeth_to_remove_names:
        remove_v.update(tooth_labels.get(t, []))

    if not remove_v:
        print("[Warn] Selected teeth have no vertices; copy original.")
        shutil.copyfile(input_path, output_path)
        return True

    with open(input_path, 'r') as f:
        lines = f.readlines()

    vertex_mapping = {}  # old->new
    new_idx = 1
    old_idx = 0
    out_lines = []

    for line in lines:
        if line.startswith('v '):
            old_idx += 1
            if old_idx in remove_v:
                continue
            vertex_mapping[old_idx] = new_idx
            new_idx += 1
            out_lines.append(line)
        elif line.startswith('f '):
            parts = line.strip().split()
            valid = True
            remapped = ['f']
            for p in parts[1:]:
                vref = p.split('/')[0]
                try:
                    vold = int(vref)
                except ValueError:
                    remapped.append(p)
                    continue
                if vold not in vertex_mapping:
                    valid = False; break
                remapped.append(p.replace(vref, str(vertex_mapping[vold]), 1))
            if valid and len(remapped) > 3:
                out_lines.append(' '.join(remapped) + '\n')
        else:
            out_lines.append(line if line.endswith('\n') else line+'\n')

    with open(output_path, 'w') as f:
        f.writelines(out_lines)

    print(f"[Info] Deleted vertices: {len(remove_v)}; Kept: {new_idx-1}")
    return True

# --- New function: delete_and_whitefill_obj ---
def delete_and_whitefill_obj(input_path, output_path, teeth_to_remove_names, tooth_labels):
    """Delete selected teeth, then paint the *border gingiva faces* white.
    Implementation: two-pass.
    - Pass1: parse vertices and faces to find kept vertices and record 'border_kept' old indices
             (faces that mix removed + kept vertices contribute their kept vertices to border set).
    - Pass2: write OBJ with remapped indices; faces whose OLD indices intersect border_kept are tagged WHITEBASE,
             others DEFAULT. Also write an MTL with WHITEBASE (pure white) and DEFAULT (slightly gray) materials.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = output_path.with_suffix('.mtl'); mtl_name = mtl_path.name

    # Collect vertices to remove
    remove_v = set()
    for t in teeth_to_remove_names:
        remove_v.update(tooth_labels.get(t, []))

    # Read
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # ------- Pass 1: detect border kept vertices -------
    old_idx = 0
    kept_vertices = set()
    for line in lines:
        if line.startswith('v '):
            old_idx += 1
            if old_idx not in remove_v:
                kept_vertices.add(old_idx)
    border_kept = set()
    for line in lines:
        if line.startswith('f '):
            parts = line.strip().split()[1:]
            olds = []
            ok = True
            for p in parts:
                vref = p.split('/')[0]
                try:
                    olds.append(int(vref))
                except ValueError:
                    ok = False; break
            if not ok or len(olds) < 3:
                continue
            has_removed = any((o in remove_v) for o in olds)
            has_kept    = any((o in kept_vertices) for o in olds)
            if has_removed and has_kept:
                for o in olds:
                    if o in kept_vertices:
                        border_kept.add(o)

    # ------- Pass 2: remap + write with materials -------
    vertex_mapping = {}
    new_idx = 1
    wrote_mtllib = False
    current_mtl = None
    out_lines = []

    def set_mtl(name):
        nonlocal current_mtl
        if current_mtl != name:
            out_lines.append(f"usemtl {name}\n")
            current_mtl = name

    old_idx = 0
    for line in lines:
        if line.startswith('mtllib '):
            continue
        if not wrote_mtllib and line.startswith('v '):
            out_lines.append(f"mtllib {mtl_name}\n"); wrote_mtllib = True
        if line.startswith('v '):
            old_idx += 1
            if old_idx in remove_v:
                continue
            vertex_mapping[old_idx] = new_idx
            new_idx += 1
            out_lines.append(line)
        elif line.startswith('f '):
            parts = line.strip().split()[1:]
            olds = []
            remapped = ['f']
            valid = True
            for p in parts:
                vref = p.split('/')[0]
                try:
                    o = int(vref)
                except ValueError:
                    remapped.append(p)
                    continue
                if o not in vertex_mapping:
                    valid = False; break
                remapped.append(p.replace(vref, str(vertex_mapping[o]), 1))
                olds.append(o)
            if valid and len(remapped) > 3:
                if any((o in border_kept) for o in olds):
                    set_mtl('WHITEBASE')
                else:
                    set_mtl('DEFAULT')
                out_lines.append(' '.join(remapped) + '\n')
        else:
            out_lines.append(line if line.endswith('\n') else line+'\n')

    with open(output_path, 'w') as f:
        f.writelines(out_lines)

    with open(mtl_path, 'w') as f:
        f.write('# Auto-generated MTL (delete+white border)\n')
        f.write('newmtl WHITEBASE\nKa 0.98 0.98 0.98\nKd 0.98 0.98 0.98\nKs 0.00 0.00 0.00\nNs 1.0\nd 1.0\nillum 1\n')
        f.write('\nnewmtl DEFAULT\nKa 0.75 0.72 0.70\nKd 0.80 0.78 0.75\nKs 0.20 0.20 0.20\nNs 200.0\nd 1.0\nillum 2\n')

    print(f"[Info] Delete+White: removed {len(remove_v)} vertices; border faces painted white.")
    return True

def create_blender_script(obj_path, output_image_path):
    abs_obj = os.path.abspath(obj_path)
    abs_out = os.path.abspath(output_image_path)
    return f'''
import bpy, mathutils, math, os, sys

def render_model(obj_path, output_path):
    print(f"[Blender] Start render: {{obj_path}}")
    # Clean scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Import OBJ with consistent axes
    try:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='-Z', axis_up='Y')
    except Exception as e:
        print("[Blender][Error] Import failed:", e)
        return False

    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        print("[Blender][Error] No mesh after import.")
        return False

    # Apply transforms, compute bbox and center
    minc = [float('inf')]*3
    maxc = [float('-inf')]*3
    for o in meshes:
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        for c in [o.matrix_world @ mathutils.Vector(v) for v in o.bound_box]:
            for i in range(3):
                minc[i] = min(minc[i], c[i])
                maxc[i] = max(maxc[i], c[i])

    center = mathutils.Vector([(minc[i]+maxc[i])/2 for i in range(3)])
    size  = mathutils.Vector([maxc[i]-minc[i] for i in range(3)])
    max_dim = max(size) if max(size) > 0 else 1.0

    for o in meshes:
        o.location -= center

    # --- Align thinnest bbox axis to Z to guarantee true TOP view ---
    size_vec = mathutils.Vector([maxc[i]-minc[i] for i in range(3)])
    thin_axis = min(range(3), key=lambda i: size_vec[i])
    if thin_axis == 0:  # X thinnest -> rotate +90° around Y (bring X to Z)
        for o in meshes:
            o.rotation_euler.rotate_axis('Y', math.radians(90))
        bpy.context.view_layer.update()
    elif thin_axis == 1:  # Y thinnest -> rotate -90° around X (bring Y to Z)
        for o in meshes:
            o.rotation_euler.rotate_axis('X', math.radians(-90))
        bpy.context.view_layer.update()

    # Re-apply transforms and re-center after alignment
    minc = [float('inf')]*3
    maxc = [float('-inf')]*3
    for o in meshes:
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        for c in [o.matrix_world @ mathutils.Vector(v) for v in o.bound_box]:
            for i in range(3):
                minc[i] = min(minc[i], c[i])
                maxc[i] = max(maxc[i], c[i])
    center = mathutils.Vector([(minc[i]+maxc[i])/2 for i in range(3)])
    size  = mathutils.Vector([maxc[i]-minc[i] for i in range(3)])
    max_dim = max(size) if max(size) > 0 else 1.0
    for o in meshes:
        o.location -= center
    
    # --- If delete_white mode: cap open holes and paint caps WHITE ---
    APPLY_CAP_WHITE = {str(FILL_MODE == 'delete_white')}
    if APPLY_CAP_WHITE:
        # Ensure a WHITEBASE material exists (or create one)
        white_mat = bpy.data.materials.get('WHITEBASE')
        if white_mat is None:
            white_mat = bpy.data.materials.new(name='WHITEBASE')
            white_mat.use_nodes = True
            nodes = white_mat.node_tree.nodes
            links = white_mat.node_tree.links
            for n in list(nodes):
                nodes.remove(n)
            diff = nodes.new(type='ShaderNodeBsdfDiffuse')
            diff.inputs['Color'].default_value = (0.97, 0.97, 0.97, 1.0)
            outp = nodes.new(type='ShaderNodeOutputMaterial')
            links.new(diff.outputs['BSDF'], outp.inputs['Surface'])

        import bmesh
        any_still_open = False
        for o in meshes:
            if o.type != 'MESH':
                continue
            # Ensure WHITEBASE slot exists and get index
            if white_mat.name not in [m.name for m in (o.data.materials or [])]:
                o.data.materials.append(white_mat)
            white_index = [i for i,m in enumerate(o.data.materials) if m and m.name==white_mat.name][0]

            me = o.data
            bm = bmesh.new()
            bm.from_mesh(me)
            bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table(); bm.faces.ensure_lookup_table()

            # Collect boundary edges (open edges)
            boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]

            # Group boundary edges into connected loops
            unvisited = set(boundary_edges)
            loops = []
            while unvisited:
                e0 = unvisited.pop()
                comp = set([e0])
                stack = [e0.verts[0], e0.verts[1]]
                while stack:
                    v = stack.pop()
                    for e in v.link_edges:
                        if e in unvisited and len(e.link_faces) == 1:
                            unvisited.remove(e)
                            comp.add(e)
                            # push both vertices so we flood along the boundary
                            stack.append(e.verts[0]); stack.append(e.verts[1])
                loops.append(list(comp))

            # Helper to compute span (max bbox edge) and perimeter of a loop
            def loop_stats(edges):
                verts = set()
                for e in edges:
                    verts.add(e.verts[0]); verts.add(e.verts[1])
                xs=[v.co.x for v in verts]; ys=[v.co.y for v in verts]; zs=[v.co.z for v in verts]
                span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
                per = sum(e.calc_length() for e in edges)
                return span, per

            # Compute stats for all boundary loops
            loop_infos = []  # (edges, span, per)
            for edges in loops:
                span, per = loop_stats(edges)
                loop_infos.append((edges, span, per))

            # Identify the single largest outer loop by span (tie-breaker by perimeter)
            skip_idx = None
            if loop_infos:
                skip_idx = max(range(len(loop_infos)), key=lambda i: (loop_infos[i][1], loop_infos[i][2]))

            # Cap all loops except the outermost one
            capped_any = False
            new_faces_total = []
            for i, (edges, span, per) in enumerate(loop_infos):
                if i == skip_idx:
                    continue
                res = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
                new_faces = res.get('faces', [])
                # Fallback: if holes_fill produced nothing, try edgenet_fill
                if not new_faces:
                    try:
                        res2 = bmesh.ops.edgenet_fill(bm, edges=edges)
                        new_faces = res2.get('faces', [])
                    except Exception:
                        new_faces = []
                if new_faces:
                    capped_any = True
                    for f in new_faces:
                        f.material_index = white_index
                    new_faces_total.extend(new_faces)

            # Robustness: recalc normals and triangulate newly added faces
            if new_faces_total:
                bmesh.ops.recalc_face_normals(bm, faces=new_faces_total)
                bmesh.ops.triangulate(bm, faces=new_faces_total)
            # Check if any boundary edges still remain after attempts
            still_open = any(len(e.link_faces) == 1 for e in bm.edges)
            bm.to_mesh(me)
            bm.free()
            if still_open:
                any_still_open = True


        # (World background darkening logic removed, always use black background later.)

    APPLY_GUMMAT = {str(FILL_MODE == 'delete')}
    if APPLY_GUMMAT:
        # Apply a neutral gray glossy material so deleted sockets show depth clearly
        mat = bpy.data.materials.new(name="GumMat")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for n in nodes: nodes.remove(n)
        diff = nodes.new(type='ShaderNodeBsdfDiffuse')
        diff.inputs['Color'].default_value = (0.78, 0.78, 0.78, 1.0)
        glossy = nodes.new(type='ShaderNodeBsdfGlossy')
        glossy.inputs['Roughness'].default_value = 0.30
        mix = nodes.new(type='ShaderNodeMixShader')
        mix.inputs['Fac'].default_value = 0.16
        outp = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(diff.outputs['BSDF'], mix.inputs[1])
        links.new(glossy.outputs['BSDF'], mix.inputs[2])
        links.new(mix.outputs['Shader'], outp.inputs['Surface'])
        for o in meshes:
            if o.type=='MESH':
                if o.data.materials:
                    o.data.materials[0] = mat
                else:
                    o.data.materials.append(mat)
                bpy.ops.object.select_all(action='DESELECT')
                o.select_set(True)
                bpy.context.view_layer.objects.active = o
                bpy.ops.object.shade_smooth()

    # Upgrade DEFAULT material to tooth-like glossy+diffuse for better detail
    for o in meshes:
        if o.type != 'MESH':
            continue
        for mat in (o.data.materials or []):
            if not mat or mat.name != 'DEFAULT':
                continue
            try:
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                for n in list(nodes):
                    nodes.remove(n)

                # Base diffuse
                n_diff = nodes.new(type='ShaderNodeBsdfDiffuse')
                n_diff.inputs['Color'].default_value = (0.95, 0.92, 0.87, 1.0)

                # Small glossy
                n_gloss = nodes.new(type='ShaderNodeBsdfGlossy')
                n_gloss.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
                n_gloss.inputs['Roughness'].default_value = 0.30

                # Pointiness + AO to emphasize creases
                n_geom = nodes.new(type='ShaderNodeNewGeometry')
                n_ramp = nodes.new(type='ShaderNodeValToRGB')
                n_ramp.color_ramp.elements[0].position = 0.35
                n_ramp.color_ramp.elements[1].position = 0.75
                n_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1)
                n_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1)

                n_ao = nodes.new(type='ShaderNodeAmbientOcclusion')
                n_ao.inputs['Distance'].default_value = 5.0

                # Darken cavities by multiplying a slightly darker tint
                n_mul = nodes.new(type='ShaderNodeMixRGB')
                n_mul.blend_type = 'MULTIPLY'
                n_mul.inputs['Fac'].default_value = 0.5
                n_mul.inputs['Color2'].default_value = (0.85, 0.83, 0.80, 1.0)

                # Combine pointiness & AO as mask
                n_mask = nodes.new(type='ShaderNodeMixRGB')
                n_mask.blend_type = 'MULTIPLY'
                n_mask.inputs['Fac'].default_value = 1.0

                n_mix = nodes.new(type='ShaderNodeMixShader')
                n_mix.inputs['Fac'].default_value = 0.16

                n_out = nodes.new(type='ShaderNodeOutputMaterial')

                # Links
                links.new(n_geom.outputs['Pointiness'], n_ramp.inputs['Fac'])
                links.new(n_ramp.outputs['Color'], n_mask.inputs['Color1'])
                links.new(n_ao.outputs['Color'],  n_mask.inputs['Color2'])

                links.new(n_diff.outputs['BSDF'], n_mul.inputs['Color1'])
                links.new(n_mask.outputs['Color'], n_mul.inputs['Fac'])
                links.new(n_mul.outputs['Color'], n_diff.inputs['Color'])

                links.new(n_diff.outputs['BSDF'], n_mix.inputs[1])
                links.new(n_gloss.outputs['BSDF'], n_mix.inputs[2])
                links.new(n_mix.outputs['Shader'], n_out.inputs['Surface'])
            except Exception:
                pass

    # ---------------- Camera: strict TOP occlusal ----------------
    # Auto-detect jaw from filename; default to lower if unknown
    name_l = os.path.basename(obj_path).lower()
    jaw = "lower" if "_lower" in name_l else ("upper" if "_upper" in name_l else "lower")
    cam_dist = max_dim * 2.4  # slightly farther to keep full arch in frame

    cam_z = cam_dist if jaw == "lower" else -cam_dist
    bpy.ops.object.camera_add(location=(0.0, 0.0, cam_z))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = (-cam.location).normalized().to_track_quat('-Z','Y')

    # Perspective top view for more surface detail (telephoto)
    cam.data.type = 'PERSP'
    cam.data.lens = 80  # a touch wider to avoid cropping
    cam.data.sensor_width = 36
    # keep distance based on bbox

    # ---------------- Lighting: balanced 3-point (dimmer for detail) ----------------
    # Key from camera direction
    bpy.ops.object.light_add(type='SUN', location=(0, 0, cam_z))
    key = bpy.context.object
    key.data.energy = 3.8
    key.data.angle = 0.16

    # Fill from left-top to lift shadows (lower energy)
    bpy.ops.object.light_add(type='AREA', location=(-max_dim*0.8, -max_dim*0.4, cam_z*0.8))
    fill = bpy.context.object
    fill.data.energy = 650
    fill.data.size = max_dim * 0.60

    # Rim from right-bottom to add edge separation (subtle)
    bpy.ops.object.light_add(type='SUN', location=(max_dim*0.8, max_dim*0.4, cam_z*1.2))
    rim = bpy.context.object
    rim.data.energy = 1.2
    rim.data.angle = 0.34

    # ---------------- Render settings & world ----------------
    scene = bpy.context.scene
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 2048
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 48
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.sample_clamp_direct = 1.0
    scene.cycles.sample_clamp_indirect = 2.0
    bpy.context.view_layer.cycles.use_denoising = True

    # Black background
    scene.render.film_transparent = False
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if not bg:
        bg = world.node_tree.nodes.new('ShaderNodeBackground')
    bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)  # black
    bg.inputs[1].default_value = 1.0

    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'
    scene.view_settings.exposure = -0.78

    try:
        bpy.ops.render.render(write_still=True)
    except Exception as e:
        print("[Blender][Error] Render failed:", e)
        return False

    print(f"[Blender] Done: {abs_out}")
    return True

ok = render_model("{abs_obj}", "{abs_out}")
if not ok:
    import sys; sys.exit(2)
'''

def render_with_blender(obj_path, out_png):
    script_path = OUT_ROOT / "blender_render_tmp.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(create_blender_script(obj_path, out_png))
    cmd = [_get_blender_bin(), "--background", "--python", str(script_path)]
    print("[Blender] run:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("----- blender stdout -----\n", res.stdout)
        print("----- blender stderr -----\n", res.stderr)
        return False
    return True

def ensure_manifest(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "case_id","variant_id","removed_teeth","obj_path","image_path","jaw","seed","mode"
            ]).writeheader()

def choose_teeth(available_teeth, patterns_list):
    """Sample one test-row pattern (with replacement) and remove ALL intersecting teeth.
    If intersection empty, fall back to removing 1 random available tooth."""
    if not available_teeth:
        return []
    sampled = random.choice(patterns_list) if patterns_list else []
    chosen = [t for t in sampled if t in available_teeth]
    if not chosen:
        chosen = [random.choice(available_teeth)]
    return chosen

def process_case(obj_path, json_path, case_id, jaw, patterns_by_jaw):
    out_dir = OUT_ROOT / jaw / case_id
    (out_dir / "obj").mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    # original
    orig_img = out_dir / "images" / f"{case_id}_original.png"
    render_with_blender(obj_path, str(orig_img))

    # read tooth vertices
    tooth_verts = read_tooth_vertices_from_json(json_path)
    available = sorted(tooth_verts.keys())

    # manifest + original image record
    manifest = OUT_ROOT / f"manifest_{jaw}.csv"
    ensure_manifest(manifest)
    with open(manifest, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case_id","variant_id","removed_teeth","obj_path","image_path","jaw","seed","mode"
        ])
        w.writerow({
            "case_id": case_id, "variant_id": 0, "removed_teeth": "",
            "obj_path": obj_path, "image_path": str(orig_img),
            "jaw": jaw, "seed": RNG_SEED, "mode": "original"
        })

    # variants
    n_variants = random.randint(AUG_MIN_VARIANTS, AUG_MAX_VARIANTS)
    print(f"[Case] {case_id} ({jaw}) -> variants: {n_variants}")
    for vid in range(1, n_variants+1):
        chosen = choose_teeth(available, patterns_by_jaw)
        tag = "_".join(chosen) if chosen else "none"
        prefix = 'missing' if FILL_MODE == 'delete' else 'white'
        suffix = f"{prefix}_v{vid}_{tag}"
        out_obj = out_dir / "obj" / f"{case_id}_{suffix}.obj"
        if FILL_MODE == 'delete':
            ok = modify_obj_by_labels(obj_path, out_obj, chosen, tooth_verts)
        elif FILL_MODE == 'delete_white':
            ok = delete_and_whitefill_obj(obj_path, out_obj, chosen, tooth_verts)
        else:  # 'white'
            ok = build_whitefill_obj(obj_path, out_obj, chosen, tooth_verts)
        if not ok:
            print(f"[Warn] {case_id} v{vid}: processing failed"); continue
        out_img = out_dir / "images" / f"{case_id}_{suffix}.png"
        render_with_blender(out_obj, str(out_img))
        with open(manifest, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "case_id","variant_id","removed_teeth","obj_path","image_path","jaw","seed","mode"
            ])
            w.writerow({
                "case_id": case_id, "variant_id": vid,
                "removed_teeth": ",".join(chosen),
                "obj_path": str(out_obj), "image_path": str(out_img),
                "jaw": jaw, "seed": RNG_SEED, "mode": "delete_white" if FILL_MODE=='delete_white' else ("delete" if FILL_MODE=='delete' else "white")
            })

def run_batch():
    random.seed(RNG_SEED); np.random.seed(RNG_SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    patterns = load_test_missing_patterns(TEST_LABELS_CSV)

    base = Path(BASE_DIR)
    processed = 0
    for jaw in ["upper", "lower"]:
        jaw_dir = base / jaw
        if not jaw_dir.exists():
            print("[Skip] missing folder:", jaw_dir); continue
        for case_dir in sorted(jaw_dir.iterdir()):
            if not case_dir.is_dir(): 
                continue
            case_id = case_dir.name
            obj = case_dir / f"{case_id}_{jaw}.obj"
            jsn = case_dir / f"{case_id}_{jaw}.json"
            if not obj.exists() or not jsn.exists():
                print(f"[Skip] {case_id}: OBJ/JSON missing"); continue
            process_case(str(obj), str(jsn), case_id, jaw, patterns[jaw])
            processed += 1
            if BATCH_LIMIT and processed >= BATCH_LIMIT:
                print(f"[Batch] Reached limit={BATCH_LIMIT}. Stop."); print(f"[Done] {processed}"); return
    print(f"[Done] Processed cases: {processed}")

if __name__ == "__main__":
    run_batch()