import json
import csv
import random
import numpy as np
from pathlib import Path
import shutil
import subprocess
import os
import sys

def _get_blender_bin():
    """Return blender binary path, preferring env var BLENDER_BIN (with ~ expansion)."""
    env = os.environ.get("BLENDER_BIN")
    if env:
        return os.path.expanduser(env)
    return "blender"

# ----------------- Config (adjust freely) -----------------
# When running in batch (default), we traverse both lower/upper under BASE_DIR.
# You can still override via env: BATCH_BASE_DIR, BATCH_LIMIT.
BASE_DIR = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split"

# (Single-case debug variables are no longer needed for default runs; if you want to run a single case,
#  set BATCH_LIMIT=1 or re-enable the run_test() path below.)
TEST_OUTPUT_DIR = Path("/home/user/lzhou/week8/data_augmentation/Aug_data")

# How many augmented variants to create for this single case
AUG_NUM_VARIANTS = 3
# How many teeth to remove per variant (meeting suggests start with 1)
TEETH_TO_REMOVE_PER_VARIANT = 1
# Fixed seed for reproducibility
RNG_SEED = 42
# Fill mode for augmentation: "delete" (remove vertices) or "black" (keep geometry, paint selected teeth black)
FILL_MODE = "delete"  # choose: "delete" or "black"
# ----------------------------------------------------------


class TeethDataAugmentorTest:
    def __init__(self):
        random.seed(RNG_SEED)
        np.random.seed(RNG_SEED)
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (TEST_OUTPUT_DIR / "obj").mkdir(parents=True, exist_ok=True)
        (TEST_OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

    def get_tooth_labels_from_json(self, json_path):
        """
        Read per-vertex labels from JSON and return: { 'tooth_<id>': [vertex_idx(1-based), ...], ... }
        Assumes label 0 = background/gingiva; others are tooth IDs.
        """
        if not os.path.exists(json_path):
            print(f"[Error] JSON file does not exist: {json_path}")
            return {}

        with open(json_path, 'r') as f:
            data = json.load(f)

        labels = data.get("labels", [])
        if not labels:
            print("[Warn] 'labels' array missing or empty in JSON.")
            return {}

        tooth_to_vertices = {}
        for i, label in enumerate(labels):
            if label != 0:
                tooth_name = f"tooth_{label}"
                tooth_to_vertices.setdefault(tooth_name, []).append(i + 1)  # OBJ is 1-based

        print(f"[Info] Teeth detected from JSON: {len(tooth_to_vertices)}")
        for name, verts in list(tooth_to_vertices.items())[:8]:
            print(f"  - {name}: {len(verts)} vertices")
        if len(tooth_to_vertices) > 8:
            print(f"  ... (+{len(tooth_to_vertices)-8} more)")
        return tooth_to_vertices

    def choose_teeth_to_remove(self, all_tooth_names, k=1):
        """Choose k teeth to remove (default 1 for meeting's initial protocol)."""
        if not all_tooth_names:
            return []
        k = max(1, min(k, len(all_tooth_names)))
        chosen = random.sample(all_tooth_names, k)
        print(f"[Info] Removing teeth: {chosen}")
        return chosen

    def modify_obj_by_labels(self, input_path, output_path, teeth_to_remove_names, tooth_labels):
        """
        Remove vertices belonging to selected teeth and drop faces touching removed vertices.
        Re-map surviving vertex indices to keep OBJ valid.
        """
        print(f"[Step] Modify OBJ: {input_path} -> {output_path}")
        if not teeth_to_remove_names:
            shutil.copyfile(input_path, output_path)
            print("[Info] No teeth selected. Copied original.")
            return True

        vertices_to_remove = set()
        for tooth_name in teeth_to_remove_names:
            verts = tooth_labels.get(tooth_name, [])
            vertices_to_remove.update(verts)

        if not vertices_to_remove:
            print("[Warn] No vertices found for selected teeth. Copy original.")
            shutil.copyfile(input_path, output_path)
            return True

        with open(input_path, 'r') as f:
            lines = f.readlines()

        modified = []
        vertex_mapping = {}  # old_idx -> new_idx
        new_idx = 1
        old_idx = 0

        for line in lines:
            if line.startswith('v '):
                old_idx += 1
                if old_idx not in vertices_to_remove:
                    modified.append(line)
                    vertex_mapping[old_idx] = new_idx
                    new_idx += 1
                else:
                    # keep a comment line for traceability (optional)
                    modified.append(f"# removed {line}")
            elif line.startswith('f '):
                parts = line.strip().split()
                new_parts = ['f']
                valid = True
                for p in parts[1:]:
                    # p can be v, v/t, v//n, v/t/n ; we only remap the first (v)
                    vref = p.split('/')[0]
                    try:
                        vold = int(vref)
                        if vold in vertex_mapping:
                            vnew = vertex_mapping[vold]
                            new_parts.append(p.replace(vref, str(vnew), 1))
                        else:
                            valid = False
                            break
                    except ValueError:
                        new_parts.append(p)
                if valid and len(new_parts) > 3:  # face must have at least 3 vertices
                    modified.append(' '.join(new_parts) + '\n')
                # else: drop degenerate/invalid face
            else:
                modified.append(line)

        with open(output_path, 'w') as f:
            f.writelines(modified)

        print(f"[Info] Removed vertices: {len(vertices_to_remove)}; Remaining vertices: {new_idx-1}")
        return True

    def build_blackfill_obj(self, input_path, output_path, teeth_to_black_names, tooth_labels):
        """
        Keep geometry intact and assign a BLACK material to faces that belong to the selected teeth.
        - Writes an OBJ that references a companion MTL with two materials: BLACK and DEFAULT.
        - A face is considered belonging to a selected tooth if ALL its vertex indices are in that tooth's vertex set.
        """
        print(f"[Step] Black-fill OBJ (no deletion): {input_path} -> {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mtl_path = output_path.with_suffix(".mtl")
        mtl_name = mtl_path.name

        # Collect vertex indices for selected teeth
        selected_vertices = set()
        for tooth in teeth_to_black_names:
            selected_vertices.update(tooth_labels.get(tooth, []))
        if not selected_vertices:
            print("[Warn] No vertices for selected teeth; copying original as-is.")
            shutil.copyfile(input_path, output_path)
            return True

        with open(input_path, 'r') as f:
            lines = f.readlines()

        out_lines = []
        current_mtl = None
        wrote_mtllib = False

        def set_mtl(name):
            nonlocal current_mtl
            if current_mtl != name:
                out_lines.append(f"usemtl {name}\n")
                current_mtl = name

        for i, line in enumerate(lines):
            if line.startswith('mtllib '):
                # We'll replace with our own
                continue
            if not wrote_mtllib and line.startswith('v '):
                # Insert mtllib header before the first vertex block
                out_lines.append(f"mtllib {mtl_name}\n")
                wrote_mtllib = True

            if line.startswith('f '):
                parts = line.strip().split()
                verts_ok = True
                face_vids = []
                for p in parts[1:]:
                    vref = p.split('/')[0]
                    try:
                        face_vids.append(int(vref))
                    except ValueError:
                        verts_ok = False
                        break
                if verts_ok and face_vids:
                    if all((vid in selected_vertices) for vid in face_vids):
                        set_mtl("BLACK")
                    else:
                        set_mtl("DEFAULT")
                out_lines.append(line if line.endswith('\n') else line + '\n')
            else:
                out_lines.append(line if line.endswith('\n') else line + '\n')

        # Write OBJ
        with open(output_path, 'w') as f:
            f.writelines(out_lines)

        # Write MTL with two materials
        with open(mtl_path, 'w') as f:
            f.write("# Auto-generated MTL for black-fill augmentation\n")
            f.write("newmtl BLACK\nKa 0.0 0.0 0.0\nKd 0.0 0.0 0.0\nKs 0.0 0.0 0.0\nNs 1.0\nd 1.0\nillum 1\n")
            f.write("\nnewmtl DEFAULT\nKa 0.0 0.0 0.0\nKd 0.90 0.88 0.85\nKs 0.05 0.05 0.05\nNs 50.0\nd 1.0\nillum 2\n")

        print(f"[Info] Wrote MTL: {mtl_path}")
        return True

    def create_blender_script(self, obj_path, output_image_path):
        """Blender python that loads OBJ, centers it, sets black background, and renders."""
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

    # Only apply a uniform tooth material when geometry was modified (delete mode).
    APPLY_TOOTH_MATERIAL = {str(FILL_MODE == "delete")}
    if APPLY_TOOTH_MATERIAL:
        # --- Tooth material: diffuse+glossy, slightly gray to avoid clipping ---
        mat = bpy.data.materials.new(name="ToothMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
        diffuse.inputs['Color'].default_value = (0.90, 0.88, 0.85, 1.0)  # slightly gray

        glossy = nodes.new(type='ShaderNodeBsdfGlossy')
        glossy.inputs['Roughness'].default_value = 0.25

        mix_shader = nodes.new(type='ShaderNodeMixShader')
        mix_shader.inputs['Fac'].default_value = 0.20  # 20% glossy

        output = nodes.new(type='ShaderNodeOutputMaterial')

        links.new(diffuse.outputs['BSDF'], mix_shader.inputs[1])
        links.new(glossy.outputs['BSDF'], mix_shader.inputs[2])
        links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

        # Assign and smooth shade
        for o in meshes:
            if o.type == 'MESH':
                if o.data.materials:
                    o.data.materials[0] = mat
                else:
                    o.data.materials.append(mat)
                bpy.ops.object.select_all(action='DESELECT')
                o.select_set(True)
                bpy.context.view_layer.objects.active = o
                bpy.ops.object.shade_smooth()

    # ---------------- Camera: strict TOP occlusal ----------------
    # Auto-detect jaw from filename; default to lower if unknown
    name_l = os.path.basename(obj_path).lower()
    jaw = "lower" if "_lower" in name_l else ("upper" if "_upper" in name_l else "lower")
    cam_dist = max_dim * 2.0

    cam_z = cam_dist if jaw == "lower" else -cam_dist
    bpy.ops.object.camera_add(location=(0.0, 0.0, cam_z))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = (-cam.location).normalized().to_track_quat('-Z','Y')

    # Orthographic (no perspective) to match textbook TOP view
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = max_dim * 1.6  # tighter framing for clearer details

    # ---------------- Lighting tuned for TOP view ----------------
    bpy.ops.object.light_add(type='SUN', location=(0, 0, cam_z))
    key = bpy.context.object
    key.data.energy = 3.0
    key.data.angle = 0.6  # softer, wider penumbra

    # ---------------- Render settings & world ----------------
    scene = bpy.context.scene
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 2048
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32

    # Black background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (0, 0, 0, 1)
        bg.inputs[1].default_value = 1.0

    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Medium High Contrast'
    scene.view_settings.exposure = -0.6

    try:
        bpy.ops.render.render(write_still=True)
    except Exception as e:
        print("[Blender][Error] Render failed:", e)
        return False

    print(f"[Blender] Done: {{output_path}}")
    return True

ok = render_model("{abs_obj}", "{abs_out}")
if not ok:
    import sys; sys.exit(2)
'''

    def _ensure_manifest(self, manifest_path):
        if not Path(manifest_path).exists():
            Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['case_id', 'variant_id', 'removed_teeth', 'obj_path', 'image_path', 'jaw', 'seed', 'mode'])
                writer.writeheader()

    def render_with_blender(self, obj_path, output_image_path):
        Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
        script_path = TEST_OUTPUT_DIR / "render_script.py"
        with open(script_path, 'w') as f:
            f.write(self.create_blender_script(obj_path, output_image_path))
        blender_bin = _get_blender_bin()
        cmd = [blender_bin, "--background", "--python", str(script_path)]
        print(f"[Info] Using Blender binary: {blender_bin}")
        if blender_bin != "blender" and not (os.path.isfile(blender_bin) and os.access(blender_bin, os.X_OK)):
            print(f"[Warn] BLENDER_BIN points to a non-executable or missing file: {blender_bin}")
        print(f"[Run] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] Rendered: {output_image_path}")
            return True
        else:
            print("[Err] Blender returned non-zero exit code:", result.returncode)
            print("----- STDOUT -----")
            print(result.stdout)
            print("----- STDERR -----")
            print(result.stderr)
            return False

    def run_test(self):
        print("[Start] Batch demo via BASE_DIR (process both upper/lower). Use BATCH_LIMIT to cap the count.")
        self.run_batch(BASE_DIR, limit=None)

    def process_case(self, obj_path, json_path, case_id, base_out_dir):
        print(f"\n[Case] {case_id}")
        jaw = "lower" if "_lower" in obj_path.lower() else ("upper" if "_upper" in obj_path.lower() else "lower")
        out_dir = Path(base_out_dir) / jaw / case_id
        (out_dir / "obj").mkdir(parents=True, exist_ok=True)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)

        # 1) Render original
        original_img_path = out_dir / "images" / f"{case_id}_original.png"
        self.render_with_blender(obj_path, str(original_img_path))

        # 2) Load labels
        tooth_labels = self.get_tooth_labels_from_json(json_path)
        all_teeth = list(tooth_labels.keys())
        if not all_teeth:
            print("[Abort] No tooth labels found.")
            return

        # Determine jaw
        # (jaw already determined above)

        # Ensure jaw-specific manifest
        manifest_path = Path(base_out_dir) / f"manifest_{jaw}.csv"
        self._ensure_manifest(manifest_path)

        # Append original (baseline)
        with open(manifest_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['case_id', 'variant_id', 'removed_teeth', 'obj_path', 'image_path', 'jaw', 'seed', 'mode'])
            writer.writerow({
                'case_id': case_id,
                'variant_id': 0,
                'removed_teeth': "",
                'obj_path': obj_path,
                'image_path': str(original_img_path),
                'jaw': jaw,
                'seed': RNG_SEED,
                'mode': 'original'
            })

        # 3) Create augmented variants
        print(f"  -> Generating {AUG_NUM_VARIANTS} variants (k={TEETH_TO_REMOVE_PER_VARIANT}, mode={FILL_MODE})")
        for vid in range(1, AUG_NUM_VARIANTS + 1):
            chosen = self.choose_teeth_to_remove(all_teeth, k=TEETH_TO_REMOVE_PER_VARIANT)
            teeth_str = "_".join(chosen)
            mode_tag = "blackfill" if FILL_MODE == "black" else "missing"
            filename_suffix = f"{mode_tag}_v{vid}_{teeth_str}"
            out_obj = out_dir / "obj" / f"{case_id}_{filename_suffix}.obj"
            if FILL_MODE == "black":
                ok = self.build_blackfill_obj(obj_path, out_obj, chosen, tooth_labels)
            else:
                ok = self.modify_obj_by_labels(obj_path, out_obj, chosen, tooth_labels)
            if not ok:
                print(f"[Warn] Variant {vid}: OBJ modify failed, skip render.")
                continue

            # 4) Render augmented
            out_img = out_dir / "images" / f"{case_id}_{filename_suffix}.png"
            self.render_with_blender(out_obj, str(out_img))

            # Append manifest entry
            with open(manifest_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['case_id', 'variant_id', 'removed_teeth', 'obj_path', 'image_path', 'jaw', 'seed', 'mode'])
                writer.writerow({
                    'case_id': case_id,
                    'variant_id': vid,
                    'removed_teeth': ",".join(chosen),
                    'obj_path': str(out_obj),
                    'image_path': str(out_img),
                    'jaw': jaw,
                    'seed': RNG_SEED,
                    'mode': FILL_MODE
                })

    def run_batch(self, base_dir, limit=None):
        """
        Traverse dataset structure:
          base_dir/
            upper/<CASE>/<CASE>_upper.obj
            upper/<CASE>/<CASE>_upper.json
            lower/<CASE>/<CASE>_lower.obj
            lower/<CASE>/<CASE>_lower.json
        For each found pair, run process_case. Optional 'limit' to process only N cases.
        Note: When run_test() is called, this method uses BASE_DIR by default to process both jaws.
        """
        base_dir = Path(base_dir)
        processed = 0
        for jaw in ["upper", "lower"]:
            jaw_dir = base_dir / jaw
            if not jaw_dir.exists():
                print(f"[Warn] Missing folder: {jaw_dir}")
                continue
            for case_dir in sorted(jaw_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                case_id = case_dir.name
                obj = case_dir / f"{case_id}_{jaw}.obj"
                jsn = case_dir / f"{case_id}_{jaw}.json"
                if not obj.exists() or not jsn.exists():
                    print(f"[Skip] Missing files for {case_id}: {obj.name if obj.exists() else 'OBJ?'} / {jsn.name if jsn.exists() else 'JSON?'}")
                    continue
                self.process_case(str(obj), str(jsn), case_id, TEST_OUTPUT_DIR)
                processed += 1
                if limit and processed >= int(limit):
                    print(f"[Info] Reached limit={limit}. Stop.")
                    return
        print(f"[Done] Batch finished. Cases processed: {processed}")


def main():
    tester = TeethDataAugmentorTest()
    base_dir = os.environ.get("BATCH_BASE_DIR", "").strip()
    limit = os.environ.get("BATCH_LIMIT", "").strip()
    if base_dir:
        print(f"[Batch] Start. BASE={base_dir}  LIMIT={limit or 'None'}")
        tester.run_batch(base_dir, limit=int(limit) if limit.isdigit() else None)
    else:
        tester.run_test()


if __name__ == "__main__":
    main()