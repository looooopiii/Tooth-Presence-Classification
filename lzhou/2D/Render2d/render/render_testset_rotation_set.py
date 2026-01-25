import bpy
import addon_utils
from pathlib import Path
import mathutils
import math
from mathutils import Vector, Matrix
import argparse
import sys
import os
import subprocess

# Try to use numpy for PCA-based upright orientation (falls back if missing)
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


def _np_eigh_sorted(cov):
    """Return eigenvalues/eigenvectors sorted ascending by eigenvalues."""
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)
    return w[idx], v[:, idx]


# =========== CONFIG ===========
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
TEST_FIRST_ONLY = False

# ==============================
# Rotation Set
# (X_deg, Y_deg, Z_deg)
# ==============================
ROTATION_SET = (
    [(0, 0, z) for z in [0, 45, 90, 135, 180, 225, 270, 315]] +
    [(90, 0, z) for z in [0, 90, 180, 270]] +
    [(-90, 0, z) for z in [0, 90, 180, 270]] +
    [(0, 90, z) for z in [0, 90, 180, 270]] +
    [(0, -90, z) for z in [0, 90, 180, 270]]
)

# ==== Auto-pick free GPUs ====
def get_free_gpus(threshold_mb=1000, max_gpus=1):
    try:
        res = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True
        )
        free = []
        for line in res.stdout.strip().splitlines():
            if not line.strip():
                continue
            gpu_id, mem = [x.strip() for x in line.split(',')]
            mem = int(mem)
            if mem < threshold_mb:
                free.append(int(gpu_id))
        free = free[:max_gpus]
        return free
    except Exception as e:
        print(f"[GPU] nvidia-smi query failed: {e}")
        return []


def set_cuda_visible_devices_if_needed(max_gpus=1):
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"].strip() != "":
        print(f"[GPU] CUDA_VISIBLE_DEVICES already set: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return
    free = get_free_gpus(threshold_mb=1000, max_gpus=max_gpus)
    if free:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in free)
        print(f"[GPU] Auto-selected free GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print("[GPU] No free GPU found (or nvidia-smi unavailable). Using default device.")


def parse_args():
    if '--' in sys.argv:
        i = sys.argv.index('--')
        script_argv = sys.argv[i+1:]
    else:
        script_argv = []

    parser = argparse.ArgumentParser(description="Render PNGs from PLY/STL with ROTATION_SET.", add_help=False)
    parser.add_argument("--in_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--in_file", type=str, default=None)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--ext", type=str, default="ply", choices=["ply", "stl"], help="Input file extension to render.")
    parser.add_argument("--max_gpus", type=int, default=1)
    args, _ = parser.parse_known_args(script_argv)

    if args.in_dir is None or args.out_dir is None:
        full_args, _ = parser.parse_known_args(sys.argv[1:])
        if args.in_dir is None:
            args.in_dir = full_args.in_dir
        if args.out_dir is None:
            args.out_dir = full_args.out_dir
        if not args.all:
            args.all = full_args.all

    if args.in_dir is None:
        args.in_dir = os.environ.get("IN_DIR")
    if args.out_dir is None:
        args.out_dir = os.environ.get("OUT_DIR")

    if args.in_dir is None:
        args.in_dir = str(DEFAULT_BASE_DIR)
    if args.out_dir is None:
        args.out_dir = str(DEFAULT_OUTPUT_DIR)

    base_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    render_all = args.all or (not TEST_FIRST_ONLY)

    try:
        print(f"[Debug] sys.argv = {sys.argv}")
    except Exception:
        pass
    print(f"[Args] in_dir={base_dir} | out_dir={out_dir} | render_all={render_all} | ext={args.ext}")

    return args, base_dir, out_dir, render_all


# enable importers
addon_utils.enable("io_mesh_ply")
addon_utils.enable("io_mesh_stl")


# =========== UNIFIED RENDER SETTINGS ===========
def setup_render_settings():
    """general render settings - consistent with other scripts"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        if hasattr(prefs, "compute_device_type"):
            prefs.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for dev in prefs.devices:
            if getattr(dev, "type", "") == 'CUDA':
                dev.use = True
            else:
                dev.use = False
    except Exception:
        pass

    scene.cycles.device = 'GPU'
    scene.cycles.samples = 512  # unified high-quality sampling
    scene.cycles.use_adaptive_sampling = True

    try:
        scene.view_layers["ViewLayer"].cycles.use_denoising = True
    except Exception:
        pass

    # unified resolution
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 2048
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = False

    # unified black background
    if scene.world is None:
        scene.world = bpy.data.worlds.new("SceneWorld")
    scene.world.use_nodes = True
    nt = scene.world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0, 0, 0, 1)
    wout = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], wout.inputs['Surface'])

    # unified color mapping
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'
    scene.view_settings.exposure = 1.5
    scene.view_settings.gamma = 1.0

    return scene


# =========== UNIFIED LIGHTING ===========
def setup_lighting(center, target):
    """unified lighting settings"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

    def add_tracked_light(kind, location, energy, size=None, spot_deg=None):
        bpy.ops.object.light_add(type=kind, location=location)
        L = bpy.context.object
        L.data.energy = energy
        if kind == 'AREA' and size is not None:
            L.data.size = size
            L.data.shape = 'SQUARE'
        if kind == 'SPOT' and spot_deg is not None:
            L.data.spot_size = math.radians(spot_deg)
            L.data.shadow_soft_size = 0.0
        c = L.constraints.new(type='TRACK_TO')
        c.target = target
        c.track_axis = 'TRACK_NEGATIVE_Z'
        c.up_axis = 'UP_Y'
        return L

    # unified lighting settings
    add_tracked_light('SPOT', (center.x, center.y - 150, center.z + 220), energy=250000, spot_deg=55)
    add_tracked_light('AREA', (center.x - 260, center.y - 200, center.z + 190), energy=85000, size=420)
    add_tracked_light('AREA', (center.x + 260, center.y + 200, center.z + 190), energy=85000, size=420)
    add_tracked_light('SPOT', (center.x - 320, center.y, center.z + 100), energy=180000, spot_deg=60)
    add_tracked_light('SPOT', (center.x + 320, center.y, center.z + 100), energy=180000, spot_deg=60)
    add_tracked_light('SPOT', (center.x, center.y - 320, center.z + 80), energy=120000, spot_deg=65)

    return target


# =========== MATERIAL ===========
def create_tooth_material():
    """unified tooth material - with AO for detail enhancement"""
    mat = bpy.data.materials.new(name="ToothMaterial_Unified")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Ambient Occlusion for detail enhancement
    ao_node = nodes.new('ShaderNodeAmbientOcclusion')
    ao_node.inputs['Distance'].default_value = 1.0

    mix_rgb = nodes.new('ShaderNodeMixRGB')
    mix_rgb.blend_type = 'MULTIPLY'
    mix_rgb.inputs['Fac'].default_value = 1.0

    # Principled BSDF
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs['Specular'].default_value = 0.5
    principled.inputs['Roughness'].default_value = 0.1
    principled.inputs['Sheen'].default_value = 0.05

    # Connect AO
    links.new(ao_node.outputs['Color'], mix_rgb.inputs[1])
    links.new(principled.inputs['Base Color'], mix_rgb.inputs[2])
    links.new(mix_rgb.outputs['Color'], principled.inputs['Base Color'])

    outp = nodes.new('ShaderNodeOutputMaterial')
    links.new(principled.outputs['BSDF'], outp.inputs['Surface'])

    return mat


# =========== ORIENTATION HELPERS ===========
def orient_top_view(model):
    dims = model.dimensions
    xyz = [dims.x, dims.y, dims.z]
    min_idx = xyz.index(min(xyz))
    if min_idx == 0:
        model.rotation_euler.rotate_axis('Y', math.radians(90))
    elif min_idx == 1:
        model.rotation_euler.rotate_axis('X', math.radians(-90))
    bpy.context.view_layer.update()

    up_count, down_count = 0, 0
    model.data.calc_normals_split()
    mw = model.matrix_world

    polys = model.data.polygons
    step = int(max(1, len(polys) // 2000))
    for i in range(0, len(polys), step):
        p = polys[i]
        n = mw.to_3x3() @ p.normal
        if n.z >= 0:
            up_count += 1
        else:
            down_count += 1

    if down_count > up_count:
        model.rotation_euler.rotate_axis('X', math.radians(180))
        bpy.context.view_layer.update()


def upright_with_pca(model):
    if not HAS_NUMPY:
        return

    verts = model.data.vertices
    if len(verts) < 8:
        return

    P = np.array([list(v.co) for v in verts], dtype=np.float64)
    mean = P.mean(axis=0)
    X = P - mean
    cov = np.cov(X.T)

    _, eigvecs_sorted = _np_eigh_sorted(cov)
    v_min = eigvecs_sorted[:, 0]
    v_mid = eigvecs_sorted[:, 1]
    v_max = eigvecs_sorted[:, 2]

    axes = [v_mid, v_max]
    lens = []
    for a in axes:
        proj = X @ a
        lens.append(float(proj.max() - proj.min()))
    if lens[1] >= lens[0]:
        x_axis = v_max
        y_axis = v_mid
    else:
        x_axis = v_mid
        y_axis = v_max

    z_axis = v_min

    R = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(R) < 0:
        y_axis = -y_axis
        R = np.column_stack((x_axis, y_axis, z_axis))

    Rm = Matrix(((R[0, 0], R[0, 1], R[0, 2]),
                 (R[1, 0], R[1, 1], R[1, 2]),
                 (R[2, 0], R[2, 1], R[2, 2])))
    model.rotation_euler = Rm.to_euler()
    bpy.context.view_layer.update()


def auto_flip_for_top_view(model, jaw_type: str):
    bpy.context.view_layer.update()
    dims = model.dimensions
    if dims.y > dims.x:
        model.rotation_euler.rotate_axis('Z', math.radians(90))
        bpy.context.view_layer.update()

    mw = model.matrix_world
    zs = [(mw @ Vector(v.co)).z for v in model.data.vertices]
    if not zs:
        return
    if HAS_NUMPY:
        z_thr = np.percentile(zs, 85)
    else:
        z_thr = (max(zs) - 0.15 * (max(zs) - min(zs)))

    top_normals_z = []
    poly = model.data.polygons
    for p in poly:
        vs = [(mw @ model.data.vertices[i].co).z for i in p.vertices]
        if max(vs) >= z_thr:
            nz = (mw.to_3x3() @ p.normal).z
            top_normals_z.append(nz)

    mean_nz = sum(top_normals_z) / len(top_normals_z) if top_normals_z else 0.0

    if mean_nz < 0:
        model.rotation_euler.rotate_axis('X', math.radians(180))
        bpy.context.view_layer.update()

    if jaw_type == 'lower':
        top_normals_z = []
        for p in model.data.polygons:
            vs = [(mw @ model.data.vertices[i].co).z for i in p.vertices]
            if max(vs) >= z_thr:
                nz = (model.matrix_world.to_3x3() @ p.normal).z
                top_normals_z.append(nz)
        mean_nz2 = sum(top_normals_z) / len(top_normals_z) if top_normals_z else 0.0
        if mean_nz2 < 0:
            model.rotation_euler.rotate_axis('X', math.radians(180))
            bpy.context.view_layer.update()


def prepare_model(model, jaw_type: str):
    """general model preparation - consistent with other scripts"""
    # Material
    tooth_mat = create_tooth_material()
    if model.data.materials:
        model.data.materials[0] = tooth_mat
    else:
        model.data.materials.append(tooth_mat)

    # Smooth + normals
    bpy.ops.object.select_all(action='DESELECT')
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    bpy.ops.object.shade_smooth()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    model.data.use_auto_smooth = True
    model.data.auto_smooth_angle = math.radians(40)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.faces_shade_smooth()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Reset rotation
    model.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    # Orientation pipeline
    orient_top_view(model)
    bpy.context.view_layer.update()

    upright_with_pca(model)
    bpy.context.view_layer.update()

    auto_flip_for_top_view(model, jaw_type)
    bpy.context.view_layer.update()

    # Center to origin
    bbox = [model.matrix_world @ mathutils.Vector(corner) for corner in model.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8.0
    model.location -= center

    # Scale to uniform size
    bpy.context.view_layer.update()
    dims = model.dimensions
    xy_max = max(dims.x, dims.y)
    if xy_max > 0:
        s = 150.0 / xy_max
        model.scale = (model.scale.x * s, model.scale.y * s, model.scale.z * s)
    bpy.context.view_layer.update()


# =========== UNIFIED CAMERA ===========
def setup_camera(model, margin=1.1):
    """unified orthographic camera setup"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in {'CAMERA', 'EMPTY'}:
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

    bbox = [model.matrix_world @ mathutils.Vector(corner) for corner in model.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8.0
    dims = model.dimensions

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(center.x, center.y, center.z))
    target = bpy.context.object

    cam_height = max(200.0, dims.z * 6.0 + 120.0)
    bpy.ops.object.camera_add(location=(center.x, center.y, center.z + cam_height))
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = max(dims.x, dims.y) * margin
    cam.data.clip_start = 0.1
    cam.data.clip_end = 10000.0

    con = cam.constraints.new(type='TRACK_TO')
    con.target = target
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)

    return cam, center, target


# =========== ROTATION APPLY ===========
def apply_rotation_xyz(model, base_rot_euler, x_deg, y_deg, z_deg):
    """
    Apply rotations in XYZ order on top of base_rot_euler.
    """
    model.rotation_euler = base_rot_euler.copy()
    bpy.context.view_layer.update()

    if x_deg != 0:
        model.rotation_euler.rotate_axis('X', math.radians(x_deg))
    if y_deg != 0:
        model.rotation_euler.rotate_axis('Y', math.radians(y_deg))
    if z_deg != 0:
        model.rotation_euler.rotate_axis('Z', math.radians(z_deg))
    bpy.context.view_layer.update()


def clean_mesh_objects():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in list(bpy.context.scene.objects):
        if obj.type == 'MESH':
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.ops.object.delete()


def import_model(filepath: Path, ext: str):
    if ext == "ply":
        bpy.ops.import_mesh.ply(filepath=str(filepath))
    elif ext == "stl":
        bpy.ops.import_mesh.stl(filepath=str(filepath))
    else:
        raise ValueError(f"Unsupported ext: {ext}")

    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        return None
    return meshes[0]


def infer_jaw_type_from_name(name: str):
    n = name.lower()
    if "upper" in n:
        return "upper"
    if "lower" in n:
        return "lower"
    return "unknown"


# =========== MAIN ===========
def main():
    args, base_dir, out_dir, render_all = parse_args()
    set_cuda_visible_devices_if_needed(max_gpus=args.max_gpus)

    scene = setup_render_settings()

    # collect files
    if '--' in sys.argv:
        i = sys.argv.index('--')
        script_argv = sys.argv[i+1:]
    else:
        script_argv = []

    tmp_parser = argparse.ArgumentParser(add_help=False)
    tmp_parser.add_argument("--in_file", type=str, default=None)
    tmp_parser.add_argument("--max", type=int, default=None)
    tmp_args, _ = tmp_parser.parse_known_args(script_argv)

    ext = args.ext.lower()

    if tmp_args.in_file:
        fpath = Path(tmp_args.in_file).expanduser().resolve()
        if not fpath.exists():
            print(f"[Error] --in_file not found: {fpath}")
            return
        files = [fpath]
        print(f"[Mode] Single-file mode. Rendering 1 file: {fpath}")
    else:
        in_files = sorted(list(base_dir.rglob(f"*.{ext}")))
        if not in_files:
            print(f"No *.{ext} files found in {base_dir}")
            return

        if render_all or (tmp_args.max is None):
            files = in_files if render_all else in_files[:1]
        else:
            files = in_files[:max(1, tmp_args.max)]
        print(f"Found {len(in_files)} {ext.upper()} files. Rendering {len(files)} file(s)...")

    sys.stdout.flush()

    rendered = 0
    for f in files:
        try:
            clean_mesh_objects()

            model = import_model(f, ext=ext)
            if model is None:
                print(f"Failed to import: {f.name}")
                continue

            print(f"Processing: {f.name}")

            jaw_type = infer_jaw_type_from_name(f.name)
            print(f"Jaw type: {jaw_type}")

            prepare_model(model, jaw_type)
            bpy.context.view_layer.update()

            base_rot = model.rotation_euler.copy()
            case_out_dir = out_dir / f.stem
            case_out_dir.mkdir(parents=True, exist_ok=True)

            # render all rotations in ROTATION_SET
            for (rx, ry, rz) in ROTATION_SET:
                apply_rotation_xyz(model, base_rot, rx, ry, rz)

                cam, center, target = setup_camera(model, margin=1.1)
                setup_lighting(center, target)

                out_name = f"{f.stem}_top_rx{rx}_ry{ry}_rz{rz}.png"
                scene.render.filepath = str(case_out_dir / out_name)

                bpy.ops.render.render(write_still=True)
                print(f"Rendered: {out_name}")
                rendered += 1

        except Exception as e:
            print(f"Error rendering {f.name}: {e}")

    print(f"Done. Rendered {rendered} image(s) to {out_dir}")


if __name__ == "__main__":
    main()
