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

ROTATION_ANGLES_DEG = [0, 90, 180, 270]

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
            if int(mem) < threshold_mb:
                free.append(int(gpu_id))
            if len(free) >= max_gpus:
                break
        print(f"[GPU] Free GPUs detected: {free}")
        return free if free else [0]
    except Exception as e:
        print(f"[GPU] Error detecting free GPUs: {e}")
        return [0]

if "CUDA_VISIBLE_DEVICES" not in os.environ or not os.environ["CUDA_VISIBLE_DEVICES"]:
    picked = get_free_gpus(threshold_mb=1000, max_gpus=1)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in picked)
    print(f"[GPU] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")


def parse_args():
    if '--' in sys.argv:
        i = sys.argv.index('--')
        script_argv = sys.argv[i+1:]
    else:
        script_argv = []

    parser = argparse.ArgumentParser(description="Render top-view PNGs from STL files.", add_help=False)
    parser.add_argument("--in_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--in_file", type=str, default=None)
    parser.add_argument("--max", type=int, default=None)
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
    print(f"[Args] in_dir={base_dir} | out_dir={out_dir} | render_all={render_all}")

    return base_dir, out_dir, render_all

# Enable STL importer
addon_utils.enable("io_mesh_stl")


# =========== UNIFIED RENDER SETTINGS ===========
def setup_render_settings():
    """general render settings - consistent with Render_test.py"""
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
    """unified lighting settings, consistent with Render_test.py"""
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


# =========== UNIFIED MATERIAL ===========
def create_tooth_material():
    """unified tooth material"""
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


# =========== UNIFIED MODEL PREP ===========
def prepare_model(model, jaw_type: str):
    """unified model preparation"""
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


# =========== MAIN ===========
def main():
    scene = setup_render_settings()

    base_dir, out_dir, render_all = parse_args()

    if '--' in sys.argv:
        i = sys.argv.index('--')
        script_argv = sys.argv[i+1:]
    else:
        script_argv = []
    tmp_parser = argparse.ArgumentParser(add_help=False)
    tmp_parser.add_argument("--in_file", type=str, default=None)
    tmp_parser.add_argument("--max", type=int, default=None)
    tmp_args, _ = tmp_parser.parse_known_args(script_argv)

    if tmp_args.in_file:
        fpath = Path(tmp_args.in_file).expanduser().resolve()
        if not fpath.exists():
            print(f"[Error] --in_file not found: {fpath}")
            return
        if not str(fpath).lower().endswith('.stl'):
            print(f"[Error] --in_file must be an STL: {fpath}")
            return
        files = [fpath]
        print(f"[Mode] Single-file mode. Rendering 1 file: {fpath}")
    else:
        stl_files = sorted(list(base_dir.rglob("*.stl")))
        if not stl_files:
            print(f"No STL files found in {base_dir}")
            return

        if render_all or (tmp_args.max is None):
            files = stl_files if render_all else stl_files[:1]
        else:
            files = stl_files[:max(1, tmp_args.max)]
        print(f"Found {len(stl_files)} STL files. Rendering {len(files)} file(s)...")

    sys.stdout.flush()

    rendered = 0
    for stl_path in files:
        try:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in list(bpy.context.scene.objects):
                if obj.type == 'MESH':
                    obj.select_set(True)
            if bpy.context.selected_objects:
                bpy.ops.object.delete()

            bpy.ops.import_mesh.stl(filepath=str(stl_path))
            meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
            if not meshes:
                print(f"Failed to import: {stl_path.name}")
                continue
            model = meshes[0]
            print(f"Processing: {stl_path.name}")

            name_low = stl_path.name.lower()
            if "upper" in name_low:
                jaw_type = "upper"
            elif "lower" in name_low:
                jaw_type = "lower"
            else:
                jaw_type = "unknown"
            print(f"Jaw type: {jaw_type}")

            prepare_model(model, jaw_type)
            bpy.context.view_layer.update()

            base_rot = model.rotation_euler.copy()

            for angle_deg in ROTATION_ANGLES_DEG:
                model.rotation_euler = base_rot.copy()
                bpy.context.view_layer.update()

                if angle_deg != 0:
                    model.rotation_euler.rotate_axis('Z', math.radians(angle_deg))
                    bpy.context.view_layer.update()

                cam, center, target = setup_camera(model, margin=1.1)
                setup_lighting(center, target)

                out_name = f"{stl_path.stem}_top_rot{angle_deg}.png"
                scene.render.filepath = str(out_dir / out_name)

                bpy.ops.render.render(write_still=True)
                print(f"Rendered: {out_name}")
                rendered += 1

        except Exception as e:
            print(f"Error rendering {stl_path.name}: {e}")

    print(f"Done. Rendered {rendered} image(s) to {out_dir}")


if __name__ == "__main__":
    main()