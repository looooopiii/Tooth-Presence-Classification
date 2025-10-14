import bpy
import addon_utils
from pathlib import Path
import mathutils
import math

# Try to use numpy for PCA-based upright orientation (falls back if missing)
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

# =========== CONFIG ===========
BASE_DIR = Path("/local/scratch/datasets/IntraOralScans/data")
OUTPUT_DIR = Path("/home/user/lzhou/week7/top_views")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_FIRST_ONLY = True  # set to False for full batch

# Enable PLY importer
addon_utils.enable("io_mesh_ply")

# === Modified for improved enamel detail and contrast ===
# =========== RENDER SETTINGS ===========
def setup_render_settings():
    """
    Cycles high-quality settings aimed at bright, crisp top views.
    - Orthographic camera will be set later.
    - World is pure black.
    - Filmic + High Contrast + higher exposure for punchy whites.
    """
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 512
    scene.cycles.use_adaptive_sampling = True
    try:
        scene.view_layers["ViewLayer"].cycles.use_denoising = True
    except Exception:
        pass

    scene.render.resolution_x = 2048
    scene.render.resolution_y = 2048
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = False

    # Pure black background via World nodes
    if scene.world is None:
        scene.world = bpy.data.worlds.new("SceneWorld")
    scene.world.use_nodes = True
    nt = scene.world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0, 0, 0, 1)
    wout = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], wout.inputs['Surface'])

    # Tone mapping for bright white enamel while keeping grooves visible
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'
    scene.view_settings.exposure = 1.5   # slightly lower for more realistic exposure
    scene.view_settings.gamma = 1.0
    return scene

# === Modified for improved enamel detail and contrast ===
# =========== LIGHTING ===========
def setup_lighting(center, target):
    """
    A bright, studio-like setup:
    - Large overhead AREA as key (soft, even white).
    - Two diagonal AREA fills to lift shadows.
    - Two shallow-angle SPOT rims to pop edges/cusps.
    All lights Track-To the model center for consistent shaping.
    """
    # Remove previous lights
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

    # Key: angled, warmer spotlight for more directional shadows
    add_tracked_light('SPOT', (center.x, center.y - 150, center.z + 220), energy=250000, spot_deg=55)

    # Two big diagonal fills (increase contrast by lowering energy further)
    add_tracked_light('AREA', (center.x - 260, center.y - 200, center.z + 190), energy=85000, size=420)
    add_tracked_light('AREA', (center.x + 260, center.y + 200, center.z + 190), energy=85000, size=420)

    # Two rim spots from sides to define cusp edges (increase energy for clearer edge definition)
    add_tracked_light('SPOT', (center.x - 320, center.y, center.z + 100), energy=180000, spot_deg=60)
    add_tracked_light('SPOT', (center.x + 320, center.y, center.z + 100), energy=180000, spot_deg=60)

    # Additional back rim spot for surface detail (increased energy)
    add_tracked_light('SPOT', (center.x, center.y - 320, center.z + 80), energy=120000, spot_deg=65)

# === Modified for improved enamel detail and contrast ===
# =========== MATERIAL ===========
def create_tooth_material():
    """
    Simpler bright enamel-like shader for high clarity:
    - Principled BSDF with high specular for porcelain-like sheen.
    - Slightly glossy, white, and neutral.
    Result: whiter teeth with crisp details.
    """
    mat = bpy.data.materials.new(name="ToothMaterial_CrispWhite")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Ambient Occlusion node to enhance crevices
    ao_node = nodes.new('ShaderNodeAmbientOcclusion')
    ao_node.inputs['Distance'].default_value = 1.0 
    mix_rgb = nodes.new('ShaderNodeMixRGB')
    mix_rgb.blend_type = 'MULTIPLY'
    mix_rgb.inputs['Fac'].default_value = 1.0

    # Principled base (whiter & glossier, but simpler)
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs['Specular'].default_value = 0.5
    principled.inputs['Roughness'].default_value = 0.1
    principled.inputs['Sheen'].default_value = 0.05

    # AO link: AO Color * Base Color -> Principled Base Color
    links.new(ao_node.outputs['Color'], mix_rgb.inputs[1])
    links.new(principled.inputs['Base Color'], mix_rgb.inputs[2])
    links.new(mix_rgb.outputs['Color'], principled.inputs['Base Color'])

    outp = nodes.new('ShaderNodeOutputMaterial')
    links.new(principled.outputs['BSDF'], outp.inputs['Surface'])

    return mat

# =========== ORIENTATION HELPERS ===========
def orient_top_view(model):
    """
    Make occlusal plane horizontal by mapping the thinnest bbox axis to Z.
    Then make sure the occlusal side faces +Z (towards the camera).
    """
    dims = model.dimensions
    xyz = [dims.x, dims.y, dims.z]
    min_idx = xyz.index(min(xyz))
    if min_idx == 0:          # X is thinnest -> rotate +90° about Y (X -> Z)
        model.rotation_euler.rotate_axis('Y', math.radians(90))
    elif min_idx == 1:        # Y is thinnest -> rotate -90° about X (Y -> Z)
        model.rotation_euler.rotate_axis('X', math.radians(-90))
    bpy.context.view_layer.update()

    # If the majority of face normals point down, flip 180° around X to face up
    up_count, down_count = 0, 0
    model.data.calc_normals_split()
    mw = model.matrix_world

    polys = model.data.polygons
    step = int(max(1, len(polys) // 2000))  # ensure pure int
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
    """
    Use 2D PCA on XY to remove tilt around Z so the arch appears straight.
    """
    if not HAS_NUMPY:
        return

    verts = model.data.vertices
    step = int(max(1, len(verts) // 60000))  # ensure pure int
    pts, mw = [], model.matrix_world
    for i in range(0, len(verts), step):
        v = mw @ verts[i].co
        pts.append((v.x, v.y))

    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] < 8:
        return

    mean = arr.mean(axis=0)
    X = arr - mean
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    i_max = int(np.argmax(eigvals))
    v1 = eigvecs[:, i_max]
    angle = math.atan2(v1[1], v1[0])

    model.rotation_euler.rotate_axis('Z', -angle)

    # Ensure upright orientation (open arch pointing downward on image)
    c, s = math.cos(-angle), math.sin(-angle)
    v2 = eigvecs[:, 1 - i_max]
    v2_rot_y = s * v2[0] + c * v2[1]
    if v2_rot_y < 0:
        model.rotation_euler.rotate_axis('Z', math.radians(180))
    bpy.context.view_layer.update()

# =========== MODEL PREP ===========
def prepare_model(model):
    """
    Assign material, fix normals, smooth shading, orient upright, center and scale.
    """
    # Material
    tooth_mat = create_tooth_material()
    if model.data.materials:
        model.data.materials[0] = tooth_mat
    else:
        model.data.materials.append(tooth_mat)

    # Smooth + normals outside + Auto Smooth for crisp edges
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

    # Orient & straighten
    orient_top_view(model)
    upright_with_pca(model)

    # Center to origin by bbox center
    bbox = [model.matrix_world @ mathutils.Vector(corner) for corner in model.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8.0
    model.location -= center

    # Recompute after centering and scale to a comfortable frame
    bpy.context.view_layer.update()
    dims = model.dimensions
    xy_max = max(dims.x, dims.y)
    if xy_max > 0:
        s = 150.0 / xy_max   # slightly larger in frame
        model.scale = (model.scale.x * s, model.scale.y * s, model.scale.z * s)
    bpy.context.view_layer.update()

# =========== CAMERA (TRUE TOP ORTHO) ===========
def setup_camera(model, margin=1.1):
    """
    Orthographic camera directly above the model looking down (+Z).
    Tighter margin than before to resemble the reference framing.
    """
    # Remove existing cameras/empties
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in {'CAMERA', 'EMPTY'}:
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

    # Compute center and dims
    bbox = [model.matrix_world @ mathutils.Vector(corner) for corner in model.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8.0
    dims = model.dimensions

    # Target empty
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(center.x, center.y, center.z))
    target = bpy.context.object

    # Camera on +Z
    cam_height = max(200.0, dims.z * 6.0 + 120.0)
    bpy.ops.object.camera_add(location=(center.x, center.y, center.z + cam_height))
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = max(dims.x, dims.y) * margin
    cam.data.clip_start = 0.1
    cam.data.clip_end = 10000.0

    # Track to center and enforce top-down rotation
    con = cam.constraints.new(type='TRACK_TO')
    con.target = target
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)

    return cam, center, target

# =========== MAIN ===========
def main():
    scene = setup_render_settings()

    # Gather .ply files (allow subfolders)
    ply_files = sorted(list(BASE_DIR.rglob("*.ply")))
    if not ply_files:
        print(f"No PLY files found in {BASE_DIR}")
        return

    files = ply_files if TEST_FIRST_ONLY else ply_files
    print(f"Found {len(ply_files)} PLY files. Rendering {len(files)} file(s)...")

    rendered = 0
    for ply_file in files:
        try:
            # Remove all mesh objects
            bpy.ops.object.select_all(action='DESELECT')
            for obj in list(bpy.context.scene.objects):
                if obj.type == 'MESH':
                    obj.select_set(True)
            if bpy.context.selected_objects:
                bpy.ops.object.delete()

            # Import PLY
            bpy.ops.import_mesh.ply(filepath=str(ply_file))
            meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
            if not meshes:
                print(f"Failed to import: {ply_file.name}")
                continue
            model = meshes[0]
            print(f"Now processing: {ply_file.name}")
            # Jaw type detection
            if "upper" in ply_file.name.lower():
                jaw_type = "upper"
            elif "lower" in ply_file.name.lower():
                jaw_type = "lower"
            else:
                jaw_type = "unknown"
            print(f"Detected jaw type: {jaw_type}")

            # Prepare model (material, normals, orientation, centering, scaling)
            prepare_model(model)

            # Camera + lighting
            cam, center, target = setup_camera(model, margin=1.1)
            setup_lighting(center, target)

            # Output path
            out_name = f"{ply_file.stem}_top.png"
            scene.render.filepath = str(OUTPUT_DIR / out_name)

            # Render
            bpy.ops.render.render(write_still=True)
            print(f"Rendered: {out_name}")
            rendered += 1

        except Exception as e:
            print(f"Error rendering {ply_file.name}: {e}")

    print(f"Done. Rendered {rendered} image(s) to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
