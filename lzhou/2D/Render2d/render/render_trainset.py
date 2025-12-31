import bpy
import addon_utils
from pathlib import Path
import mathutils
import math
from mathutils import Vector, Matrix
import os

# Try to use numpy for PCA-based upright orientation
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


# === CONFIG ===
BASE_DIR = Path("/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split")
OUTPUT_DIR = Path("/home/user/lzhou/week15/render_output/train")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# only top view for trainset
ROTATION_ANGLES_DEG = [0]

# === ENABLE OBJ IMPORT ADDON ===
addon_utils.enable("io_scene_obj")


# =========== UNIFIED RENDER SETTINGS ===========
def setup_render_settings():
    """general render settings"""
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
    scene.cycles.samples = 512  # unified sample count
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
    """basic top view orientation"""
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
    """PCA-based upright orientation"""
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
    """Ensure the occlusal surface faces +Z"""
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


# === VALIDATION FUNCTIONS ===
def validate_obj_import(obj_file):
    """Validate OBJ file import and mesh quality"""
    if not bpy.context.selected_objects:
        print(f" Failed to import: {obj_file}")
        return None
    
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        print(f" No mesh found in: {obj_file}")
        return None
    
    model = meshes[0]
    if len(model.data.vertices) == 0:
        print(f" Empty mesh: {obj_file}")
        return None
    
    print(f" Imported: {obj_file.name} ({len(model.data.vertices)} vertices)")
    return model


# === MAIN PROCESSING LOOP ===
def main():
    scene = setup_render_settings()
    
    total_processed = 0
    total_rendered = 0
    errors = []
    
    for jaw in ['upper', 'lower']:
        folder = BASE_DIR / jaw
        if not folder.exists():
            print(f" Folder not found: {folder}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing {jaw} jaw...")
        print(f"{'='*60}")
        
        for patient_folder in folder.iterdir(): 
            if not patient_folder.is_dir():
                continue
                
            obj_file = patient_folder / f"{patient_folder.name}_{jaw}.obj"
            if not obj_file.exists():
                print(f" OBJ file not found: {obj_file}")
                continue
            
            total_processed += 1
            print(f"\n[{total_processed}] Processing: {patient_folder.name}")

            # Check if already rendered
            subdir_path = OUTPUT_DIR / f"{jaw}jaw"
            out_path = subdir_path / f"{patient_folder.name}_{jaw}_top.png"
            
            if out_path.exists():
                print(f" Skipping (already rendered): {out_path.name}")
                total_rendered += 1
                continue

            try:
                # Clear existing mesh objects
                bpy.ops.object.select_all(action='DESELECT')
                for obj in bpy.context.scene.objects:
                    if obj.type == 'MESH':
                        obj.select_set(True)
                bpy.ops.object.delete()
                
                # Import OBJ
                bpy.ops.import_scene.obj(
                    filepath=str(obj_file), 
                    axis_forward='-Z', 
                    axis_up='Y'
                )
                
                # Validate import
                model = validate_obj_import(obj_file)
                if model is None:
                    errors.append(f"Import failed: {obj_file}")
                    continue
                
                # Detect jaw type
                jaw_type = jaw  # 'upper' or 'lower'
                print(f"Jaw type: {jaw_type}")
                
                # Prepare model once for all views
                prepare_model(model, jaw_type)
                bpy.context.view_layer.update()

                # Setup camera and lighting (unified settings)
                cam, center, target = setup_camera(model, margin=1.1)
                setup_lighting(center, target)
                
                # Output path
                subdir_path = OUTPUT_DIR / f"{jaw}jaw"
                subdir_path.mkdir(parents=True, exist_ok=True)
                out_name = f"{patient_folder.name}_{jaw}_top.png"
                out_path = subdir_path / out_name
                scene.render.filepath = str(out_path)
                
                # Render
                print(f"  Rendering...")
                bpy.ops.render.render(write_still=True)
                print(f"  Saved: {out_name}")
                total_rendered += 1
                    
            except Exception as e:
                error_msg = f"Error processing {obj_file}: {str(e)}"
                print(f" {error_msg}")
                errors.append(error_msg)
                continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RENDERING COMPLETE!")
    print(f"{'='*60}")
    print(f"Patients processed: {total_processed}")
    print(f"Images rendered: {total_rendered}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if errors:
        print(f"\n Errors encountered ({len(errors)}):")
        for error in errors[:10]:
            print(f"  • {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("\n No errors encountered!")


if __name__ == "__main__":
    main()