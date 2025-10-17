import bpy
import bmesh
import addon_utils
from pathlib import Path
import mathutils
import re
import math

# ===================== CONFIG =====================
AUG_ROOT = Path("/home/user/lzhou/week10/output/augment_random")   
OUTPUT_ROOT = Path("/home/user/lzhou/week10/output/render_aug_random") 
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# only top view
CAMERA_VIEWS = {"top": (0.0, -1.0, 0.0)}

# ============== RENDER / SCENE SETTINGS ===========
def setup_render_settings():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16
    scene.cycles.use_adaptive_sampling = True
    # Try to enable GPU
    try:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for d in bpy.context.preferences.addons['cycles'].preferences.devices:
            d.use = True
        scene.cycles.device = 'GPU'
    except Exception:
        scene.cycles.device = 'CPU'

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 2048
    scene.render.film_transparent = False

    # Black background
    if scene.world is None:
        scene.world = bpy.data.worlds.new("SceneWorld")
    scene.world.use_nodes = True
    nt = scene.world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new(type='ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0, 0, 0, 1)
    world_out = nt.nodes.new(type='ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], world_out.inputs['Surface'])
    return scene

def enable_obj_import():
    addon_utils.enable("io_scene_obj", default_set=True, persistent=True)

# =================== LIGHTING ======================
def setup_lighting():
    # Clear old lights
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()
    # Key light + two fill lights
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 100))
    key = bpy.context.object; key.data.energy = 5.0; key.data.angle = 0.2
    bpy.ops.object.light_add(type='SUN', location=(80, -80, 60))
    f1 = bpy.context.object; f1.data.energy = 2.5; f1.data.angle = 0.6
    bpy.ops.object.light_add(type='SUN', location=(-80, 80, 60))
    f2 = bpy.context.object; f2.data.energy = 2.5; f2.data.angle = 0.6

# =================== MATERIAL ======================
def create_tooth_material():
    mat = bpy.data.materials.new(name="ToothMaterial")
    mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    diffuse = nt.nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs['Color'].default_value = (0.95, 0.92, 0.87, 1.0)
    glossy = nt.nodes.new(type='ShaderNodeBsdfGlossy')
    glossy.inputs['Roughness'].default_value = 0.15
    mix_shader = nt.nodes.new(type='ShaderNodeMixShader')
    mix_shader.inputs['Fac'].default_value = 0.25
    out = nt.nodes.new(type='ShaderNodeOutputMaterial')
    nt.links.new(diffuse.outputs['BSDF'], mix_shader.inputs[1])
    nt.links.new(glossy.outputs['BSDF'], mix_shader.inputs[2])
    nt.links.new(mix_shader.outputs['Shader'], out.inputs['Surface'])
    return mat

# ============== MODEL PREPARATION ===================
def fill_holes_in_object(obj, smooth_factor=0.2):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    # Fill every boundary hole (including outer rim openings)
    bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
    # Light smoothing to avoid star-like shading artifacts on the cap
    bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=smooth_factor,
                          use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(me)
    bm.free()
    me.update()
    # ---- No bpy.ops here (safe in background). Use data API instead ----
    # Smooth shading without requiring context/active object
    if hasattr(me, 'polygons'):
        for p in me.polygons:
            p.use_smooth = True
    # Auto smooth for cleaner shading edges
    try:
        me.use_auto_smooth = True
        me.auto_smooth_angle = math.radians(40)
    except Exception:
        pass

def prepare_model(model_obj):
    fill_holes_in_object(model_obj)
    # Material and shading
    mat = create_tooth_material()
    if model_obj.data.materials: model_obj.data.materials[0] = mat
    else: model_obj.data.materials.append(mat)
    bpy.ops.object.select_all(action='DESELECT')
    model_obj.select_set(True)
    bpy.context.view_layer.objects.active = model_obj
    # Use data API for smoothing (operator-free)
    if hasattr(model_obj.data, 'polygons'):
        for p in model_obj.data.polygons:
            p.use_smooth = True
    try:
        model_obj.data.use_auto_smooth = True
        model_obj.data.auto_smooth_angle = math.radians(40)
    except Exception:
        pass

    # Move to origin
    bbox = [model_obj.matrix_world @ mathutils.Vector(corner) for corner in model_obj.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8.0
    model_obj.location -= center

    # Estimate camera distance (proportional to model size)
    dims = model_obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    optimal_distance = max_dim * 2.8
    return optimal_distance

# =================== CAMERA =========================
def setup_camera(direction_vec, optimal_distance):
    # Clear old camera/empty objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in ['CAMERA', 'EMPTY']:
            obj.select_set(True)
    bpy.ops.object.delete()

    v = mathutils.Vector(direction_vec).normalized()
    cam_loc = v * float(optimal_distance)
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    target = bpy.context.object
    c = cam.constraints.new(type='TRACK_TO'); c.target = target
    c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'
    cam.data.type = 'PERSP'; cam.data.lens = 85; cam.data.sensor_width = 36
    return cam

# ================= VALIDATION ======================
def validate_obj_import(obj_path: Path):
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        print(f"[WARN] No mesh found in: {obj_path}"); return None
    m = meshes[0]
    if len(m.data.vertices) == 0:
        print(f"[WARN] Empty mesh: {obj_path}"); return None
    print(f"[OK] Imported: {obj_path.name} ({len(m.data.vertices)} vertices)")
    return m

def clear_meshes():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

# ================== MAIN LOOP =======================
def main():
    enable_obj_import()
    scene = setup_render_settings()
    setup_lighting()

    total_processed = 0
    total_rendered = 0
    errors = []

    # Try both upper and lower (skip if not exist)
    for jaw in ["upper", "lower"]:
        jaw_dir = AUG_ROOT / jaw
        if not jaw_dir.exists():
            print(f"[INFO] Missing jaw dir: {jaw_dir}"); continue
        print(f"\n=== Processing {jaw} ===")

        def _randcopy_key(p: Path):
            m = re.search(r'_randomcopy_(\d+)\.obj$', p.name)
            return int(m.group(1)) if m else 10**9

        for case_dir in sorted([p for p in jaw_dir.iterdir() if p.is_dir()]):
            obj_files = sorted(case_dir.glob("*_randomcopy_*.obj"), key=_randcopy_key)
            # If you want to strictly render only the first 3 variants, uncomment the next line:
            # obj_files = obj_files[:3]
            if not obj_files:
                # fallback: render any .obj files if pattern not found
                obj_files = sorted(case_dir.glob("*.obj"))
            if not obj_files:
                continue

            out_dir = OUTPUT_ROOT / jaw / case_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for obj_path in obj_files:
                total_processed += 1
                try:
                    clear_meshes()
                    # Import OBJ (aligned with most dental coordinates)
                    bpy.ops.import_scene.obj(
                        filepath=str(obj_path),
                        axis_forward='-Z',
                        axis_up='Y'
                    )
                    model = validate_obj_import(obj_path)
                    if model is None:
                        errors.append(f"Invalid mesh: {obj_path}"); continue

                    optimal_distance = prepare_model(model)

                    # Render from each view
                    for view_name, direction in CAMERA_VIEWS.items():
                        setup_camera(direction, optimal_distance)
                        out_path = out_dir / f"{obj_path.stem}_{view_name}.png"
                        scene.render.filepath = str(out_path)
                        bpy.ops.render.render(write_still=True)
                        print(f"[SAVE] {out_path}")
                        total_rendered += 1

                except Exception as e:
                    msg = f"Error processing {obj_path}: {e}"
                    print("[ERROR]", msg); errors.append(msg); continue

    # Summary
    print("\n" + "="*60)
    print(" RENDERING COMPLETE")
    print(f"   Models processed: {total_processed}")
    print(f"   Images rendered:  {total_rendered}")
    print(f"   Output root:      {OUTPUT_ROOT}")
    if errors:
        print(f"\n Errors ({len(errors)}):")
        for e in errors[:12]: print("  •", e)
        if len(errors) > 12: print(f"  ... and {len(errors)-12} more")
    else:
        print(" No errors encountered.")

if __name__ == "__main__":
    main()