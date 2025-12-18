import bpy
import addon_utils
from pathlib import Path
import mathutils
import os
import sys
import math
import bmesh

# ===================== CONFIG =====================
AUG_ROOT = Path("/home/user/lzhou/week10/output/augment_test")
OUTPUT_ROOT = Path("/home/user/lzhou/week15/render_output/render_aug_test") 
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# camera views to render
CAMERA_VIEWS = {
    "top": (0.0, -1.0, 0.0),
}

# =========== UNIFIED RENDER SETTINGS ===========
def setup_render_settings():
    """统一的渲染设置 - 与其他脚本完全一致"""
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
    scene.cycles.samples = 512  # 统一高质量采样
    scene.cycles.use_adaptive_sampling = True
    
    try:
        scene.view_layers["ViewLayer"].cycles.use_denoising = True
    except Exception:
        pass

    # 统一分辨率
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 2048
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = False

    # 统一黑色背景
    if scene.world is None:
        scene.world = bpy.data.worlds.new("SceneWorld")
    scene.world.use_nodes = True
    nt = scene.world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0, 0, 0, 1)
    wout = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], wout.inputs['Surface'])

    # 统一色调映射
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'
    scene.view_settings.exposure = 1.5
    scene.view_settings.gamma = 1.0
    
    return scene


# =========== UNIFIED LIGHTING ===========
def setup_lighting(center=(0, 0, 0)):
    """统一的光照设置 - 与其他脚本完全一致"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

    # Create target for tracking
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=center)
    target = bpy.context.object
    target.name = "LightTarget"

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

    cx, cy, cz = center
    # 统一的光照配置
    add_tracked_light('SPOT', (cx, cy - 150, cz + 220), energy=250000, spot_deg=55)
    add_tracked_light('AREA', (cx - 260, cy - 200, cz + 190), energy=85000, size=420)
    add_tracked_light('AREA', (cx + 260, cy + 200, cz + 190), energy=85000, size=420)
    add_tracked_light('SPOT', (cx - 320, cy, cz + 100), energy=180000, spot_deg=60)
    add_tracked_light('SPOT', (cx + 320, cy, cz + 100), energy=180000, spot_deg=60)
    add_tracked_light('SPOT', (cx, cy - 320, cz + 80), energy=120000, spot_deg=65)


# =========== UNIFIED MATERIAL ===========
def create_tooth_material():
    """统一的牙齿材质 - 与其他脚本完全一致"""
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


# ============== HOLE FILLING UTILITY ===============
def fill_holes_in_object(obj, smooth_factor=0.2):
    """Fill all boundary holes and lightly smooth the patched area."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
    bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=smooth_factor,
                          use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(me)
    bm.free()
    try:
        me.calc_normals()
    except Exception:
        pass
    me.update()
    if hasattr(me, 'polygons'):
        for p in me.polygons:
            p.use_smooth = True
    try:
        me.use_auto_smooth = True
        me.auto_smooth_angle = math.radians(40)
    except Exception:
        pass


# ============== MODEL PREPARATION ===================
def prepare_model(model_obj):
    """准备模型 - 使用统一材质"""
    fill_holes_in_object(model_obj)
    
    # 统一材质
    mat = create_tooth_material()
    if model_obj.data.materials:
        model_obj.data.materials[0] = mat
    else:
        model_obj.data.materials.append(mat)

    # Smooth shading
    bpy.ops.object.select_all(action='DESELECT')
    model_obj.select_set(True)
    bpy.context.view_layer.objects.active = model_obj
    
    if hasattr(model_obj.data, 'polygons'):
        for p in model_obj.data.polygons:
            p.use_smooth = True
    
    try:
        model_obj.data.use_auto_smooth = True
        model_obj.data.auto_smooth_angle = math.radians(40)
    except Exception:
        pass

    # Center model
    bbox = [model_obj.matrix_world @ mathutils.Vector(corner) for corner in model_obj.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8.0
    model_obj.location -= center

    # Calculate optimal distance
    dims = model_obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    optimal_distance = max_dim * 2.8
    
    return optimal_distance


# =================== CAMERA =========================
def setup_camera(direction_vec, optimal_distance):
    """设置相机 - 保持原有视角逻辑"""
    # Delete old camera/empty objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in ['CAMERA', 'EMPTY']:
            # 跳过光照目标
            if obj.name == "LightTarget":
                continue
            obj.select_set(True)
    bpy.ops.object.delete()

    v = mathutils.Vector(direction_vec).normalized()
    cam_loc = v * float(optimal_distance)

    # Add camera
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    # Add target empty object
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    target = bpy.context.object

    c = cam.constraints.new(type='TRACK_TO')
    c.target = target
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'

    cam.data.type = 'PERSP'
    cam.data.lens = 85
    cam.data.sensor_width = 36
    
    return cam


# ================= VALIDATION ======================
def validate_obj_import(obj_path: Path):
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        print(f"[WARN] No mesh found in: {obj_path}")
        return None
    m = meshes[0]
    if len(m.data.vertices) == 0:
        print(f"[WARN] Empty mesh: {obj_path}")
        return None
    print(f"[OK] Imported: {obj_path.name} ({len(m.data.vertices)} vertices)")
    return m


# ============== CLEAN SCENE MESHES ==================
def clear_meshes():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()


# ============== ENABLE OBJ IMPORT ADDON ===========
def enable_obj_import():
    addon_utils.enable("io_scene_obj", default_set=True, persistent=True)


# ================== MAIN LOOP =======================
def main():
    enable_obj_import()
    scene = setup_render_settings()
    # Setup lighting once at origin (will be tracked to models)
    setup_lighting(center=(0, 0, 0))

    total_processed = 0
    total_rendered = 0
    errors = []

    # upper/lower
    for jaw in ["upper", "lower"]:
        jaw_dir = AUG_ROOT / jaw
        if not jaw_dir.exists():
            print(f"[INFO] Missing jaw dir: {jaw_dir}")
            continue
        print(f"\n{'='*60}")
        print(f"Processing {jaw} jaw...")
        print(f"{'='*60}")

        # each case directory
        for case_dir in sorted([p for p in jaw_dir.iterdir() if p.is_dir()]):
            obj_files = sorted(case_dir.glob("*.obj"))
            if not obj_files:
                continue

            for obj_path in obj_files:
                total_processed += 1
                try:
                    clear_meshes()
                    
                    # Import OBJ
                    bpy.ops.import_scene.obj(
                        filepath=str(obj_path),
                        axis_forward='-Z',
                        axis_up='Y'
                    )

                    model = validate_obj_import(obj_path)
                    if model is None:
                        errors.append(f"Invalid mesh: {obj_path}")
                        continue

                    optimal_distance = prepare_model(model)

                    # Output path
                    out_dir = OUTPUT_ROOT / jaw / case_dir.name
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Render views
                    for view_name, direction in CAMERA_VIEWS.items():
                        cam = setup_camera(direction, optimal_distance)
                        out_path = out_dir / f"{obj_path.stem}_{view_name}.png"
                        scene.render.filepath = str(out_path)
                        
                        print(f"  Rendering {obj_path.stem}...")
                        bpy.ops.render.render(write_still=True)
                        print(f"  ✓ Saved: {out_path.name}")
                        total_rendered += 1

                except Exception as e:
                    msg = f"Error processing {obj_path}: {e}"
                    print(f"[ERROR] {msg}")
                    errors.append(msg)
                    continue

    # Summary
    print(f"\n{'='*60}")
    print("RENDERING COMPLETE")
    print(f"{'='*60}")
    print(f"Models processed: {total_processed}")
    print(f"Images rendered:  {total_rendered}")
    print(f"Output root:      {OUTPUT_ROOT}")
    
    if errors:
        print(f"\n⚠ Errors ({len(errors)}):")
        for e in errors[:12]:
            print(f"  • {e}")
        if len(errors) > 12:
            print(f"  ... and {len(errors)-12} more")
    else:
        print("\n✓ No errors encountered.")

if __name__ == "__main__":
    main()