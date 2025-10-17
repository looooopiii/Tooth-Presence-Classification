import bpy
import addon_utils
from pathlib import Path
import mathutils
import os
import sys
import re
import torch
import subprocess

# ===================== CONFIG =====================
AUG_ROOT = Path("/home/user/lzhou/week10/output/augment_test")
OUTPUT_ROOT = Path("/home/user/lzhou/week10/output/render_aug_test") 
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# camera views to render
CAMERA_VIEWS = {
    "top": (0.0, -1.0, 0.0),
}

# ============== RENDER / SCENE SETTINGS ===========
def setup_render_settings():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16
    scene.cycles.use_adaptive_sampling = True
    # GPU or CPU
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

    # World background to black
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

# ============== ENABLE OBJ IMPORT ADDON ===========
def enable_obj_import():
    addon_utils.enable("io_scene_obj", default_set=True, persistent=True)

# =================== LIGHTING ======================
def setup_lighting():
    # delete existing lights
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()

    # Top sunlight
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 100))
    key = bpy.context.object
    key.data.energy = 5.0
    key.data.angle = 0.2

    # Side fill light
    bpy.ops.object.light_add(type='SUN', location=(80, -80, 60))
    fill1 = bpy.context.object
    fill1.data.energy = 2.5
    fill1.data.angle = 0.6

    bpy.ops.object.light_add(type='SUN', location=(-80, 80, 60))
    fill2 = bpy.context.object
    fill2.data.energy = 2.5
    fill2.data.angle = 0.6

# =================== MATERIAL ======================
def create_tooth_material():
    mat = bpy.data.materials.new(name="ToothMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    diffuse = nt.nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs['Color'].default_value = (0.95, 0.92, 0.87, 1.0)

    glossy = nt.nodes.new(type='ShaderNodeBsdfGlossy')
    glossy.inputs['Roughness'].default_value = 0.15

    mix_shader = nt.nodes.new(type='ShaderNodeMixShader')
    mix_shader.inputs['Fac'].default_value = 0.25  # 25% specular, 75% diffuse

    out = nt.nodes.new(type='ShaderNodeOutputMaterial')

    nt.links.new(diffuse.outputs['BSDF'], mix_shader.inputs[1])
    nt.links.new(glossy.outputs['BSDF'], mix_shader.inputs[2])
    nt.links.new(mix_shader.outputs['Shader'], out.inputs['Surface'])
    return mat

# ============== MODEL PREPARATION ===================
def prepare_model(model_obj):
    # material
    mat = create_tooth_material()
    if model_obj.data.materials:
        model_obj.data.materials[0] = mat
    else:
        model_obj.data.materials.append(mat)

    # smooth shading
    bpy.ops.object.select_all(action='DESELECT')
    model_obj.select_set(True)
    bpy.context.view_layer.objects.active = model_obj
    bpy.ops.object.shade_smooth()

    bbox = [model_obj.matrix_world @ mathutils.Vector(corner) for corner in model_obj.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8.0
    model_obj.location -= center

    # scale to fit in unit cube
    dims = model_obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    optimal_distance = max_dim * 2.8
    return optimal_distance

# =================== CAMERA =========================
def setup_camera(direction_vec, optimal_distance):
    # delete old camera/empty objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in ['CAMERA', 'EMPTY']:
            obj.select_set(True)
    bpy.ops.object.delete()

    v = mathutils.Vector(direction_vec).normalized()
    cam_loc = v * float(optimal_distance)

    # add camera
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    # add target empty object, track to origin
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

# ================== MAIN LOOP =======================
def main():
    enable_obj_import()
    scene = setup_render_settings()
    setup_lighting()

    total_processed = 0
    total_rendered = 0
    errors = []

    # upper/lower
    for jaw in ["upper", "lower"]:
        jaw_dir = AUG_ROOT / jaw
        if not jaw_dir.exists():
            print(f"[INFO] Missing jaw dir: {jaw_dir}")
            continue
        print(f"\n=== Processing {jaw} ===")

        # each case directory
        for case_dir in sorted([p for p in jaw_dir.iterdir() if p.is_dir()]):
            # find all .obj files in the directory
            obj_files = sorted(case_dir.glob("*.obj"))
            if not obj_files:
                continue

            for obj_path in obj_files:
                total_processed += 1
                try:
                    clear_meshes()
                    # import obj
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

                    # output path: /render_aug_test/{jaw}/{case}/{stem}_top.png
                    out_dir = OUTPUT_ROOT / jaw / case_dir.name
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # render top views
                    for view_name, direction in CAMERA_VIEWS.items():
                        cam = setup_camera(direction, optimal_distance)
                        out_path = out_dir / f"{obj_path.stem}_{view_name}.png"
                        scene.render.filepath = str(out_path)
                        bpy.ops.render.render(write_still=True)
                        print(f"[SAVE] {out_path}")
                        total_rendered += 1

                except Exception as e:
                    msg = f"Error processing {obj_path}: {e}"
                    print("[ERROR]", msg)
                    errors.append(msg)
                    continue

    # summary
    print("\n" + "=" * 60)
    print(" RENDERING COMPLETE")
    print(f"   Models processed: {total_processed}")
    print(f"   Images rendered:  {total_rendered}")
    print(f"   Output root:      {OUTPUT_ROOT}")
    if errors:
        print(f"\n Errors ({len(errors)}):")
        for e in errors[:12]:
            print("  •", e)
        if len(errors) > 12:
            print(f"  ... and {len(errors)-12} more")
    else:
        print(" No errors encountered.")

if __name__ == "__main__":
    main()