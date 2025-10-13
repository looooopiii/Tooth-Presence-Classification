# Command to execute: /home/user/tbrighton/blender/blender-3.6.11-linux-x64/blender --background --python /home/user/tbrighton/blender-scripts/rendering_views_ply_corrected.py

import bpy
import addon_utils
from pathlib import Path
import mathutils
import os

# === CONFIG ===
BASE_DIR = Path("/local/scratch/datasets/IntraOralScans/data")
OUTPUT_DIR = Path("/home/user/tbrighton/blender_outputs/test_ply_views")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === GPU CONFIGURATION ===
available_gpus = [3, 1]

# === ENABLE PLY IMPORT ADDON ===
addon_utils.enable("io_mesh_ply")

# === CORRECTED JAW-SPECIFIC CAMERA SETUP ===
def get_camera_views(jaw_type):
    """Return appropriate camera view based on jaw type"""
    if jaw_type.lower() == 'upper':
        return {"top": (0, -100, 0)}      # Front view for upper jaw
    elif jaw_type.lower() == 'lower':
        return {"lingual": (0, 100, 0)}   # Back view for lower jaw
    else:
        return {"top": (0, -100, 0)}     #default

# === GPU SETUP ===
def setup_gpu_rendering():
    scene = bpy.context.scene
    scene.cycles.device = 'GPU'
    
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.refresh_devices()
    
    enabled_count = 0
    cuda_device_index = 0
    
    for device in cprefs.devices:
        if device.type in {'CUDA', 'OPENCL'}:
            if cuda_device_index in available_gpus:
                device.use = True
                enabled_count += 1
            else:
                device.use = False
            cuda_device_index += 1
        else:
            device.use = False
    
    print(f"Enabled {enabled_count} GPU(s) for rendering")

# === RENDER SETTINGS ===
def setup_render_settings():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    setup_gpu_rendering()
    
    scene.cycles.samples = 64
    scene.cycles.device = 'GPU'
    scene.cycles.feature_set = 'SUPPORTED'
    
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    
    # Background
    if scene.world is None:
        scene.world = bpy.data.worlds.new("SceneWorld")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.1, 0.1, 0.1, 1)
    bg.inputs[1].default_value = 0.2
    
    return scene

# === SIMPLE LIGHTING ===
def setup_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()
    
    # Three simple lights
    bpy.ops.object.light_add(type='SUN', location=(50, -50, 100))
    key_light = bpy.context.object
    key_light.data.energy = 6
    key_light.data.angle = 0.2
    
    bpy.ops.object.light_add(type='SUN', location=(-30, 30, 80))
    fill_light = bpy.context.object
    fill_light.data.energy = 4
    fill_light.data.angle = 0.3
    
    bpy.ops.object.light_add(type='SUN', location=(0, 0, -60))
    bottom_light = bpy.context.object
    bottom_light.data.energy = 2
    bottom_light.data.angle = 0.4

# === SIMPLE MATERIAL ===
def create_tooth_material():
    material = bpy.data.materials.new(name="ToothMaterial")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    
    bsdf = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.95, 0.92, 0.87, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.3
    bsdf.inputs['Specular'].default_value = 0.4
    bsdf.inputs['Subsurface'].default_value = 0.0
    
    output = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return material

# === SIMPLIFIED MODEL PREPARATION (NO ROTATIONS) ===
def prepare_model(model, jaw_type):
    tooth_material = create_tooth_material()
    if model.data.materials:
        model.data.materials[0] = tooth_material
    else:
        model.data.materials.append(tooth_material)
    
    bpy.ops.object.select_all(action='DESELECT')
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    bpy.ops.object.shade_smooth()
    
    # Just center the model - no rotations
    bbox = [model.matrix_world @ mathutils.Vector(corner) for corner in model.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8.0
    model.location -= center
    
    bpy.context.view_layer.update()
    
    dimensions = model.dimensions
    max_dimension = max(dimensions)
    optimal_distance = max_dimension * 2.5
    
    return optimal_distance

# === CAMERA SETUP ===
def setup_camera(location, optimal_distance):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in ['CAMERA', 'EMPTY']:
            obj.select_set(True)
    bpy.ops.object.delete()
    
    direction_vec = mathutils.Vector(location).normalized()
    cam_location = direction_vec * optimal_distance
    
    bpy.ops.object.camera_add(location=cam_location)
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    target = bpy.context.object
    
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    camera.data.type = 'PERSP'
    camera.data.lens = 85
    camera.data.sensor_width = 36
    
    return camera

# === VALIDATION ===
def validate_ply_import(ply_file):
    if not bpy.context.selected_objects:
        return None
    
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        return None
    
    model = meshes[0]
    if len(model.data.vertices) == 0:
        return None
    
    print(f" Imported: {ply_file.name} ({len(model.data.vertices)} vertices)")
    return model

# === FILENAME PARSING ===
def parse_filename(filename):
    stem = filename.stem
    patient_id = stem.split('_')[0]
    
    filename_lower = stem.lower()
    if 'lower' in filename_lower:
        jaw_type = 'lower'
    elif 'upper' in filename_lower:
        jaw_type = 'upper'
    else:
        jaw_type = 'unknown'
        
    return patient_id, jaw_type

# === MAIN LOOP ===
def main():
    scene = setup_render_settings()
    
    total_processed = 0
    total_rendered = 0
    errors = []
    
    print(f"Processing PLY files from: {BASE_DIR}")
    
    ply_files = list(BASE_DIR.glob("*.ply"))
    if not ply_files:
        print(f"No PLY files found in {BASE_DIR}")
        return
    
    print(f"Found {len(ply_files)} PLY files to process")
    
    for ply_file in ply_files:
        total_processed += 1
        patient_id, jaw_type = parse_filename(ply_file)
        
        print(f"\n Processing: {ply_file.name} ({jaw_type})")
        
        # Get jaw-specific camera views
        camera_views = get_camera_views(jaw_type)
        view_name = list(camera_views.keys())[0]
        camera_angle = camera_views[view_name]
        print(f" Using camera view: {view_name} {camera_angle}")
        
        # Check existing files
        all_views_exist = all(
            (OUTPUT_DIR / f"{patient_id}_{jaw_type}_{view_name}.png").exists()
            for view_name in camera_views.keys()
        )
        
        if all_views_exist:
            print(f"Skipping: View exists for {patient_id}_{jaw_type}")
            total_rendered += len(camera_views)
            continue
        
        try:
            # Clear mesh objects
            bpy.ops.object.select_all(action='DESELECT')
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    obj.select_set(True)
            bpy.ops.object.delete()
            
            # Import PLY
            bpy.ops.import_mesh.ply(filepath=str(ply_file))
            model = validate_ply_import(ply_file)
            if model is None:
                errors.append(f"Import failed: {ply_file}")
                continue
            
            # Setup lighting and model (no rotations)
            setup_lighting()
            optimal_distance = prepare_model(model, jaw_type)
            
            # Render the view
            for view_name, camera_location in camera_views.items():
                out_path = OUTPUT_DIR / f"{patient_id}_{jaw_type}_{view_name}.png"
                
                if out_path.exists():
                    print(f"  View exists: {view_name}")
                    total_rendered += 1
                    continue
                
                print(f"  Rendering {view_name} view...")
                setup_camera(camera_location, optimal_distance)
                scene.render.filepath = str(out_path)
                bpy.ops.render.render(write_still=True)
                print(f"  Saved: {out_path.name}")
                total_rendered += 1
                
        except Exception as e:
            error_msg = f"Error processing {ply_file}: {str(e)}"
            print(f"  {error_msg}")
            errors.append(error_msg)
            continue
    
    # Summary
    print(f"\n{'='*50}")
    print(f"RENDERING COMPLETE!")
    print(f"PLY files processed: {total_processed}")
    print(f"Images rendered: {total_rendered}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors[:5]:
            print(f"  • {error}")

if __name__ == "__main__":
    main()
