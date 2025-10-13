#command to execute: /home/user/tbrighton/blender/blender-3.6.11-linux-x64/blender --background --python /home/user/tbrighton/blender-scripts/rendering_views.py

import bpy
import addon_utils
from pathlib import Path
import mathutils
import os

# === CONFIG ===
BASE_DIR = Path("/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split")
OUTPUT_DIR = Path("/home/user/tbrighton/blender_outputs/multi_views")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === ENABLE OBJ IMPORT ADDON ===
addon_utils.enable("io_scene_obj")

# === CAMERA VIEW CONFIG ===
camera_views = {
    #"occlusal": (0, 0, 100),        # Top-down
    "top": (0, -100, 0),        # Front view
    # "lingual": (0, 100, 0),         # Back/lingual view
    # "mesial": (-100, 0, 0),         # Left lateral
    # "distal": (100, 0, 0),          # Right lateral
    # "oblique_front": (-70, -70, 50), # Angled front view
    # "oblique_back": (70, 70, 50),    # Angled back view
}

# === RENDER SETTINGS ===
def setup_render_settings():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64  
    scene.cycles.device = 'GPU'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = 2048  
    scene.render.resolution_y = 2048
    scene.render.film_transparent = False
    
    # Setup world background
    if scene.world is None:
        scene.world = bpy.data.worlds.new("SceneWorld")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0, 0, 0, 1)  # Black background
    
    return scene

# === LIGHTING SETUP ===
def setup_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()
    
    # Primary key light
    bpy.ops.object.light_add(type='SUN', location=(50, -50, 100))
    key_light = bpy.context.object
    key_light.data.energy = 5
    key_light.data.angle = 0.1
    
    # Fill light
    bpy.ops.object.light_add(type='SUN', location=(-30, 30, 80))
    fill_light = bpy.context.object
    fill_light.data.energy = 2.5
    fill_light.data.angle = 0.15
    
    # Rim light for edge definition
    bpy.ops.object.light_add(type='SUN', location=(0, 100, 50))
    rim_light = bpy.context.object
    rim_light.data.energy = 3
    rim_light.data.angle = 0.2
    
    # Bottom fill to reduce harsh shadows
    bpy.ops.object.light_add(type='SUN', location=(0, 0, -50))
    bottom_light = bpy.context.object
    bottom_light.data.energy = 1.5
    bottom_light.data.angle = 0.3

# === MATERIAL SETUP ===
def create_tooth_material():
    material = bpy.data.materials.new(name="ToothMaterial")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    
    # Principled BSDF
    bsdf = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.95, 0.92, 0.87, 1.0)  # Natural tooth color
    bsdf.inputs['Roughness'].default_value = 0.2
    bsdf.inputs['Specular'].default_value = 0.3
    bsdf.inputs['IOR'].default_value = 1.6  # Tooth enamel IOR
    bsdf.inputs['Subsurface'].default_value = 0.1  # Slight subsurface scattering
    bsdf.inputs['Subsurface Color'].default_value = (0.9, 0.8, 0.6, 1.0)
    
    # Output
    output = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return material

# === MODEL PREPARATION ===
def prepare_model(model):
    # Apply material
    tooth_material = create_tooth_material()
    if model.data.materials:
        model.data.materials[0] = tooth_material
    else:
        model.data.materials.append(tooth_material)
    
    # Smooth shading
    bpy.ops.object.select_all(action='DESELECT')
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    bpy.ops.object.shade_smooth()
    
    # Center model to origin
    bbox = [model.matrix_world @ mathutils.Vector(corner) for corner in model.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8.0
    model.location -= center
    
    # Calculate optimal camera distance
    dimensions = model.dimensions
    max_dimension = max(dimensions)
    optimal_distance = max_dimension * 2.8  # Adjusted for good framing
    
    return optimal_distance

# === CAMERA SETUP ===
def setup_camera(location, optimal_distance):
    # Delete existing cameras and empties
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in ['CAMERA', 'EMPTY']:
            obj.select_set(True)
    bpy.ops.object.delete()
    
    # Normalize direction and scale by optimal distance
    direction_vec = mathutils.Vector(location).normalized()
    cam_location = direction_vec * optimal_distance
    
    # Add camera
    bpy.ops.object.camera_add(location=cam_location)
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    
    # Add target at origin for tracking
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    target = bpy.context.object
    
    # Track to constraint
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    # Camera settings
    camera.data.type = 'PERSP'
    camera.data.lens = 85  # Slightly telephoto for less distortion
    camera.data.sensor_width = 36
    
    return camera

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
    
    print(f"✅ Successfully imported: {obj_file.name} ({len(model.data.vertices)} vertices)")
    return model

# === MAIN PROCESSING LOOP ===
def main():
    scene = setup_render_settings()
    
    # Track statistics
    total_processed = 0
    total_rendered = 0
    errors = []
    
    # Process each jaw type
    for jaw in ['upper', 'lower']:
        folder = BASE_DIR / jaw
        if not folder.exists():
            print(f" Folder not found: {folder}")
            continue
            
        print(f"\n Processing {jaw} jaw...")
        
        for patient_folder in folder.iterdir():
            if not patient_folder.is_dir():
                continue
                
            obj_file = patient_folder / f"{patient_folder.name}_{jaw}.obj"
            if not obj_file.exists():
                print(f" OBJ file not found: {obj_file}")
                continue
            
            total_processed += 1
            print(f"\n Processing patient: {patient_folder.name}")
            
            # Check if all views already exist
            all_views_exist = True
            for view_name in camera_views.keys():
                out_path = OUTPUT_DIR / f"{patient_folder.name}_{jaw}_{view_name}.png"
                if not out_path.exists():
                    all_views_exist = False
                    break
            
            if all_views_exist:
                print(f"⏭️ Skipping: All views already rendered for {patient_folder.name}_{jaw}")
                total_rendered += len(camera_views)
                continue
            
            try:
                # Clear existing mesh objects
                bpy.ops.object.select_all(action='DESELECT')
                for obj in bpy.context.scene.objects:
                    if obj.type == 'MESH':
                        obj.select_set(True)
                bpy.ops.object.delete()
                
                # Import OBJ
                bpy.ops.import_scene.obj(filepath=str(obj_file), axis_forward='-Z', axis_up='Y')
                
                # Validate import
                model = validate_obj_import(obj_file)
                if model is None:
                    errors.append(f"Import failed: {obj_file}")
                    continue
                
                # Setup lighting (do this once per model)
                setup_lighting()
                
                # Prepare model and get optimal camera distance
                optimal_distance = prepare_model(model)
                
                # Render each view
                for view_name, camera_location in camera_views.items():
                    out_path = OUTPUT_DIR / f"{patient_folder.name}_{jaw}_{view_name}.png"
                    
                    # Skip if this specific view already exists
                    if out_path.exists():
                        print(f"⏭  View already exists: {out_path.name}")
                        total_rendered += 1
                        continue
                    
                    print(f"🎥 Rendering {view_name} view...")
                    
                    # Setup camera for this view
                    camera = setup_camera(camera_location, optimal_distance)
                    
                    # Set render output path
                    scene.render.filepath = str(out_path)
                    
                    # Render
                    bpy.ops.render.render(write_still=True)
                    print(f" Saved: {out_path.name}")
                    total_rendered += 1
                    
            except Exception as e:
                error_msg = f"Error processing {obj_file}: {str(e)}"
                print(f" {error_msg}")
                errors.append(error_msg)
                continue
    
    # Print summary
    print(f"\n{'='*50}")
    print(f" RENDERING COMPLETE!")
    print(f" Patients processed: {total_processed}")
    print(f"Images rendered: {total_rendered}")
    print(f" Output directory: {OUTPUT_DIR}")
    
    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   • {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
    else:
        print("✅ No errors encountered!")

if __name__ == "__main__":
    main()



# camera_views = {
#     #"occlusal": (0, 0, 100),        # Top-down 
#     "Top": (0, -100, 0),        # Front view
#     # "lingual": (0, 100, 0),         # Back/lingual view
#     # "mesial": (-100, 0, 0),         # Left lateral
#     # "distal": (100, 0, 0),          # Right lateral
#     # "oblique_front": (-70, -70, 50), # Angled front view
#     # "oblique_back": (70, 70, 50),    # Angled back view
# }
