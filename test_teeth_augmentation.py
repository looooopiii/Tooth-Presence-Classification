import json
import csv
import random
import numpy as np
from pathlib import Path
import shutil
import subprocess
import os

# Test paths
TEST_OBJ_PATH = "blender-scripts/dataset/TeethSeg/3DTeethLand_challenge_train_test_split/lower/0AAQ6BO3/0AAQ6BO3_lower.obj"
TEST_JSON_PATH = "blender-scripts/dataset/TeethSeg/3DTeethLand_challenge_train_test_split/lower/0AAQ6BO3/0AAQ6BO3_lower.json"
TEST_CASE_ID = "0AAQ6BO3"
TEST_OUTPUT_DIR = Path("/home/user/rhong/baseline/data_augmentation/test_data")


class TeethDataAugmentorTest:
    def __init__(self):
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (TEST_OUTPUT_DIR / "obj").mkdir(parents=True, exist_ok=True)
        (TEST_OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

    def get_tooth_labels_from_json(self, json_path):
        """
        Reads vertex labels from a JSON file and creates a mapping from tooth IDs to vertex indices.
        Returns a dictionary where keys are tooth names and values are lists of corresponding vertex indices.
        """
        if not os.path.exists(json_path):
            print(f"Error: JSON file does not exist at {json_path}")
            return {}

        with open(json_path, 'r') as f:
            data = json.load(f)

        labels = data.get("labels", [])
        if not labels:
            print("Warning: 'labels' array not found in JSON file.")
            return {}

        tooth_to_vertices = {}
        for i, label in enumerate(labels):
            if label != 0:  # 0 typically represents background or gingiva
                tooth_name = f"tooth_{label}"
                if tooth_name not in tooth_to_vertices:
                    tooth_to_vertices[tooth_name] = []
                # Vertex indices in OBJ files are 1-based
                tooth_to_vertices[tooth_name].append(i + 1)

        print(f"Found labels for {len(tooth_to_vertices)} teeth from JSON file.")
        for name, vertices in tooth_to_vertices.items():
            print(f"  {name}: {len(vertices)} vertices")
        return tooth_to_vertices

    def create_missing_teeth_variation(self, all_tooth_names):
        """Selects 1 to 10 teeth to remove."""
        if not all_tooth_names:
            return []
        num_to_remove = random.randint(1, min(len(all_tooth_names), 10))
        teeth_to_remove = random.sample(all_tooth_names, num_to_remove)
        print(f"Selecting the following {num_to_remove} teeth to remove: {teeth_to_remove}")
        return teeth_to_remove

    def modify_obj_by_labels(self, input_path, output_path, teeth_to_remove_names, tooth_labels):
        """Removes teeth from the OBJ file based on JSON labels."""
        print(f"Modifying OBJ file: {input_path} -> {output_path}")
        if not teeth_to_remove_names:
            shutil.copyfile(input_path, output_path)
            print("No teeth selected for removal, copying original file.")
            return True

        vertices_to_remove = set()
        for tooth_name in teeth_to_remove_names:
            if tooth_name in tooth_labels:
                vertices_to_remove.update(tooth_labels[tooth_name])

        if not vertices_to_remove:
            print("Warning: No corresponding tooth labels found, no removal performed.")
            shutil.copyfile(input_path, output_path)
            return True

        with open(input_path, 'r') as f:
            lines = f.readlines()

        modified_lines = []
        vertex_mapping = {}
        new_vertex_index = 1
        vertex_count = 0

        for line in lines:
            if line.startswith('v '):
                vertex_count += 1
                if vertex_count not in vertices_to_remove:
                    modified_lines.append(line)
                    vertex_mapping[vertex_count] = new_vertex_index
                    new_vertex_index += 1
                else:
                    modified_lines.append(f"# {line}")
            elif line.startswith('f '):
                face_parts = line.strip().split()
                new_face_parts = ['f']
                valid_face = True
                for part in face_parts[1:]:
                    vertex_ref = part.split('/')[0]
                    try:
                        old_vertex_idx = int(vertex_ref)
                        if old_vertex_idx in vertex_mapping:
                            new_vertex_idx = vertex_mapping[old_vertex_idx]
                            new_part = part.replace(vertex_ref, str(new_vertex_idx), 1)
                            new_face_parts.append(new_part)
                        else:
                            valid_face = False
                            break
                    except ValueError:
                        new_face_parts.append(part)
                if valid_face and len(new_face_parts) > 1:
                    modified_lines.append(' '.join(new_face_parts) + '\n')
            else:
                modified_lines.append(line)

        with open(output_path, 'w') as f:
            f.writelines(modified_lines)

        print(f"Successfully removed {len(vertices_to_remove)} vertices.")
        return True

    def create_blender_script(self, obj_path, output_image_path, debug=False):
        """Blender script: dynamically adjusts camera and renders."""
        abs_obj_path = os.path.abspath(obj_path)
        abs_output_path = os.path.abspath(output_image_path)

        script_content = f'''
import bpy
import mathutils
import math
import os

def render_model(obj_path, output_path):
    print(f"Starting render for {{obj_path}}")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.obj(filepath=obj_path)

    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not imported_objects:
        print("Error: No meshes imported!")
        return False

    # Adjust bounding box and center
    min_coords = [float('inf')]*3
    max_coords = [float('-inf')]*3
    for obj in imported_objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        local_bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        for c in local_bbox:
            for i in range(3):
                min_coords[i] = min(min_coords[i], c[i])
                max_coords[i] = max(max_coords[i], c[i])

    bbox_center = mathutils.Vector([(min_coords[i]+max_coords[i])/2 for i in range(3)])
    bbox_size = mathutils.Vector([max_coords[i]-min_coords[i] for i in range(3)])
    max_dim = max(bbox_size) if max(bbox_size) > 0 else 1.0

    for obj in imported_objects:
        obj.location -= bbox_center

    # Camera setup
    cam_distance = max_dim * 1.5
    cam_angle_azimuth = math.radians(-45)
    cam_angle_elevation = math.radians(60)

    cx = cam_distance * math.cos(cam_angle_elevation) * math.cos(cam_angle_azimuth)
    cy = cam_distance * math.cos(cam_angle_elevation) * math.sin(cam_angle_azimuth)
    cz = cam_distance * math.sin(cam_angle_elevation)
    bpy.ops.object.camera_add(location=(cx, cy, cz))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    direction = -cam.location.normalized()
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = direction.to_track_quat('-Z', 'Y')

    # Lighting
    bpy.ops.object.light_add(type='SUN', location=(10, 10, 10))
    light = bpy.context.object
    light.data.energy = 10

    # Render settings
    scene = bpy.context.scene
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128

    # Enhance contrast: use Filmic color management
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'
    
    # Set background color
    world = bpy.context.scene.world
    if world and world.use_nodes:
        bg = world.node_tree.nodes.get('Background')
        if bg:
            bg.inputs[0].default_value = (1, 1, 1, 1)

    bpy.ops.render.render(write_still=True)
    print(f"Render complete: {{output_path}}")
    return True

render_model("{abs_obj_path}", "{abs_output_path}")
'''
        return script_content

    def render_with_blender(self, obj_path, output_image_path):
        Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
        script_path = TEST_OUTPUT_DIR / "render_script.py"
        with open(script_path, 'w') as f:
            f.write(self.create_blender_script(obj_path, output_image_path))
        cmd = ["blender", "--background", "--python", str(script_path)]
        print(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Render complete: {output_image_path}")
            return True
        else:
            print(f"Render failed. Error output:")
            print(result.stderr)
            return False

    def run_test(self):
        print("Starting test...")

        # Step 1: Render original model
        print("\n--- Rendering original model ---")
        self.render_with_blender(TEST_OBJ_PATH, str(TEST_OUTPUT_DIR / "images" / f"{TEST_CASE_ID}_original.png"))

        # Step 2: Read tooth labels from JSON file
        print("\n--- Reading tooth labels from JSON file ---")
        tooth_labels = self.get_tooth_labels_from_json(TEST_JSON_PATH)
        all_tooth_names = list(tooth_labels.keys())

        # Step 3: Select teeth to remove
        teeth_to_remove = self.create_missing_teeth_variation(all_tooth_names)

        # Step 4: Modify OBJ and render
        if teeth_to_remove:
            modified_obj_path = TEST_OUTPUT_DIR / "obj" / f"{TEST_CASE_ID}_missing.obj"
            self.modify_obj_by_labels(TEST_OBJ_PATH, modified_obj_path, teeth_to_remove, tooth_labels)

            print("\n--- Rendering modified model ---")
            self.render_with_blender(modified_obj_path, str(TEST_OUTPUT_DIR / "images" / f"{TEST_CASE_ID}_missing.png"))
        else:
            print("No teeth selected for removal, skipping modification and render.")

        print("\nTest complete.")


def main():
    tester = TeethDataAugmentorTest()
    tester.run_test()


if __name__ == "__main__":
    main()
