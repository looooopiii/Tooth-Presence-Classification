# from pytorch3d.io import load_obj
# from pytorch3d.renderer import (
#     FoVPerspectiveCameras, look_at_view_transform,
#     RasterizationSettings, MeshRenderer, MeshRasterizer
# )
# import torch
# import matplotlib.pyplot as plt
# # Load OBJ
# verts, faces, _ = load_obj("00OMSZGW_upper.obj")

# # Set up renderer
# R, T = look_at_view_transform(dist=3, elev=0, azim=0)  # Adjust angles
# cameras = FoVPerspectiveCameras(device="cpu", R=R, T=T)
# raster_settings = RasterizationSettings(image_size=512)
# renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras, raster_settings))

# # Render and save
# image = renderer(verts, faces)
# plt.imshow(image[0, ..., :3].cpu().numpy())
# plt.savefig("front_view.png")

# import trimesh
# import matplotlib.pyplot as plt

# # Load OBJcls

# mesh = trimesh.load("model.obj")

# # Render orthographic projections
# for view in ['x', 'y', 'z', '-x', '-y', '-z']:
#     # Create a scene and set the camera direction
#     scene = mesh.scene()
#     scene.camera_transform = scene.camera.look_at(
#         directions=[view]
#     )
#     # Save the image
#     img = scene.save_image(resolution=(1024, 1024))
#     with open(f"{view}_view.png", "wb") as f:
#         f.write(img)

import bpy
import os

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Set render settings (adjust as needed)
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 1920  # Width
bpy.context.scene.render.resolution_y = 1080  # Height
bpy.context.scene.render.resolution_percentage = 100

# Path to your OBJ file (change this)
obj_file = "model.obj"

# Import the OBJ file
bpy.ops.import_scene.obj(filepath=obj_file)

# (Optional) Auto-center the object
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
bpy.ops.object.location_clear()

# (Optional) Set up lighting & camera (if needed)
bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 5))
bpy.ops.object.camera_add(location=(5, -5, 3))
bpy.context.scene.camera = bpy.context.object  # Set active camera

# (Optional) Simple material (if OBJ has no materials)
mat = bpy.data.materials.new(name="DefaultMaterial")
mat.use_nodes = True
for obj in bpy.context.selected_objects:
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

# Set output path (change this)
output_png = "D:/Studies/UZH-M.Sc/Master's Project/Tooth-Presence-Classification/output_images/render.png"
bpy.context.scene.render.filepath = output_png

# Render and save
bpy.ops.render.render(write_still=True)

print(f"Rendered PNG saved to: {output_png}")
