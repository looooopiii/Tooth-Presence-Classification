import open3d as o3d
import numpy as np
import os

# Load mesh
mesh = o3d.io.read_triangle_mesh("0EJBIPTC_lower.obj")
mesh.compute_vertex_normals()

# Get bounding box and center
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
extent = bbox.get_extent()
max_extent = max(extent)

print(f"Original mesh center: {center}")
print(f"Mesh extent (x,y,z): {extent}")

# Center mesh at origin by translating it
mesh.translate(-center)

# After centering, new center should be near zero
bbox_centered = mesh.get_axis_aligned_bounding_box().get_center()
print(f"Centered mesh center: {bbox_centered}")

# Camera distance from origin (centered mesh)
distance = max_extent * 2.0

def set_view(vis, eye, lookat=[0,0,0], up=[0,1,0]):
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()

    forward = np.array(lookat) - np.array(eye)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up_corrected = np.cross(forward, right)

    R = np.vstack([right, up_corrected, forward]).T
    t = -R @ np.array(eye)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(params)

output_folder = "./Sliced_images"
os.makedirs(output_folder, exist_ok=True)

# Define camera views (eye positions) relative to centered mesh
views = {
    "top":    {"eye": [0, 0, distance],    "up": [0, 1, 0]},
    "bottom": {"eye": [0, 0, -distance],   "up": [0, -1, 0]},
    "front":  {"eye": [0, -distance, 0],   "up": [0, 0, 1]},
    "back":   {"eye": [0, distance, 0],    "up": [0, 0, 1]},
    "left":   {"eye": [-distance, 0, 0],   "up": [0, 0, 1]},
    "right":  {"eye": [distance, 0, 0],    "up": [0, 0, 1]},
}

print("\nCamera views (eye coordinates):")
for name, params in views.items():
    print(f"{name}: eye = {params['eye']}, up = {params['up']}")

for view_name, params in views.items():
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    set_view(vis, params["eye"], lookat=[0,0,0], up=params["up"])
    vis.poll_events()
    vis.update_renderer()

    save_path = os.path.join(output_folder, f"{view_name}_view.png")
    vis.capture_screen_image(save_path)
    vis.destroy_window()
    print(f"Saved {save_path}")
