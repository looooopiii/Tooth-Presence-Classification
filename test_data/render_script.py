
import bpy
import mathutils
import math
import os

def render_model(obj_path, output_path):
    print(f"开始渲染 {obj_path}")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.obj(filepath=obj_path)

    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not imported_objects:
        print("错误: 没有导入到任何网格！")
        return False

    # 调整包围盒和居中
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

    # 相机设置
    cam_distance = max_dim * 1.5
    cam_angle_azimuth = math.radians(-45)
    cam_angle_elevation = math.radians(0)
    cx = cam_distance * math.cos(cam_angle_elevation) * math.cos(cam_angle_azimuth)
    cy = cam_distance * math.cos(cam_angle_elevation) * math.sin(cam_angle_azimuth)
    cz = cam_distance * math.sin(cam_angle_elevation)
    bpy.ops.object.camera_add(location=(cx, cy, cz))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    direction = -cam.location.normalized()
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = direction.to_track_quat('-Z', 'Y')

    # 灯光
    bpy.ops.object.light_add(type='SUN', location=(10, 10, 10))
    light = bpy.context.object
    light.data.energy = 10

    # 渲染设置
    scene = bpy.context.scene
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128

    # ✅ 增强对比度：使用Filmic对比
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'

    bpy.ops.render.render(write_still=True)
    print(f"渲染完成: {output_path}")
    return True

render_model("/home/user/rhong/baseline/data_augmentation/test_data/obj/0AAQ6BO3_missing.obj", "/home/user/rhong/baseline/data_augmentation/test_data/images/0AAQ6BO3_missing.png")
