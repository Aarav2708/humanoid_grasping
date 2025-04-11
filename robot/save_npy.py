import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import json
import os
import time
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask

# ---------- RealSense Setup ----------
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

# ---------- Capture Frame ----------
time.sleep(2)
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
aligned_depth_frame = aligned_frames.get_depth_frame()
color_frame = aligned_frames.get_color_frame()

if not aligned_depth_frame or not color_frame:
    pipeline.stop()
    exit()

rgb = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(aligned_depth_frame.get_data())
depth_map = (depth_image * depth_scale).astype(np.float32)

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
              [0, intrinsics.fy, intrinsics.ppy],
              [0, 0, 1]])

# Normalize RGB for Open3D
rgb = ImageConverter.from_numpy_format(rgb.astype(np.float32) / 255.0).image_in_opencv_format

# ---------- Load Segmentation Waypoints ----------
with open("./segmented_waypoints.json", "r") as f:
    all_waypoints = json.load(f)

# ---------- Build Dataset for All Objects ----------
multi_object_data = {}

for obj_name, contours in all_waypoints.items():
    if not contours:
        print(f"[WARNING] No contours found for {obj_name}, skipping...")
        continue

    # Flatten contours to COCO polygon format
    poly_list = []
    for contour in contours:
        if not contour or len(contour) < 3:
            continue
        flat = [coord for pt in contour for coord in pt]
        poly_list.append(flat)

    if not poly_list:
        print(f"[WARNING] No valid polygons for {obj_name}, skipping...")
        continue

    # Generate segmentation mask
    segmap = np.zeros_like(depth_map, dtype=np.uint16)
    obj_map = BinarySegmentationMask.from_coco_segmentation_mask(
        poly_list,
        width=segmap.shape[1],
        height=segmap.shape[0]
    ).bitmap
    segmap[obj_map == 1] = 1
    segmap = segmap.astype(np.float32)

    # Store per-object data
    multi_object_data[obj_name] = {
        'rgb': rgb,
        'depth': depth_map,
        'K': K,
        'seg': segmap
    }

# ---------- Save All Data in Single .npy ----------
output_path = 'captured_data'
os.makedirs(output_path, exist_ok=True)

np.save(os.path.join(output_path, 'multi_object_data.npy'), multi_object_data)
print(f"[INFO] Saved data for {len(multi_object_data)} objects to multi_object_data.npy")

pipeline.stop()


# import pyrealsense2 as rs
# import numpy as np
# import open3d as o3d
# import cv2
# from airo_camera_toolkit.utils.image_converter import ImageConverter
# from airo_camera_toolkit.utils.annotation_tool import Annotation, get_manual_annotations
# from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
# import time
# import os

# pipeline = rs.pipeline()
# config = rs.config()
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# profile = pipeline.start(config)
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# align_to = rs.stream.color
# align = rs.align(align_to)

# time.sleep(2)
# frames = pipeline.wait_for_frames()
# aligned_frames = align.process(frames)
# aligned_depth_frame = aligned_frames.get_depth_frame()
# color_frame = aligned_frames.get_color_frame()

# if not aligned_depth_frame or not color_frame:
#     pipeline.stop()
#     exit()

# rgb = np.asanyarray(color_frame.get_data())
# depth_image = np.asanyarray(aligned_depth_frame.get_data())
# depth_map = (depth_image * depth_scale).astype(np.float32)

# intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
# K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
#               [0, intrinsics.fy, intrinsics.ppy],
#               [0, 0, 1]])

# rgb = ImageConverter.from_numpy_format(rgb.astype(np.float32) / 255.0).image_in_opencv_format

# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     o3d.geometry.Image(rgb),
#     o3d.geometry.Image(depth_map),
#     depth_scale=1.0,
#     depth_trunc=1.0,
#     convert_rgb_to_intensity=False,
# )

# o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
#     640, 480, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
# )

# pcd_legacy = o3d.geometry.PointCloud.create_from_rgbd_image(
#     rgbd_image, o3d_intrinsics
# )

# pcd = np.asarray(pcd_legacy.points)
# segmap = np.zeros_like(depth_map, dtype=np.uint16)

# import json

# # Load saved waypoints from SAM output
# with open("./segmented_waypoints.json", "r") as f:
#     loaded_waypoints = json.load(f)

# # Format for COCO-compatible polygon
# poly_list = []
# for contour in loaded_waypoints:
#     flat = [coord for pt in contour for coord in pt]  # flatten [[x, y], ...] to [x1, y1, x2, y2, ...]
#     poly_list.append(flat)

# # Create segmentation mask
# obj_map = BinarySegmentationMask.from_coco_segmentation_mask(
#     poly_list,
#     width=segmap.shape[1],
#     height=segmap.shape[0]
# ).bitmap

# segmap[obj_map == 1] = 1

# segmap = segmap.astype(np.float32)

# # Define the directory path
# directory_path = 'captured_data'

# # Ensure the directory exists
# if not os.path.exists(directory_path):
#     os.makedirs(directory_path)

# data = {
#     'rgb': rgb,
#     'depth': depth_map,
#     'K': K,
#     'seg': segmap
# }
# np.save(os.path.join(directory_path, 'current.npy'), data)

# pipeline.stop()