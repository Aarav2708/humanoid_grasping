import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask
import time
import os
import json

# ---------- RealSense Initialization ----------
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

# ---------- Capture RGB and Depth ----------
time.sleep(2)
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
aligned_depth_frame = aligned_frames.get_depth_frame()
color_frame = aligned_frames.get_color_frame()

if not aligned_depth_frame or not color_frame:
    pipeline.stop()
    exit()

rgb_raw = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(aligned_depth_frame.get_data())
depth_map = (depth_image * depth_scale).astype(np.float32)

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
              [0, intrinsics.fy, intrinsics.ppy],
              [0, 0, 1]])

# Normalize RGB image to float32 and convert to OpenCV format
rgb = ImageConverter.from_numpy_format(rgb_raw.astype(np.float32) / 255.0).image_in_opencv_format

# ---------- Iterate through segmented_waypoints ----------
seg_dir = "segmented_waypoints"
save_dir = "captured_data"
os.makedirs(save_dir, exist_ok=True)

for file in os.listdir(seg_dir):
    if not file.endswith(".json"):
        continue

    name = os.path.splitext(file)[0]
    json_path = os.path.join(seg_dir, file)

    # Load saved waypoints
    with open(json_path, "r") as f:
        loaded_waypoints = json.load(f)

    # Flatten waypoints to COCO format
    poly_list = []
    for contour in loaded_waypoints:
        flat = [coord for pt in contour for coord in pt]  # flatten [[x, y], ...]
        poly_list.append(flat)

    # Generate segmentation mask
    segmap = np.zeros_like(depth_map, dtype=np.uint16)
    obj_map = BinarySegmentationMask.from_coco_segmentation_mask(
        poly_list,
        width=segmap.shape[1],
        height=segmap.shape[0]
    ).bitmap
    segmap[obj_map == 1] = 1
    segmap = segmap.astype(np.float32)

    # Save to .npy
    output_path = os.path.join(save_dir, f"{name}.npy")
    data = {
        'rgb': rgb,
        'depth': depth_map,
        'K': K,
        'seg': segmap
    }
    np.save(output_path, data)
    print(f"[INFO] Saved data to {output_path}")

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