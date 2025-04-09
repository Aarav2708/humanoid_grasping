import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Load the saved data
data = np.load('captured_data/current.npy', allow_pickle=True).item()
rgb = data['rgb']
depth_map = data['depth']
K = data['K']
segmap = data['seg']

# Create RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(rgb),
    o3d.geometry.Image(depth_map),
    depth_scale=1.0,
    depth_trunc=1.0,
    convert_rgb_to_intensity=False,
)

# Set camera intrinsics
o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    640, 480, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
)

# Create point cloud from RGBD image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, o3d_intrinsics
)

# Extract points that belong to the segmented object
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
segmented_indices = np.where(segmap.flatten() == 1)[0]

# Make sure indices are within bounds
valid_indices = segmented_indices[segmented_indices < len(points)]
segmented_points = points[valid_indices]
segmented_colors = colors[valid_indices]

# Create a new point cloud for the segmented object
segmented_pcd = o3d.geometry.PointCloud()
segmented_pcd.points = o3d.utility.Vector3dVector(segmented_points)
segmented_pcd.colors = o3d.utility.Vector3dVector(segmented_colors)

# Remove outliers for more stable pose estimation
segmented_pcd, _ = segmented_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Compute the oriented bounding box
oriented_bbox = segmented_pcd.get_oriented_bounding_box()
oriented_bbox.color = (1, 0, 0)  # Red color for the bounding box

# Get the 8 vertices of the bounding box
box_points = np.asarray(oriented_bbox.get_box_points())

# Get the center of the bounding box (3D translation)
center = oriented_bbox.center

# Get the rotation matrix (3D orientation)
rotation_matrix = oriented_bbox.R

# Get the dimensions of the bounding box
extent = oriented_bbox.extent

# Convert rotation matrix to Euler angles (in radians)
rotation_matrix_copy = np.array(rotation_matrix, copy=True)
r = Rotation.from_matrix(rotation_matrix_copy)
euler_angles = r.as_euler('xyz', degrees=True)

print("3-DOF Pose Estimation Results:")
print(f"Translation (x, y, z): {center}")
print(f"Rotation (Euler angles in degrees): {euler_angles}")
print(f"Dimensions (width, height, depth): {extent}")

# Create a coordinate frame at the center of the bounding box
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=center
)
coordinate_frame.rotate(rotation_matrix, center=center)

# Visualize the point cloud, bounding box, and coordinate frame
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.add_geometry(segmented_pcd)
visualizer.add_geometry(oriented_bbox)
visualizer.add_geometry(coordinate_frame)

# Set some visualization options
opt = visualizer.get_render_option()
opt.background_color = np.array([0, 0, 0])
opt.point_size = 3.0

# Run the visualizer
visualizer.run()
visualizer.destroy_window()
def draw_3d_bbox_on_image(rgb_image, box_points, K):
    # Project 3D points to 2D image plane
    box_points_2d = []
    for point in box_points:
        # Project point: p' = K * p
        x, y, z = point
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
        box_points_2d.append([int(u), int(v)])
    
    box_points_2d = np.array(box_points_2d).astype(np.int32)
    
    # Define the edges of the cube
    edges = [ 
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ]
    
    # Draw the edges
    img = rgb_image.copy()
    for edge in edges:
        pt1 = tuple(box_points_2d[edge[0]])
        pt2 = tuple(box_points_2d[edge[1]])
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    return img

# Project the 3D bounding box to the image
bbox_image = draw_3d_bbox_on_image(rgb, box_points, K)

# Display the image with the projected bounding box
plt.figure(figsize=(10, 8))
plt.imshow(bbox_image)
plt.title('3D Bounding Box Projected onto RGB Image')
plt.axis('off')
plt.show()



# import numpy as np
# import json
# import os
# os.environ["OPEN3D_VISUALIZER"] = "legacy"
# import open3d as o3d

# # Load .npy data
# data = np.load("captured_data/current.npy", allow_pickle=True).item()
# depth_map = data["depth"]
# segmap = data["seg"]
# K = data["K"]

# # Extract 3D points from segmentation
# v_coords, u_coords = np.where(segmap == 1)
# z = depth_map[v_coords, u_coords]
# valid = z > 0
# z = z[valid]
# u_coords = u_coords[valid]
# v_coords = v_coords[valid]

# fx, fy = K[0, 0], K[1, 1]
# cx, cy = K[0, 2], K[1, 2]

# x = (u_coords - cx) * z / fx
# y = (v_coords - cy) * z / fy
# points = np.vstack((x, y, z)).T

# # Estimate centroid
# centroid = points.mean(axis=0)
# pose_xyz = {
#     "x": float(centroid[0]),
#     "y": float(centroid[1]),
#     "z": float(centroid[2])
# }
# with open("object_position_xyz.json", "w") as f:
#     json.dump(pose_xyz, f, indent=4)

# print("[INFO] Estimated object position:", pose_xyz)

# # Open3D visualization with bounding box
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# # Bounding box (choose one)
# aabb = pcd.get_axis_aligned_bounding_box()
# aabb.color = (1, 0, 0)  # Red

# obb = pcd.get_oriented_bounding_box()
# obb.color = (0, 1, 0)  # Green

# # Coordinate frame
# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

# # Visualize with both AABB and OBB
# o3d.visualization.draw_geometries([pcd, aabb, obb, frame])
