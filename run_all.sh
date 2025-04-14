#!/bin/bash

# Conda init
__conda_setup="$('/home/hpm-mv/miniconda3/condabin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/hpm-mv/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/hpm-mv/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/hpm-mv/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Step 1: Run SAM and Zero-shot
echo "[INFO] Running SAM and ZERO-SHOT RECOGNITION..."
conda activate sam_env
export QT_QPA_PLATFORM=xcb
export DISPLAY=:1
python3 /home/hpm-mv/parent_graspnet/humanoid_grasping/zeroshot.py
python3 /home/hpm-mv/parent_graspnet/humanoid_grasping/sam_deploy.py
conda deactivate

# Step 2: Save RGB-D + segmentation
echo "[INFO] Saving RGB-D + Segmentation Mask..."
conda activate airo-mono
python3 /home/hpm-mv/parent_graspnet/humanoid_grasping/robot/save_npy.py
conda deactivate

# Step 3: Contact-GraspNet inference
echo "[INFO] Running Contact-GraspNet..."
conda activate contact_graspnet
python3 /home/hpm-mv/parent_graspnet/humanoid_grasping/contact_graspnet/inference.py
conda deactivate

# Step 4: Extract best grasp and send goal to robot
# Step 4: Extract best grasps and send goal to robot one by one
echo "[INFO] Extracting each best grasp and sending goals one-by-one..."
ros2 launch reachy_moveit_config demo.launch.py &
sleep 5  # Give it time to start
ros2 run dual_panda_moveit2 temp &
sleep 5
python3 - <<EOF
import json
import os
import sys

json_path = "best_grasp_pose.json"
if not os.path.exists(json_path):
    print("[ERROR] best_grasp_pose.json not found.")
    sys.exit(1)

with open(json_path, "r") as f:
    data = json.load(f)

if not data:
    print("[ERROR] No grasp data found.")
    sys.exit(1)

for obj_id, pose in data.items():
    pos = pose["translation"]
    quat = pose["quaternion"]

    print(f"[INFO] Sending goal for object: {obj_id}")
    
    cmd = f'''
ros2 action send_goal /move_robot dual_panda_moveit2_interface/action/Move \"
use_joint_state: false
left_pose:
  position: {{x: {pos["x"]}, y: {pos["y"]}, z: {pos["z"]}}}
  orientation: {{x: {quat["x"]}, y: {quat["y"]}, z: {quat["z"]}, w: {quat["w"]}}}
right_pose:
  position: {{x: -0.05935275829217748,
      y: -0.202,
      z: 0.251014939959976}}
  orientation: {{x: 0.0,
      y: -0.0868902910275923,
      z: 0.0,
      w: 0.9962178864711978}}
joint_values: []
\"'''
    os.system(cmd)
EOF



# # Execute ROS 2 action goal
# eval "$cmd"

# echo "[INFO] All steps completed."

# : '
# ros2 launch reachy_moveit_config demo.launch.py 
# ros2 run dual_panda_moveit2 temp 
# source ~/Documents/Building-Humanoids/Software/ros2_ws/install/setup.bash

# ros2 service call /spawn_object dual_panda_moveit2_interface/srv/SpawnObject "{objects: [
#   {
#     id: 'cup',
#     type: 'box',
#     dimensions: [0.1, 0.2, 0.3],
#     pose: {
#       position: {
#         x: 0.63781619369983673,
#         y: 0.021657239645719445,
#         z: -0.050873242318630135
#       },
#       orientation: {
#         x: -0.042633306575430685,
#         y: 0.0856574309988673,
#         z: 0.09042666966630225,
#         w: 0.9912962337758482
#       }
#     }
#   },
  
# ]}"


# ros2 action send_goal /move_robot dual_panda_moveit2_interface/action/Move "{
#   use_joint_state: false,
#   head_pose: {
#     position: {
#       x: 0.01999999999999999,
#       y: 0.0,
#       z: 0.06105000000000005
#     },
#     orientation: {
#       x: 0.0,
#       y: 0.0,
#       z: 0.0,
#       w: 1.0
#     }
#   },
#   left_pose: {
#     position: {
#       x: 0.3514922559261322,
#       y: 0.06164176017045967,
#       z: -0.054040618240833206
#     },
#     orientation: {
#       x: 0.01653517252111733,
#       y: 0.10221834120654999,
#       z: 0.23234277073567186,
#       w: 0.967106424173446
#     }
#   },
#   right_pose: {
#     position: {
#       x: 0.11377056587257307,
#       y: -0.202,
#       z: -0.7338852146903048
#     },
#     orientation: {
#       x: 0.0,
#       y: -0.0868902910275923,
#       z: 0.0,
#       w: 0.9962178864711978
#     }
#   },
#   joint_values: []
# }"

# Default Pose:

# ---
# head:
#   header:
#     stamp:
#       sec: 0
#       nanosec: 0
#     frame_id: head
#   pose:
#     position:
#       x: 0.01999999999999999
#       y: 0.0
#       z: 0.06105000000000005
#     orientation:
#       x: 0.0
#       y: 0.0
#       z: 0.0
#       w: 1.0
# left_pose:
#   header:
#     stamp:
#       sec: 0
#       nanosec: 0
#     frame_id: head
#   pose:
#     position:
#       x: 0.11377056587257307
#       y: 0.178
#       z: -0.7338852146903048
#     orientation:
#       x: 0.0
#       y: -0.0868902910275923
#       z: 0.0
#       w: 0.9962178864711978
# right_pose:
#   header:
#     stamp:
#       sec: 0
#       nanosec: 0
#     frame_id: head
#   pose:
#     position:
#       x: 0.11377056587257307
#       y: -0.202
#       z: -0.7338852146903048
#     orientation:
#       x: 0.0
#       y: -0.0868902910275923
#       z: 0.0
#       w: 0.9962178864711978
