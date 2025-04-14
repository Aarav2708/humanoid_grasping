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
conda activate zshot
export QT_QPA_PLATFORM=xcb
export DISPLAY=:1
python3 /home/hpm-mv/parent_graspnet/humanoid_grasping/zeroshot.py
conda deactivate
conda activate sam_env
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


echo "[INFO] Extracting each best grasp and sending goals one-by-one..."

# python3 - <<EOF
# import json
# import os
# import sys

# json_path = "best_grasp_pose.json"
# if not os.path.exists(json_path):
#     print("[ERROR] best_grasp_pose.json not found.")
#     sys.exit(1)

# with open(json_path, "r") as f:
#     data = json.load(f)

# if not data:
#     print("[ERROR] No grasp data found.")
#     sys.exit(1)

# for obj_id, pose in data.items():
#     pos = pose["translation"]
#     quat = pose["quaternion"]

#     print(f"[INFO] Sending goal for object: {obj_id}")
    
#     cmd = f'''
#     ros2 action send_goal /move_robot dual_panda_moveit2_interface/action/Move "{{
#     use_joint_state: false,
#     head_pose: {{
#         position: {{
#         x: 0.01999999999999999,
#         y: 0.0,
#         z: 0.06105000000000005
#         }},
#         orientation: {{
#         x: 0.0,
#         y: 0.0,
#         z: 0.0,
#         w: 1.0
#         }}
#     }},
#     left_pose: {{
#         position: {{x: {pos["x"]}, y: {pos["y"]}, z: {pos["z"]}}},
#         orientation: {{x: {quat["x"]}, y: {quat["y"]}, z: {quat["z"]}, w: {quat["w"]}}}
#     }},
#     right_pose: {{
#         position: {{
#         x: 0.11377056587257307,
#         y: -0.202,
#         z: -0.7338852146903048
#         }},
#         orientation: {{
#         x: 0.0,
#         y: -0.0868902910275923,
#         z: 0.0,
#         w: 0.9962178864711978
#         }}
#     }},
#     joint_values: []
#     }}"
#     '''
#     os.system(cmd)
# EOF