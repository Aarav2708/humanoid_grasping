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
python3 zeroshot.py
python3 sam_deploy.py
conda deactivate

# Step 2: Save RGB-D + segmentation
echo "[INFO] Saving RGB-D + Segmentation Mask..."
conda activate airo-mono
python3 robot/save_npy.py
conda deactivate

# Step 3: Contact-GraspNet inference
echo "[INFO] Running Contact-GraspNet..."
conda activate contact_graspnet
python contact_graspnet/inference.py
conda deactivate

# Step 4: Extract best grasp and send goal to robot
echo "[INFO] Extracting best grasp and sending goal to robot..."

# cmd=$(python3 - <<EOF
# import json
# import os
# json_path = os.path.join("results", "best_grasp_pose.json")
# if not os.path.exists(json_path):
#     print("[ERROR] best_grasp_pose.json not found.")
#     exit(1)

# with open(json_path, "r") as f:
#     data = json.load(f)

# pos = data["translation"]
# quat = data["quaternion"]

# # Create action goal command
# goal = f'''
# ros2 action send_goal /move_robot dual_panda_moveit2_interface/action/Move "
# use_joint_state: false
# left_pose:
#   position: {{x: {pos["x"]}, y: {pos["y"]}, z: {pos["z"]}}}
#   orientation: {{x: {quat["x"]}, y: {quat["y"]}, z: {quat["z"]}, w: {quat["w"]}}}
# right_pose:
#   position: {{x: 0, y: 0, z: 0}}
#   orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}
# joint_values: []
# "
# '''
# print(goal.strip())
# EOF
# )

# # Execute ROS 2 action goal
# eval "$cmd"

echo "[INFO] All steps completed."
