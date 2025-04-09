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

# Step 1: Run SAM
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
python contact_graspnet/inference.py \
       --np_path=captured_data/current.npy \
       --local_regions --filter_grasps
conda deactivate

echo "[INFO] All steps completed."
