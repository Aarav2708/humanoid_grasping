import numpy as np
import pathlib

filepath = pathlib.Path(__file__)

# read the predictions for the 'scene.npz' file
grasp_predictions_file = filepath.parents[1] / "results" / "predictions_current.npz"
grasp_predictions_dict = np.load(grasp_predictions_file, allow_pickle=True)
grasp_predictions_in_camera_frame = grasp_predictions_dict["pred_grasps_cam"]

# dict as np array, need to get object first and then select dict key for segmask
grasp_predictions_in_camera_frame = grasp_predictions_in_camera_frame.item()[1]

grasp_scores = grasp_predictions_dict["scores"]

# dict as np array, need to get object first and then select dict key
grasp_scores = grasp_scores.item()[1]

highest_idx = np.argmax(grasp_scores)
best_grasp_in_camera_frame = grasp_predictions_in_camera_frame[highest_idx]

print(best_grasp_in_camera_frame)
