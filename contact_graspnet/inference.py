import os
import argparse
import numpy as np
import glob
import tensorflow.compat.v1 as tf
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import config_utils
from data import load_available_input_data
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image


def inference_on_folder(global_config, checkpoint_dir, input_folder, K=None, local_regions=True,
                        skip_border_objects=False, filter_grasps=True, segmap_id=None,
                        z_range=[0.2, 1.8], forward_passes=1):

    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    saver = tf.train.Saver(save_relative_paths=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')

    os.makedirs('results', exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(input_folder, "*.npy")))

    if not input_files:
        print(f"[ERROR] No .npy files found in {input_folder}")
        return

    for np_path in input_files:
        name = Path(np_path).stem
        print(f"[INFO] Processing {name}.npy")

        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(np_path, K=K)

        if segmap is None and (local_regions or filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud...')
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
                depth, cam_K, segmap=segmap, rgb=rgb,
                skip_border_objects=skip_border_objects, z_range=z_range)
        else:
            pc_segments = {}

        print(f"[INFO] Running grasp prediction...")
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
            sess, pc_full, pc_segments=pc_segments,
            local_regions=local_regions, filter_grasps=filter_grasps,
            forward_passes=forward_passes)

        # Save .npz
        npz_filename = f"results/predictions_{name}.npz"
        np.savez(npz_filename,
                 pred_grasps_cam=pred_grasps_cam,
                 scores=scores,
                 contact_pts=contact_pts)
        print(f"[INFO] Saved: {npz_filename}")

        # Extract best grasp
        try:
            pred_dict = pred_grasps_cam.item()
            score_dict = scores.item()

            if segmap_id not in pred_dict:
                print(f"[WARNING] segmap_id {segmap_id} not found in predictions.")
                continue

            best_idx = np.argmax(score_dict[segmap_id])
            best_grasp = pred_dict[segmap_id][best_idx]

            translation = best_grasp[:3, 3].tolist()
            rot_matrix = best_grasp[:3, :3]
            quaternion = R.from_matrix(rot_matrix).as_quat().tolist()

            grasp_dict = {
                "translation": {
                    "x": translation[0],
                    "y": translation[1],
                    "z": translation[2]
                },
                "quaternion": {
                    "x": quaternion[0],
                    "y": quaternion[1],
                    "z": quaternion[2],
                    "w": quaternion[3]
                }
            }

            os.makedirs("best_grasp_poses", exist_ok=True)
            json_path = f"best_grasp_poses/best_grasp_pose_{name}.json"

            with open(json_path, "w") as f:
                json.dump(grasp_dict, f, indent=4)
            print(f"[INFO] Best grasp saved to: {json_path}")

            # Optionally: visualize
            show_image(rgb, segmap)
            visualize_grasps(pc_full, {segmap_id: [best_grasp]},
                             scores={segmap_id: [score_dict[segmap_id][best_idx]]},
                             plot_opencv_cam=True, pc_colors=pc_colors)

        except Exception as e:
            print(f"[ERROR] Could not extract best grasp for {name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001',
                        help='Checkpoint directory')
    parser.add_argument('--input_folder', default='captured_data',
                        help='Folder containing .npy files')
    parser.add_argument('--K', default=None,
                        help='Flat camera matrix: "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2, 1.8],
                        help='Z value range for cropping point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False,
                        help='Use local segmentation-based regions')
    parser.add_argument('--filter_grasps', action='store_true', default=False,
                        help='Filter grasps using segmentation mask')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,
                        help='Ignore objects at border of image')
    parser.add_argument('--forward_passes', type=int, default=1,
                        help='Number of inference passes')
    parser.add_argument('--segmap_id', type=int, default=0,
                        help='Segment ID to extract best grasp from')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[],
                        help='Overwrite config parameters')

    FLAGS = parser.parse_args()
    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)

    inference_on_folder(global_config, FLAGS.ckpt_dir, FLAGS.input_folder, z_range=eval(str(FLAGS.z_range)),
                        K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps,
                        segmap_id=FLAGS.segmap_id, forward_passes=FLAGS.forward_passes,
                        skip_border_objects=FLAGS.skip_border_objects)

# import os
# import sys
# import argparse
# import numpy as np
# import time
# import glob
# import cv2
# import pathlib
# import tensorflow.compat.v1 as tf
# import json
# from scipy.spatial.transform import Rotation as R


# tf.disable_eager_execution()
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(BASE_DIR))
# import config_utils
# from data import regularize_pc_point_count, depth2pc, load_available_input_data

# from contact_grasp_estimator import GraspEstimator
# from visualization_utils import visualize_grasps, show_image

# def inference(global_config, checkpoint_dir, input_paths, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
#     """
#     Predict 6-DoF grasp distribution for given model and input data
    
#     :param global_config: config.yaml from checkpoint directory
#     :param checkpoint_dir: checkpoint directory
#     :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
#     :param K: Camera Matrix with intrinsics to convert depth to point cloud
#     :param local_regions: Crop 3D local regions around given segments. 
#     :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
#     :param filter_grasps: Filter and assign grasp contacts according to segmap.
#     :param segmap_id: only return grasps from specified segmap_id.
#     :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
#     :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
#     """
    
#     # Build the model
#     grasp_estimator = GraspEstimator(global_config)
#     grasp_estimator.build_network()

#     # Add ops to save and restore all the variables.
#     saver = tf.train.Saver(save_relative_paths=True)

#     # Create a session
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.allow_soft_placement = True
#     sess = tf.Session(config=config)

#     # Load weights
#     grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
#     os.makedirs('results', exist_ok=True)

#     # Process example test scenes
#     for p in glob.glob(input_paths):
#         print('Loading ', p)

#         pc_segments = {}
#         segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)
        
#         if segmap is None and (local_regions or filter_grasps):
#             raise ValueError('Need segmentation map to extract local regions or filter grasps')

#         if pc_full is None:
#             print('Converting depth to point cloud(s)...')
#             pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
#                                                                                     skip_border_objects=skip_border_objects, z_range=z_range)

#         print('Generating Grasps...')
#         pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
#                                                                                           local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

#         # Save results
#         np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
#                   pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

#         # Visualize results          
#         show_image(rgb, segmap)
#         visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        
#     if not glob.glob(input_paths):
#         print('No files found: ', input_paths)
        
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
#     parser.add_argument('--np_path', default='/home/hpm-mv/parent_graspnet/contact_graspnet/robot/captured_data/duster.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
#     parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
#     parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
#     parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
#     parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
#     parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
#     parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
#     parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
#     parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
#     parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
#     FLAGS = parser.parse_args()

#     global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
#     print(str(global_config))
#     print('pid: %s'%(str(os.getpid())))

#     inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
#                 K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
#                 forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)

#     filepath = pathlib.Path(__file__)

#     # read the predictions for the 'scene.npz' file
#     grasp_predictions_file = filepath.parents[1] / "results" / "predictions_current.npz"
#     grasp_predictions_dict = np.load(grasp_predictions_file, allow_pickle=True)
#     grasp_predictions_in_camera_frame = grasp_predictions_dict["pred_grasps_cam"]

#     # dict as np array, need to get object first and then select dict key for segmask
#     grasp_predictions_in_camera_frame = grasp_predictions_in_camera_frame.item()[1]

#     grasp_scores = grasp_predictions_dict["scores"]

#     # dict as np array, need to get object first and then select dict key
#     grasp_scores = grasp_scores.item()[1]

#     highest_idx = np.argmax(grasp_scores)
#     best_grasp_in_camera_frame = grasp_predictions_in_camera_frame[highest_idx]

#     print(best_grasp_in_camera_frame)
#         # Extract translation (x, y, z)
#     translation = best_grasp_in_camera_frame[:3, 3].tolist()

#     # Extract rotation matrix and convert to quaternion
#     rotation_matrix = best_grasp_in_camera_frame[:3, :3]
#     rotation = R.from_matrix(rotation_matrix)
#     quaternion = rotation.as_quat().tolist()  # [x, y, z, w]

#     # Prepare dictionary to save
#     grasp_dict = {
#         "translation": {
#             "x": translation[0],
#             "y": translation[1],
#             "z": translation[2]
#         },
#         "quaternion": {
#             "x": quaternion[0],
#             "y": quaternion[1],
#             "z": quaternion[2],
#             "w": quaternion[3]
#         }
#     }

#     # Save to JSON
#     json_path = filepath.parents[1] / "best_grasp_pose.json"
#     with open(json_path, 'w') as f:
#         json.dump(grasp_dict, f, indent=4)

#     print(f"Saved best grasp pose to {json_path}")
