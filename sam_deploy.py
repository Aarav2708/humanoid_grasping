import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import pyrealsense2 as rs
import json

# ---------- Step 1: Capture Frame from RealSense ----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("[INFO] Warming up RealSense camera...")
for _ in range(10):
    pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
if not color_frame:
    raise RuntimeError("Could not get frame from RealSense.")

image_bgr = np.asanyarray(color_frame.get_data())
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
pipeline.stop()

# ---------- Step 2: Load bounding box from JSON ----------
with open("bounding_box.json", "r") as f:
    bbox_data = json.load(f)

x1 = int(bbox_data["box"]["x1"])
y1 = int(bbox_data["box"]["y1"])
x2 = int(bbox_data["box"]["x2"])
y2 = int(bbox_data["box"]["y2"])

input_box = np.array([x1, y1, x2, y2])

print(f"[INFO] Loaded bounding box: {input_box.tolist()}")

# ---------- Step 3: Load SAM and perform segmentation ----------
sam_checkpoint = "sam_vit_h.pth"  # Update with your checkpoint path
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cuda")
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False
)

# ---------- Step 4: Extract and save contours as waypoints ----------
mask_uint8 = (masks[0] * 255).astype(np.uint8)
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
waypoints = [contour.squeeze().tolist() for contour in contours if contour.shape[0] >= 3]

with open("segmented_waypoints.json", "w") as f:
    json.dump(waypoints, f)

print("[INFO] Waypoints saved to segmented_waypoints.json")

# ---------- Step 5: Display the segmented result ----------
with open("segmented_waypoints.json", "r") as f:
    loaded_waypoints = json.load(f)

image_to_draw = image_rgb.copy()
for contour in loaded_waypoints:
    contour_np = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
    cv2.drawContours(image_to_draw, [contour_np], -1, (255, 0, 0), 2)

# Convert RGB to BGR for OpenCV
image_bgr = cv2.cvtColor(image_to_draw, cv2.COLOR_RGB2BGR)

cv2.imshow("Loaded Waypoints", image_bgr)
cv2.waitKey(3000)  # 5000 ms = 5 seconds
cv2.destroyAllWindows()

