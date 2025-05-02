import json
import torch
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import cv2
from transformers import AutoProcessor, GroundingDinoForObjectDetection

# --- Load action.json ---
with open("/home/hpm-mv/Hardik/FINAL_TRIAL/action_data/action.json", "r") as f:
    tasks = json.load(f)
objects = list(set([task["object"] for task in tasks]))
prompt = ". ".join(objects) + "."

# --- Load model and processor ---
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = GroundingDinoForObjectDetection.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# --- Capture frame from RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("[INFO] Capturing RealSense frame...")
for _ in range(30):  # warm-up
    pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())
pipeline.stop()

# --- Convert to PIL image ---
image_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

# --- Run zero-shot detection ---
inputs = processor(images=image_pil, text=prompt, return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

# --- Post-process ---
target_sizes = torch.tensor([image_pil.size[::-1]], device=model.device)
results = processor.post_process_grounded_object_detection(
    outputs, target_sizes=target_sizes, box_threshold=0.3, text_threshold=0.25
)[0]

# --- Format output ---
output_data = {}
for score, label_text, box in zip(results["scores"], results["text_labels"], results["boxes"]):
    for obj in objects:
        if obj.lower() in label_text.lower():
            output_data[" ".join(obj.split())] = {
                "label": obj,
                "confidence": round(score.item(), 3),
                "box": {
                    "x1": round(box[0].item(), 2),
                    "y1": round(box[1].item(), 2),
                    "x2": round(box[2].item(), 2),
                    "y2": round(box[3].item(), 2)
                }
            }
            # --- Draw boxes + labels ---
            b = box.int().cpu().numpy()
            label_str = f"{obj}: {score.item():.2f}"
            cv2.rectangle(color_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(color_image, label_str, (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# --- Save JSON ---
with open("bounding_boxes.json", "w") as f:
    json.dump(output_data, f, indent=4)
print("[INFO] Saved bounding boxes to bounding_boxes.json")

# --- Show image for 3 seconds only ---
output_path = "/home/hpm-mv/parent_graspnet/humanoid_grasping/detected_objects.png"
cv2.imwrite(output_path, color_image)
print(f"[INFO] Segmented waypoints saved to: {output_path}")
