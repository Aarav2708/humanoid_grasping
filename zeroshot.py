import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import json
from PIL import Image
from transformers import AutoProcessor, OmDetTurboForObjectDetection

# Load model and processor
processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
model.eval()

# Start RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    # Warm up the camera
    for _ in range(30):
        pipeline.wait_for_frames()

    # Capture one frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("No frame captured from RealSense.")

    # Convert to OpenCV and PIL
    color_image = np.asanyarray(color_frame.get_data())
    image_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    # Text label to detect
    input_text = input("Enter the object to grasp: ")
    text_labels = [input_text]

    # Run inference
    inputs = processor(image_pil, text=text_labels, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=[(image_pil.height, image_pil.width)],
        text_labels=text_labels,
        threshold=0.3,
        nms_threshold=0.3,
    )

    result = results[0]
    boxes, scores, labels = result["boxes"], result["scores"], result["text_labels"]

    if boxes.numel() == 0:
        print("No objects detected.")
    else:
        # Get best detection
        max_idx = scores.argmax().item()
        box = [round(v, 2) for v in boxes[max_idx].tolist()]
        label = labels[max_idx]
        confidence = round(scores[max_idx].item(), 3)

        # Save bounding box info to JSON
        bbox_data = {
            "label": label,
            "confidence": confidence,
            "box": {
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3]
            }
        }
        with open("bounding_box.json", "w") as f:
            json.dump(bbox_data, f, indent=4)

        # Draw on image
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(color_image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show image for 3 seconds
        cv2.imshow("Best Detection", color_image)
        cv2.waitKey(3000)  # 3000 milliseconds = 3 seconds
        cv2.destroyAllWindows()

finally:
    pipeline.stop()

# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import torch
# import json
# from PIL import Image
# from transformers import AutoProcessor, GroundingDinoForObjectDetection

# # Load Grounding DINO model and processor
# model_id = "IDEA-Research/grounding-dino-tiny"
# processor = AutoProcessor.from_pretrained(model_id)
# model = GroundingDinoForObjectDetection.from_pretrained(model_id)
# model.eval()

# # Start RealSense pipeline
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)

# try:
#     # Warm up the camera
#     for _ in range(30):
#         pipeline.wait_for_frames()

#     # Capture one frame
#     frames = pipeline.wait_for_frames()
#     color_frame = frames.get_color_frame()
#     if not color_frame:
#         raise RuntimeError("No frame captured from RealSense.")

#     # Convert to OpenCV and PIL
#     color_image = np.asanyarray(color_frame.get_data())
#     image_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

#     # Text prompt
#     input_text = input("Enter the object to grasp: ")
#     text_prompt = f"{input_text}."

#     # Process input
#     inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Post-process
#     results = processor.post_process_grounded_object_detection(
#         outputs=outputs,
#         target_sizes=[(image_pil.height, image_pil.width)],
#         text_threshold=0.3,
#         box_threshold=0.3
#     )

#     result = results[0]
#     boxes, scores, labels = result["boxes"], result["scores"], result["labels"]

#     if boxes.shape[0] == 0:
#         print("No objects detected.")
#     else:
#         max_idx = scores.argmax().item()
#         box = [round(v, 2) for v in boxes[max_idx].tolist()]
#         label = labels[max_idx]
#         confidence = round(scores[max_idx].item(), 3)

#         # Save bounding box to JSON
#         bbox_data = {
#             "label": label,
#             "confidence": confidence,
#             "box": {
#                 "x1": box[0],
#                 "y1": box[1],
#                 "x2": box[2],
#                 "y2": box[3]
#             }
#         }
#         with open("bounding_box.json", "w") as f:
#             json.dump(bbox_data, f, indent=4)

#         # Draw box and label
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(color_image, f"{label} {confidence:.2f}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         # Show image
#         cv2.imshow("Grounding DINO Detection", color_image)
#         cv2.waitKey(3000)
#         cv2.destroyAllWindows()

# finally:
#     pipeline.stop()
