import os
import cv2
import torch
import time
from PIL import Image
from transformers import AutoProcessor
from omlab.models import OmDetTurboForObjectDetection

# Suppress warnings and logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def detect_objects_live(model_id, object_text, camera_index=0):
    """
    Detect objects described by input text from a webcam stream continuously.
    Display live bounding boxes and centroids for the specified object.

    Args:
        model_id: The model ID for OmDet Turbo.
        object_text: Description of the object to detect (e.g., "coffee cup").
        camera_index: The index of the webcam to use.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = OmDetTurboForObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Error: Unable to access the webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Looking for: {object_text} (Press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        inputs = processor(images=image, text=[object_text], return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        processing_time = time.time() - start_time

        original_height, original_width = frame.shape[:2]
        results = processor.post_process_grounded_object_detection(
            outputs,
            classes=[object_text],
            target_sizes=[(original_height, original_width)],
            score_threshold=0.4,
            nms_threshold=0.3
        )[0]

        for score, class_name, box in zip(results["scores"], results["classes"], results["boxes"]):
            x_min, y_min, x_max, y_max = map(int, box[:4])
            centroid_x = (x_min + x_max) // 2
            centroid_y = (y_min + y_max) // 2

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{class_name} ({score:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Centroid: ({centroid_x},{centroid_y})", (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("OmDet Turbo - Live Detection", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    try:
        object_input = input("Enter object to detect (e.g., 'coffee cup'): ").strip()
        model_id = "omlab/omdet-turbo-swin-tiny-hf"
        detect_objects_live(model_id, object_input)
    except Exception as e:
        print(str(e))
