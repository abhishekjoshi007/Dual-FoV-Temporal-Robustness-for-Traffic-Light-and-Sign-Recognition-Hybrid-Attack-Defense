import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

# Paths
image_dir = r"C:\Users\rohan\OneDrive\Desktop\Genai\AVML\udacity\images"
output_dir = r"C:\Users\rohan\OneDrive\Desktop\Genai\AVML\udacity\trafficsign"
model_path = "yolov8n.pt"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv8 model
model = YOLO(model_path)

def create_json(objects, frame_id, version):
    return {
        "root": {
            "CapturedObjects": objects if objects else 0,
            "FrameId": frame_id,
            "Timestamp": 0,
            "TimestampMiddle": 0,
            "VERSION": version
        }
    }

def detect_traffic_light_color(image, x1, y1, x2, y2):
    """Detect traffic light color: RED, YELLOW, GREEN."""
    crop = image[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # HSV color ranges for red (two ranges because red wraps HSV)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    green_lower = np.array([40, 50, 50])
    green_upper = np.array([90, 255, 255])

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = mask_red1 + mask_red2
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    red_count = cv2.countNonZero(mask_red)
    yellow_count = cv2.countNonZero(mask_yellow)
    green_count = cv2.countNonZero(mask_green)

    max_count = max(red_count, yellow_count, green_count)
    if max_count == 0:
        return "UNKNOWN"
    elif max_count == red_count:
        return "RED"
    elif max_count == yellow_count:
        return "YELLOW"
    else:
        return "GREEN"

# Process each image
for frame_id, img_file in enumerate(os.listdir(image_dir), start=1):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image {img_path}")
        continue

    results = model(img)
    detected_objs = []
    obj_id = 1

    for res in results:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)

            # Traffic light color detection
            if cls_name.upper() == "TRAFFIC LIGHT":
                light_color = detect_traffic_light_color(img, x1, y1, x2, y2)
            else:
                light_color = None

            # Draw bounding box and label on image
            label = f"{cls_name.upper()}"
            if light_color:
                label += f" {light_color}"

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            obj_entry = {
                "ActorName": f"{cls_name.upper()} {obj_id}",
                "ObjectId": obj_id,
                "BoundingBox3D Origin X": (x1 + x2) / 2,
                "BoundingBox3D Origin Y": (y1 + y2) / 2,
                "BoundingBox3D Origin Z": 0.0,
                "BoundingBox3D Extent X": (x2 - x1) / 2,
                "BoundingBox3D Extent Y": (y2 - y1) / 2,
                "BoundingBox3D Extent Z": 0.0,
                "BoundingBox3D Orientation Quat W": 1.0,
                "BoundingBox3D Orientation Quat X": 0.0,
                "BoundingBox3D Orientation Quat Y": 0.0,
                "BoundingBox3D Orientation Quat Z": 0.0,
                "ObjectType": cls_name.upper(),
                "Truncated": 0,
                "ObjectMeta": {
                    "SourceID": obj_id,
                    "SubType": cls_name,
                    "Text": "",
                    "OcrVersion": 10006
                }
            }

            if light_color:
                obj_entry["LightColor"] = light_color

            detected_objs.append(obj_entry)
            obj_id += 1

    # Save JSON
    json_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + ".json")
    with open(json_path, "w") as f:
        json.dump(create_json(detected_objs, frame_id, 10306), f, indent=2)

    # Save image with bounding boxes
    out_img_path = os.path.join(output_dir, img_file)
    cv2.imwrite(out_img_path, img)

    print(f"Processed {img_file}: detected {len(detected_objs)} objects")

print("All done!")
