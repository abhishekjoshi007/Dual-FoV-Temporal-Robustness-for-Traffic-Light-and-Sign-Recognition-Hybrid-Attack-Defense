import os
import json
from ultralytics import YOLO
from datetime import datetime

# Directories
image_dir = r"C:\Users\rohan\OneDrive\Desktop\Genai\AVML\udacity\images"
light_dir = r"C:\Users\rohan\OneDrive\Desktop\Genai\AVML\udacity\tarfficlight"

os.makedirs(light_dir, exist_ok=True)

# Load traffic light YOLO model
light_model = YOLO("best_traffic_med_yolo_v8.pt")

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

for img_file in os.listdir(image_dir):
    if not img_file.lower().endswith(('.jpg','.png','.jpeg')):
        continue
    img_path = os.path.join(image_dir, img_file)
    frame_id = int(datetime.now().timestamp())
    obj_id = 0

    # Traffic Light detection only
    light_objs = []
    for res in light_model(img_path):
        for box in res.boxes:
            cls_name = light_model.names[int(box.cls[0])]  # e.g., "red", "green", "yellow"
            x1,y1,x2,y2 = map(float, box.xyxy[0])
            light_objs.append({
                "ActorName": f"TRAFFIC_LIGHT {obj_id}",
                "ObjectId": obj_id,
                "BoundingBox3D Origin X": (x1+x2)/2,
                "BoundingBox3D Origin Y": (y1+y2)/2,
                "BoundingBox3D Origin Z": 0.0,
                "BoundingBox3D Extent X": (x2-x1)/2,
                "BoundingBox3D Extent Y": (y2-y1)/2,
                "BoundingBox3D Extent Z": 0.0,
                "BoundingBox3D Orientation Quat W": 1.0,
                "BoundingBox3D Orientation Quat X": 0.0,
                "BoundingBox3D Orientation Quat Y": 0.0,
                "BoundingBox3D Orientation Quat Z": 0.0,
                "ObjectType": "TRAFFIC_LIGHT",
                "Truncated": 0,
                "ObjectMeta": {
                    "SourceID": obj_id,
                    "SubType": cls_name,
                    "Text": "",
                    "OcrVersion": 10006
                }
            })
            obj_id += 1

    # Save JSON files (only traffic light)
    with open(os.path.join(light_dir, os.path.splitext(img_file)[0] + ".json"), "w") as f:
        json.dump(create_json(light_objs, frame_id, 10306), f, indent=2)

print("Done with traffic light detection and JSON generation!")
