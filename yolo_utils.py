from typing import List
import cv2
import math
from ultralytics import engine
import numpy as np

classNames = [
    "Bangus",
    "Big head carp",
    "Black spotted Barb",
    "Clownfish",
    "Gold fish",
    "Gourami",
    "Knife fish",
    "Mackerel",
    "Orchid Dottyback",
    "Pangas",
    "Pomfrets",
    "Rainbowfish",
    "Red-Tilapia",
    "Tuna",
    "Yellow Tang",
    "Zebrafish",
    "cat-fish",
    "fish-Mullet",
    "fish-Perch",
    "fish_Goby",
    "fish_Mosquito Fish",
    "fish_Mudfish",
    "puffer",
    "snake head",
]

def image_processing(results: List[engine.results.Results], img: np.ndarray):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            )  # convert to int values
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)
            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])
            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(
                img, classNames[cls], org, font, fontScale, color, thickness
            )
            
def image_processing_onnx(outputs: np.ndarray, img: np.ndarray):
    # Assuming outputs is a numpy array with shape [1, num_detections, 6]
    # where each detection is [x1, y1, x2, y2, score, class]
    detections = outputs[0]
    
    boxes = detections[:, :4]  # Extract bounding box coordinates
    scores = detections[:, 4]  # Extract confidence scores
    classes = detections[:, 5]  # Extract class labels
    
    # Process each detection
    for box, score, cls in zip(boxes, scores, classes):
        if score > 0.5:  # Filter out low-confidence detections
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil(score * 100) / 100
            print("Confidence --->", confidence)
            cls = int(cls)
            print("Class name -->", classNames[cls])
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
    
    return img