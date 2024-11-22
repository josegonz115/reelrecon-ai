import time
from typing import List
import numpy as np
from ultralytics import YOLOv10, engine
import cv2
import math
# import ultralytics

model = YOLOv10("runs2/detect/train/weights/best.pt")

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

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Get the actual width and height of the frames
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {width}x{height}")

    all_fps = []

    while True:
        start_time = time.time()
        success, img = cap.read()
        results: List[engine.results.Results] = model(img, stream=True)
        image_processing(results, img)
        cv2.imshow("Webcam", img)

        end_time = time.time()  # End time
        fps = 1 / (end_time - start_time)  # Calculate FPS
        print(f"FPS: {fps:.2f}")
        all_fps.append(fps)


        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

    # calc avg fps    
    print('avg fps:', sum(all_fps) / len(all_fps))

def warm_up_model(model, num_passes=5):
    """Warm up the model by running a few dummy inferences."""
    dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(num_passes):
        model(dummy_input, stream=True)


if __name__ == "__main__":
    main()

    
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
# ]

# model = YOLO("yolo-Weights/yolov8n.pt")
# source: https://huggingface.co/jameslahm/yolov10n
# model = YOLOv10.from_pretrained('jameslahm/yolov10n')
