import time
from typing import List
import numpy as np
from ultralytics import YOLOv10, engine
import cv2
import torch
from yolo_utils import image_processing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLOv10("runs2/detect/train/weights/best.pt")
model.to(device)

"""
WARNING ⚠️ torch.Tensor inputs should be normalized 0.0-1.0 but max value is 4.806208610534668. Dividing input by 255.
0: 640x640 (no detections), 12.0ms
Speed: 0.1ms preprocess, 12.0ms inference, 65.3ms postprocess per image at shape (1, 3, 640, 640)
"""

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
