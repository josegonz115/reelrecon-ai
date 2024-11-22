import cv2
import onnxruntime as ort
import numpy as np

from yolo_utils import image_processing_onnx

ort_sesh = ort.InferenceSession(
    'runs2/detect/train/weights/best.onnx', 
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

input_name = ort_sesh.get_inputs()[0].name
input_shape = ort_sesh.get_inputs()[0].shape

"""
Expected Model Output:
    Model input name: images
    Model input shape: [1, 3, 640, 640]
    Model input type: tensor(float)
"""

def warm_up_model(num_iterations=5):
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    for _ in range(num_iterations):
        ort_sesh.run(None, {input_name: dummy_input})

def process_image(img):
    # preprocess image
    img_tensor = img.astype(np.float32)
    img_resized = cv2.resize(img_tensor, (640, 640))
    img_tensor = img_resized.transpose(2, 0, 1)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # inference
    outputs = ort_sesh.run(None, {input_name: img_tensor})
    # process outputs
    processed_img = image_processing_onnx(outputs[0], img_tensor)
    img_processed = processed_img.squeeze(0).transpose(1, 2, 0)
    # encode
    ret, img_encoded = cv2.imencode(".jpg", img_processed)
    if not ret:
        raise ValueError("Failed to encode image")
    return img_encoded.tobytes()