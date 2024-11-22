# help: https://python-socketio.readthedocs.io/en/stable/server.html
import asyncio
import socketio
from yolo_working import model, warm_up_model
from yolo_utils import image_processing
from turbojpeg import TurboJPEG, TJPF_BGR

jpeg = TurboJPEG()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=[
    'http://localhost:3000', 
    "*",
], engineio_logger=True)
app = socketio.ASGIApp(sio)  # can also wrap FastAPI and other ASGI apps by adding args

warm_up_model(model)

@sio.event
async def connect(sid, environ):
    # handle user auth here later if needed or mapping of user entities in app and sid assigned to client
    # environ has headers
    print("Client connected:", sid)


@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)


@sio.on("frame_binary")
async def handle_frame_binary(sid, data):
    # original way that works
    # np_arr = np.frombuffer(data, np.uint8)
    # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # the turbojpeg way
    img = jpeg.decode(data, pixel_format=TJPF_BGR)

    def process_image(img):
        results = model(img, stream=True)
        image_processing(results, img)
        # original way
        # ret, img_encoded = cv2.imencode(".jpg", img)
        # if not ret:
        #     raise ValueError("Failed to encode image")
        # return img_encoded.tobytes()  

        # turbo way
        img_encoded = jpeg.encode(img)
        if img_encoded is None:
            raise ValueError("Failed to encode image")
        return img_encoded
    try:
        img_bytes = await asyncio.to_thread(process_image, img)
        await sio.emit("processed_frame", img_bytes, to=sid)
    except ValueError as e:
        print(f"Error processing image: {e}")
        # await sio.emit("error", {"message": "Failed to process image"}, to=sid)

# http://0.0.0.0:1947
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1947)



# @sio.on("frame")
# async def handle_frame(sid, data):
#     # decode base64
#     img_data = base64.b64decode(data)
#     np_arr = np.frombuffer(img_data, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     # run model inference and image processing in a thread
#     def process_image(img):
#         # img = cv2.resize(img, (320, 240)) # delete maybe -- resizing could help performance ?
#         results = model(img, stream=True)
#         image_processing(results, img)
#         _, img_encoded = cv2.imencode(".jpg", img)
#         return base64.b64encode(img_encoded).decode("utf-8")

#     img_base64 = await asyncio.to_thread(process_image, img)

#     # Send the processed frame back to the frontend
#     await sio.emit("processed_frame", img_base64, to=sid)