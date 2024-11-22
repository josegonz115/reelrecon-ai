import socketio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from yolo_working import model, warm_up_model
from yolo_utils import image_processing
from av import VideoFrame
import time

sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins=[
    'http://localhost:3000', 
    "*",
], engineio_logger=True)
app = web.Application()
sio.attach(app)

pcs = set()

warm_up_model(model)

class VideoTransformTrack(VideoStreamTrack):
    """
    Receives video frames from the client, processes them using the YOLO model,
    and sends back the processed frames at 6 FPS.
    """
    def __init__(self, track):
        super().__init__()  # Initialize the base class
        self.track = track
        self.last_processed_time = 0
        self.last_processed_frame = None  

    async def recv(self):
        frame = await self.track.recv()
        current = time.time()
        processing_interval = 1 / 3 # 3fps
        if current - self.last_processed_time >= processing_interval or self.last_processed_frame is None:
            self.last_processed_time = current
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            image_processing(results, img)
            self.last_processed_frame = img
        else:
            img = self.last_processed_frame
        new_frame = VideoFrame.from_ndarray(img, format="bgr24", )
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)

@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)

@sio.on('signal')
async def on_signal(sid, data):
    print(f"Signal received from {sid}")

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        print(f'ICE connection state is {pc.iceConnectionState}')
        if pc.iceConnectionState == 'failed':
            await pc.close()
            pcs.discard(pc)

    @pc.on('track')
    def on_track(track):
        print(f'Received track: {track.kind}')
        if track.kind == 'video':
            local_track = VideoTransformTrack(track)
            pc.addTrack(local_track)

    if 'type' in data:
        # This is an offer or answer
        desc = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
        await pc.setRemoteDescription(desc)
        if desc.type == 'offer':
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            # Send the answer back to the client
            await sio.emit('signal', {
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type
            }, to=sid)
    elif 'candidate' in data:
        # This is an ICE candidate
        candidate = data['candidate']
        await pc.addIceCandidate(candidate)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=1947)