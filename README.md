# reelrecon-ai


## Plan
1. Use object detection to crop fish
2. Maintain aspect ratio during cropping
3. Add 10-15% padding around fish
4. Standardize image sizes
5. Then train classifier on cropped images

### other requirements
- to install turbojpeg
```
brew install jpeg-turbo
```


### Notes on Webrtc
	•	Signaling: Mechanism to exchange connection information (SDP and ICE candidates) between peers.
	•	Media Streams: Audio/Video data transmitted over the connection.
	•	STUN/TURN Servers: Assist with NAT traversal.