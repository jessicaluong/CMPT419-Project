# Setup 
## Install dependencies 
pip install opencv-python  
pip install mediapipe

For certain Mac devices, may need specific MediaPipe version 0.10.9:   
pip install mediapipe==0.10.9

## Exit webcam capture 
To exit webcam capture, press 'q' or 'Escape'. 

## Devices with multiple cameras
If your device has more than one camera, specify the '--camera' flag.  
Try out different numbers (0, 1, 2, etc.) until you are able to get webcam feed.  
Run ```python main.py --camera 1``` to specify camera.

## Unity communication 
Run ```python main.py --use_unity``` to use Unity communication.  
Run ```python main.py --use_unity --dev_mode``` to use Unity communication and try out sending signals to Unity through command-line input. 

