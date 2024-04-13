# Structure of Code and Dataset

## TODO 

# Setup 
## Install dependencies 
pip install opencv-python  
pip install mediapipe

For certain Mac devices, may need specific MediaPipe version 0.10.9:   
pip install mediapipe==0.10.9

# Run Program  

## Step 1. Start the Python application 
If your device has one webcam, run 'python3 python/main.py'.  
Otherwise, you may need to specify the '--camera' flag.  
Try out different numbers (0, 1, 2, etc.) until you are able to get webcam feed.  
Run 'python3 Python/main.py --camera 1 ' to specify camera.

Please note that this starts a webcam capture. The program processes just your coordinates and not your video.  

## Step 2. Start the Unity application
Double-click the Unity application to start it.   

## Step 3. Social signal detection 
The virtual agent in Unity will now react to the social signals detected through Python.  
Our agent is able to detect: raise hand, thumbs up, thumbs down, cheering, crossing arms, and clapping. 

## Step 4. Quit applications
To quit Python webcam capture, press 'q' or 'Escape'.  
To quit Unity application, press 'Command-Q' on Mac. 

# Reflection

## TODO

# For Development Use

## Unity communication 
Run ```python3 Python/main.py --use_unity``` to use Unity communication.  
Run ```python3 Python/main.py --dev_mode``` to display recognized signal to screen and print to terminal. 