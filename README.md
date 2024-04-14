# Setting Up the Project Environment

To run this project, you may want to set up a Python virtual environment and install the required packages.

## Prerequisites
- Python 3.11.4

If you would like to build the Unity application yourself:
- Unity (https://unity.com/download)
- Unity Account

## Python Setup

Install the required packages: ```pip install -r requirements.txt```
## Unity Setup for Building Application 

1. Open the Unity Project.

2. Find the Scene under Project/Asset/SimpleNaturePack/Scenes/SimpleNaturePack_Demo.unity. Drag SimpleNaturePack_Demo to Hierarchy. Please delete the default scene under Hierarchy if you plan to build the application. 

3. Under Hierarchy, press Jammo. Under Inspector, ensure the checkboxes of Respond to Signals (Script), Use Tcp Connection, and Unity Client (Script) are checked. 

    a. To run without building the application, press the 'Play' button found in the top-middle of the screen. To stop, press this button again. 

    b. To build application, go to File > Build Settings. Check that the Target Platform and Architecture matches your machine. Press Build and Run. 

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

# Structure of Code and Dataset

Our project is organized into three folders: Python, Unity, and Jupyter. 

## Python folder



## Unity folder


## Jupyter folder 


# Reflection

## TODO

# For Development Use

## Unity communication 
Run ```python3 Python/main.py --use_unity``` to use Unity communication.  
Run ```python3 Python/main.py --dev_mode``` to display recognized signal to screen and print to terminal. 