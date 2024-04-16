# Table of Contents
1. [System Overview](#overview)
2. [System Compatibility](#compatibility)
3. [Project Setup](#setup)
4. [Self-Evaluation of the Project](#evaluation)
5. [For Development Use](#dev)

<a name="overview"></a>
# System Overview

### Python Components
Python is utilized as the backend of our system, responsible for:
- **Webcam Data Acquisition**: Directly captures video data from the webcam. While this involves interaction with hardware, it serves as the initial data input step for our backend processing.
- **Feature Extraction and Signal Classification**: Processes the captured video to detect and classify social signals using both pretrained and custom-trained machine learning models.
- **TCP Client**: Handles communication by sending processed data to the Unity application.

### Unity Components
Unity functions as the frontend, focusing on visualization and interaction:
- **TCP Server**: Receives data from Python and integrates it into the application.
- **Virtual Agent Display**: Visualizes responses in real-time, providing an interactive experience by reacting to the classified signals from the backend.

### Interaction Flow
The integration between Python and Unity allows Python to handle the heavy lifting of data processing (backend), while Unity focuses on user-facing elements (frontend), making the system efficient and responsive.

## Structure of Code and Dataset <a name="structure"></a>

### Python Folder Structure

The Python component of our project handles the backend operations including video capture, data processing, and server management. Here's a breakdown of the files:

- live_stream_processing.py:
  - Purpose: Captures video frames from a specified webcam and processes them to detect and interpret social signals, sending recognized signals to Unity.
  - Functionality: Utilizes the MediaPipe Holistic model to continuously capture frames, detect landmarks within those frames, and analyze these landmarks to recognize predefined social signals based on keypoints.

- python_server.py:
  - Purpose: Manages the backend server that listens for incoming client connections on a specified port.
  - Functionality: Initializes a TCP server that facilitates communication between the Python backend and the Unity frontend, ensuring that recognized social signals are effectively transmitted to the Unity application.

- main.py:
  - Purpose: Serves as the entry point for the Python application, setting up and coordinating several components. 
  - Functionality: Handles the setup of threading events and queues, initiates server communication if interfacing with Unity is enabled, and starts the webcam processing. 

### Unity Folder Structure 
Our Unity project is organized into several subfolders within the 'Assets' directory: 'Jammo-Character', 'SimpleNaturePack', and 'Scripts'. The files in these folders collectively support the interactive features of our virtual agent, ensuring it can dynamically respond to the analyzed social signals processed by our Python backend. Key files are detailed below. 

#### Jammo-Character
This folder contains the virtual agent used in our project, sourced from the Unity Asset Store (https://assetstore.unity.com/packages/3d/characters/jammo-character-mix-and-jam-158456) and modified to meet our project's specific needs. Key components include:

- Animations Folder:
  - **Various Animations**: Animations such as pointing, nodding, shaking head, excitement, shrugging, clapping, and idle are integrated. These were sourced from Mixamo and customized to fit the virtual agent's interaction needs.
  - **Jammo_Mixamo.controller**: Manages the transitions and triggers between animations. Each animation is activated through a trigger parameter, which is set off by a corresponding string message received from the Python machine learning model backend.

#### SimpleNaturePack
Used primarily as the scenic backdrop for the virtual agent's environment. It was obtained through the Unity Asset Store (https://assetstore.unity.com/packages/3d/environments/landscapes/low-poly-simple-nature-pack-162153).

- Scenes Folder:
  - **SimpleNaturePack_Demo.unity**: This scene has been customized through the Unity UI to set up and display the virtual environment appropriately for our application.

#### Scripts 
Contains essential scripts that drive the virtual agent's behaviors and the application's network communication capabilities:

- **RespondToSignals.cs**: This script handles incoming signals from the Python backend. It changes the virtual agent’s animations and eye states in response to these signals, allowing dynamic interactions based on real-time data.
- **UnityClient.cs**: Functions as the TCP client within Unity. It establishes and maintains a network connection with the Python server, processing incoming signals for the agent.
- **UnityMainThreadDispatcher.cs**: A utility script that facilitates safe calls to Unity’s main thread from asynchronous processes or secondary threads. This is crucial for ensuring smooth and responsive updates to the UI and the agent's animations in real time.

### Jupyter Folder Structure  

Our Jupyter notebooks document the iterative development and testing of our machine learning models, showcasing different approaches and their outcomes. We explored three main approaches and our final model was created through 'ADCreateModel.ipynb'. Below are the key notebooks:

- **SequentialImageDetection.ipynb**:
  - **Approach**: Initial model testing using a convolution-based neural network.
  - **Details**: Aimed to label each frame of input video, the model struggled with significant variability and unsuitability of the training data for this type of application.
  - **Performance**: Did not yield satisfactory results, leading to a pivot towards more specialized models and data preprocessing strategies.

- **ADCreateModelOld.ipynb**:
  - **Approach**: Second iteration using a more structured data collection approach.
  - **Details**: This approach captured 30 frames per sequence for 30 sequences per label, using LSTM and dense layers for the model architecture. Data was primarily collected from one group member, with variations in signal execution to address initial overfitting issues.
  - **Performance**: Achieved adequate results, indicating a need for further data diversification and model tuning.

- **ADCreateModel.ipynb**:
  - **Approach**: Third iteration of our model development.
  - **Details**: Expanded our dataset to include all group members and introduced variations in signal execution to combat overfitting.
  - **Performance**: Showed promising results when tested on group members, though generalization to the external test set highlighted areas for potential improvement.

### Data Folder Structure 
TODO: 

<a name="compatibility"></a>
# System Compatibility

## Tested Environments 

This project has been developed and thoroughly tested under the following configuration:

- **Operating System**: macOS
  - **Version**: macOS Sonoma 14.2.1
  - **Hardware**: iMac (Retina 5K, 27-inch, 2020)
  - **Processor**: 3.8 GHz Quad-Core Intel Core i7
  - **Memory**: 32 GB 2667 MHz DDR4
  - **Graphics**: AMD Radeon Pro 5700 XT 16 GB

## Notes on Compatibility

- **macOS**: The project runs smoothly without any known issues on macOS systems similar to the tested configuration.
- **Windows**: Some team members have reported issues when running the project on various Windows configurations. We currently do not have a fix for this problem.

<a name="setup"></a>
# Project Setup 

## Prerequisites

Ensure you have the following software installed to run the project successfully:

- Python 3.11.4

For those looking to build or modify the Unity application:
- Unity: Download and install Unity fron Unity's download page (https://unity.com/download).
- Unity account: You will need to create a Unity account if you don't already have one. 

## Python Setup

### Install Rquired Packages 

Before running the Python components of the project, you'll need to install necessary libraries:  
```pip install -r requirements.txt```

## Unity Setup

### Step 1: Open the Project

1. Launch Unity and open the project directory. 

### Step 2: Setup Scripts and Components 

1. Configure the Scene: 
    - Navigate to `Project > Assets > SimpleNaturePack > Scenes` and locate the scene named `SimpleNaturePack_Demo`.
    - Drag `SimpleNaturePack_Demo.unity` into the Hierarchy.
    - Remove any default scenes from the Hierarchy to avoid conflicts.

2. Configure Components: 
    - Select `Jammo` in the Hierarchy. 
    - In the Inspector, ensure that the Respond to Signals (Script), Use Tcp Connection, and Unity Client (Script) are checked. 

### Step 3: Running the Project within Unity 

1. Running Within Unity: 
    - Simply press the 'Play' button at the top-middle of the Unity interface to start the application in development mode. This action initiates the client side of the TCP connection. Press the button again to stop.

2. Building the Application: 
    - Navigate to `File > Build Settings`.
    - Ensure the `Target Platform `and Archit`ecture are correctly set for your operating system.
    - Click `Build and Run` to compile and execute the application. Close the built application after confirming it launches correctly.

**Note**: The Python backend server must be running before you start the Unity application to ensure proper communication between the TCP client (Unity) and the TCP server (Python).

<a name="run"></a>
# Running the Project 

### Step 1: Start the Python Application
Ensure the Python backend is ready to handle data:
- Run the following command in your terminal: `python3 python/main.py`
- If you have multiple webcams and need to specify one, use the `--camera` flag. Example: `python3 Python/main.py --camera 1`

**Note**: This starts the webcam capture. The program processes your movements but does not record video.

### Step 2: Start the Unity Application
- Double-click the built Unity application to start it.

### Step 3: Social Signal Detection
- The virtual agent in Unity will now react to social signals detected through Python. The agent can detect gestures such as raising a hand, thumbs up, thumbs down, cheering, crossing arms, and clapping.

### Step 4: Quitting the Applications
- To stop the Python webcam capture, press 'q' or 'Escape'.
- To close the Unity application, press 'Command-Q' on Mac or use the standard window close button if the application is built in Windowed mode.

# Self-Evaluation of the Project <a name="evaluation"></a>

## Reflection on the Proposal Objectives

Our initial project proposal proposed a Unity-based virtual agent that could recognize and respond to a variety of classroom gestures, utilizing a Python machine learning model to process these gestures in real-time. Our motivation was to enhance student engagement and provide a supportive tool for educators, through advanced object and action detection techniques. 

### Achievements

- Real-Time Gesture Recognition: We successfully implemented a system where our virtual agent, developed in Unity, interacts dynamically with gestures identified through a Python-based machine learning model. This integration allows for seamless recognition and response to gestures, facilitating a more interactive online classroom experience.
- Application in Virtual Learning: We refined our initial concept to better suit the current educational landscape by adapting our virtual agent for online classrooms. This adjustment meant designing the agent to function effectively within a Zoom setting, with each student potentially having a personalized agent.

## Changes to the Approach

Initially, our project proposal aimed to explore a wide range of human action detection techniques, categorizing them into three groups: detecting objects within each frame using image detectors such as AlexNet, YOLO, and SSD; extracting key frames for video detection with SVMs; and leveraging temporal data between consecutive frames using models like GoogLeNet, ResNet-101, T-CNN, and TCN.

However, as the project progressed, we identified that the MediaPipe framework offered a more efficient solution for real-time landmark detection, which was crucial for our needs in processing live video streams. MediaPipe's ability to provide rapid and accurate landmark detection allowed us to capture the dynamics of human gestures more effectively than the static or frame-based detection methods initially considered.

Given the real-time nature of our application, we changed our approach to use LSTM and Dense layers, replacing the planned diverse set of detection models. LSTMs, being adept at handling time-series data, allowed us to analyze sequences of gesture-related data effectively, capturing temporal relationships that are vital for accurate gesture recognition. This method aligns better with our project’s requirement for real-time processing and continuous gesture recognition.

The Dense layers following the LSTM architecture serve to interpret the features extracted over time, classifying them into our predefined gesture categories. This approach not only simplified our model architecture but also enhanced its performance by ensuring that the gesture recognition process is both efficient and scalable, adapting seamlessly to the live interaction demands of our system.

By focusing on LSTM and Dense layers, fed by the landmarks detected by MediaPipe, we were able to achieve an effective solution for real-time gesture recognition, which enchances the responsiveness and accuracy of our virtual agent.

## Changes to Data Collection

Originally, we planned to use existing datasets, such as EduNet and SCB-dataset. However, as our project progressed, we encountered limitations with these datasets in terms of their applicability to our specific requirements. The pre-existing datasets did not offer the precise control or specificity needed for effective training of our model.

To address these challenges, we opted to create our own dataset from scratch, using live streamed videos of our team members. This approach allowed us to tailor the data collection process to fit exactly the needs of our project, ensuring that each gesture was captured in a controlled environment with consistent camera positioning.

The new dataset was designed to include:

- Single gesture per sequence: Ensuring a clear label for each data entry.
- Single person per sequence: This simplifies the model's task by removing unnecessary complexity from other individuals.
- Controlled variations in gestures: For example, the 'crossing arms' gesture was recorded in multiple variations, including with no hands showing, both hands showing, only the right hand showing, and only the left hand showing.

Creating our dataset in this manner provided several benefits:

- Increased relevance and precision: The data directly corresponds to the gestures our system needs to recognize.
- Enhanced adaptability and performance: With data that closely matches the use case (i.e., single student on Zoom), our model can learn more effectively and perform more reliably in real-world scenarios.
- Ability to capture subtle variations: This is crucial for robust gesture recognition, as it helps the model generalize better across different instances of the same gesture.

This customized approach to data collection not only improved the quality and relevance of our training data but also improved the overall performance of our gesture recognition system.

## Changes to Evaluation

### Recognition Testing:

We evaluated our model by testing through the Jupyter notebooks, using confusion matrices and performance metrics to assess the accuracy of gesture recognition. 

### Interaction Testing:

We did not conduct a formal human study. Instead, we conducted live interaction sessions with friends and family, who interacted with the system in real-time. This evaluation helped us gauge the system's responsiveness and gather user feedback. We were also able to further assess the model's recognition capabilities this way. 

<a name="dev"></a>
# For Development Use 

## Unity communication 
To run in Development mode: ```python3 Python/main.py --dev_mode```. This does not start the a Unity server. Instead, it prints the recognized signal to the terminal (every 3 seconds if confidence is over 50%), and also draws landmarks and probability of each action to the webcam display.
