import cv2
import numpy as np
import os
import mediapipe as mp

"""
Utility file for variables and functions used in data and model creation.
"""

# VARIABLES YOU MIGHT WANT TO CHANGE
#--------------------------------

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('../data/AD_data')

# Actions that we try to detect
actions = np.array(['raise_hand', 'thumbs_up', 'thumbs_down', 'cheer', 'cross_arms', 'clap', 'neutral'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

#---------------------------------



pose = np.zeros(132)
face = np.zeros(1404)
lh = np.zeros(21*3)
rh = np.zeros(21*3)

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    """
    Processes an image to detect and extract landmarks using the MediaPipe Holistic model.

    Parameters:
        image: The image in which landmarks are to be detected.
        holistic: The MediaPipe Holistic model object.

    Returns:
        results : An object containing detected landmarks such as pose, face, and hand landmarks.
        image: The processed image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    """
    Draw the detected landmarks on the image.
    This function is used in development mode.
    Adapted from https://github.com/nicknochnack/Body-Language-Decoder/blob/main/Body%20Language%20Decoder%20Tutorial.ipynb
    """
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    """
    Draw the detected landmarks on the image STYLISHLY.
    This function is used in development mode.
    Adapted from https://github.com/nicknochnack/Body-Language-Decoder/blob/main/Body%20Language%20Decoder%20Tutorial.ipynb
    """
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    """
    Extracts keypoints from the pose, left hand, and right hand landmarks from the provided results.

    This function processes pose and hand landmarks detected in an image or video frame, converting them into
    a flat numpy array. If landmarks for a specific part (pose, left hand, right hand) are not found,
    it fills the respective part of the array with zeros.

    Parameters:
        results: A result object from MediaPipe Holistic containing various landmark detections.

    Returns:
        numpy.ndarray: A 1D array containing all the extracted keypoints for pose, left hand, and right hand.
                       Each keypoint consists of its x, y, z coordinates, and a visibility score (for pose only).
                       If no keypoints are detected for a specific part, that segment of the array is filled with zeros.
                       The pose segment is 132 elements long (33 keypoints * 4 attributes each), and each hand
                       segment is 63 elements long (21 keypoints * 3 attributes each).
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])