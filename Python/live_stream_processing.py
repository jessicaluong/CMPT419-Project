# Adapted from https://github.com/nicknochnack/Body-Language-Decoder/blob/main/Body%20Language%20Decoder%20Tutorial.ipynb
import mediapipe as mp
import cv2

# Use MediaPipe holistic model (https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md)
mp_holistic = mp.solutions.holistic 
# Draw detections to the screen (used for development)  
mp_drawing = mp.solutions.drawing_utils

def detect_landmarks(frame, holistic):
    """
    Process the image with MediaPipe Holistic to detect landmarks.
    Returns the results which contain landmarks and image in RGB. 
    """
    image_rgb.flags.writeable = False   
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect landmarks
    results = holistic.process(image_rgb)

    return results, image_rgb
    
def display_landmarks(image_rgb, results):
    """
    Draw the detected landmarks on the image.
    Returns image in BGR. 
    (Used for development - may be removed in the future.)
    """
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 1. Draw face landmarks
    mp_drawing.draw_landmarks(image_bgr, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                )
    
    # 2. Right hand
    mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                )

    # 3. Left Hand
    mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                )

    # 4. Pose Detections
    mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )
    
    return image_bgr

def capture_and_process_webcam(camera_number, shutdown_event, signal_queue, use_unity):
    '''
    Capture live stream from webcam and process frames.
    '''
    # Use the Holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Opens webcam for video capturing
        cap = cv2.VideoCapture(camera_number)

        # Flag to ensure the example signal is processed only once (for development purposes)
        signal_sent = False

        try: 
            while cap.isOpened():
                ret, frame = cap.read()

                # Check if frame is grabbed
                if not ret:
                    print("Ignoring empty camera frame.")
                    continue

                # Detect landmarks
                results, image = detect_landmarks(frame, holistic)

                # Display webcam and landmarks to the screen (used for development)

                # Display landmarks
                display_frame = display_landmarks(image, results)

                # Show the frame with landmarks
                cv2.imshow('Webcam with Landmarks', display_frame)

                ###################################################################

                # TODO: implement signal recognition 
                # Landmarks from "results" that may be useful: pose_landmarks, left_hand_landmarks, and right_hand_landmarks.  
                # See https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md for details. 
                # See https://youtu.be/We1uB79Ci-w?si=_4hgLIfUbnqioD1S&t=1165 for ideas on training/testing.
                if not signal_sent:
                    example_detected_signal = "bored"
                    if use_unity:
                        # If using Unity, add signal to queue and send to Unity 
                        signal_queue.put(example_detected_signal)  
                    else:
                        # If not using Unity, print signal 
                        print(f"Example detected signal (not sent to Unity): {example_detected_signal}")
                    
                    signal_sent = True  # Process one signal only 

                ###################################################################


                # Press 'q' or 'Escape' key to quit
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    shutdown_event.set()
                    break

        finally: 
            cap.release()
            cv2.destroyAllWindows()

