import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Use MediaPipe holistic model (https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md)
mp_holistic = mp.solutions.holistic 
# Draw detections to the screen 
mp_drawing = mp.solutions.drawing_utils
# Trained model
model = load_model('python/model.keras')

def prob_viz(res, actions, input_frame):
    colors = [(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245)]
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def extract_keypoints(results):
    pose = np.zeros(132)
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

def detect_landmarks(image, holistic):
    """
    Process the image using MediaPipe Holistic to extract landmarks.
    Returns the results which contain the landmarks and image. 
    """
    image.flags.writeable = False   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return results, image
    
def draw_landmarks(image, results):
    """
    Draw the detected landmarks on the image.
    Adapted from https://github.com/nicknochnack/Body-Language-Decoder/blob/main/Body%20Language%20Decoder%20Tutorial.ipynb
    """

    # Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                )

    # Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                )

    # Pose 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )
    
    return image

def capture_and_process_webcam(camera_number, shutdown_event, signal_queue, use_unity):
    '''
    Capture live stream from webcam and process frames.
    '''
    # Variables 
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    actions = np.array(['raise_hand', 'thumbs_up', 'thumbs_down', 'cheer', 'cross_arms', 'clap', 'superman', 'x', 'neutral'])

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

                # Detect landmarks in frame using MediaPipe
                results, image = detect_landmarks(frame, holistic)

                # Annotate frame with landmarks
                draw_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    # image = prob_viz(res, actions, image)

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

                # Show the frame with landmarks
                cv2.imshow('Webcam with Landmarks', image)

                # Press 'q' or 'Escape' key to quit
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    shutdown_event.set()
                    break

        finally: 
            cap.release()
            cv2.destroyAllWindows()

