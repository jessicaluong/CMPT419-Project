import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# MediaPipe holistic model to extract landmarks from live stream frames
# https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md
mp_holistic = mp.solutions.holistic 
# Draw detections to the screen 
mp_drawing = mp.solutions.drawing_utils
# Trained model to classify social signals based on landmarks 
model = load_model('python/model.keras')

def prob_viz(res, signals, input_frame):
    colors = [(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245)]
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, signals[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, left_hand, right_hand])

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

def capture_and_process_webcam(camera_number, shutdown_event, signal_queue, use_unity, dev_mode):
    '''
    Capture live stream from webcam and process frames.
    '''
    # Variables for signal recognition
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    sequence_length = 30

    # TODO: change signals array after get new model 
    signals = np.array(['raise_hand', 'thumbs_up', 'thumbs_down', 'cheer', 'cross_arms', 'clap', 'superman', 'x', 'neutral'])

    # Variables for sending signals to Unity every 3 seconds
    last_sent_time = time.time()  # Initialize the last sent time
    send_interval = 3  # seconds

    # Use the Holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Opens webcam for video capturing
        cap = cv2.VideoCapture(camera_number)

        try: 
            while cap.isOpened():
                ret, frame = cap.read()

                # Check if frame is grabbed
                if not ret:
                    print("Ignoring empty camera frame.")
                    continue

                # Detect landmarks in frame using MediaPipe
                results, image = detect_landmarks(frame, holistic)

                # Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-sequence_length:]
                
                if len(sequence) == sequence_length:
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    predictions.append(np.argmax(res))
                    
                    if not dev_mode: 
                        # Determine if it is time to send the majority signal
                        current_time = time.time()
                        if current_time - last_sent_time >= send_interval:
                            # Calculate majority signal over the last interval
                            if predictions:
                                most_common = np.bincount(predictions).argmax()
                                most_common_signal = signals[most_common]
                                confidence = np.max(np.bincount(predictions)) / len(predictions)

                                # Check if confidence exceeds threshold
                                if confidence > threshold:
                                    if use_unity:
                                        # Send signal to Unity
                                        signal_queue.put(most_common_signal)
                                    else:
                                        # Print signal for debugging purposes
                                        print(f"Signal detected: {most_common_signal}")

                            # Reset the predictions list and update the last sent time
                            predictions = []
                            last_sent_time = current_time

                    else: 
                        # Annotate frame with landmarks
                        image = draw_landmarks(image, results)

                        print(signals[np.argmax(res)])

                        # Viz logic
                        if np.unique(predictions[-10:])[0]==np.argmax(res): 
                            if res[np.argmax(res)] > threshold: 
                                
                                if len(sentence) > 0: 
                                    if signals[np.argmax(res)] != sentence[-1]:
                                        sentence.append(signals[np.argmax(res)])
                                else:
                                    sentence.append(signals[np.argmax(res)])

                        if len(sentence) > 5: 
                            sentence = sentence[-5:]

                        # Viz probabilities
                        image = prob_viz(res, signals, image)

                        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                        cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Show the frame with landmarks 
                cv2.imshow('Webcam', image)

                # Press 'q' or 'Escape' key to quit
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    shutdown_event.set()
                    break

        finally: 
            cap.release()
            cv2.destroyAllWindows()

