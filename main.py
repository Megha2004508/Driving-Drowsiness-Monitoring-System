import cv2
import mediapipe as mp
import pyttsx3
from scipy.spatial import distance
import time

# Initialize pyttsx3 for audio alerts
engine = pyttsx3.init()
# Configure voice for better reliability
voices = engine.getProperty('voices')
if voices:
    engine.setProperty('voice', voices[0].id)  # Use first available voice
    print(f"Voice initialized: {voices[0].name}")
else:
    print("Warning: No TTS voices found. Alerts may be silent.")
engine.setProperty('rate', 150)  # Slower speech

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Constants for detection thresholds (lowered slightly for testing)
YAWN_THRESHOLD = 0.8  # Adjust based on testing
YAWN_ALERT_COUNT = 2  # Number of yawns within the time window to trigger alert
TIME_WINDOW = 60      # Time window in seconds for yawning
EYE_THRESHOLD = 0.2   # Threshold for closed eyes (lowered for testing)
EYE_CLOSED_TIME_THRESHOLD = 2  # Seconds eyes need to be closed for drowsiness

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth_landmarks):
    vertical_distance = distance.euclidean(mouth_landmarks[3], mouth_landmarks[6])
    horizontal_distance = distance.euclidean(mouth_landmarks[0], mouth_landmarks[4])
    if horizontal_distance == 0:  # Avoid division by zero
        return 0
    return vertical_distance / horizontal_distance

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks):
    vertical_1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    horizontal = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    if horizontal == 0:  # Avoid division by zero
        return 0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Camera setup
cap = cv2.VideoCapture(0)

# Variables to track yawning
yawn_timestamps = []

# Variables to track drowsiness
closed_eyes_frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if unavailable
closed_eye_frame_threshold = int(fps * EYE_CLOSED_TIME_THRESHOLD)

# Alert flags to prevent spam (optional: add cooldown)
yawn_alerted = False
drowsy_alerted = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Convert the frame to RGB as Mediapipe requires
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Extract mouth landmarks (Mediapipe indices)
            mouth_landmarks = [
                (face_landmarks.landmark[i].x * frame.shape[1], 
                 face_landmarks.landmark[i].y * frame.shape[0]) 
                for i in [61, 291, 0, 13, 14, 17, 78]  # Specific points for mouth
            ]

            # Extract eye landmarks (Mediapipe indices) - Corrected labels
            left_eye_landmarks = [  # Left eye indices
                (face_landmarks.landmark[i].x * frame.shape[1], 
                 face_landmarks.landmark[i].y * frame.shape[0]) 
                for i in [33, 160, 158, 133, 153, 144]
            ]
            right_eye_landmarks = [  # Right eye indices
                (face_landmarks.landmark[i].x * frame.shape[1], 
                 face_landmarks.landmark[i].y * frame.shape[0]) 
                for i in [362, 385, 387, 263, 373, 380]
            ]

            # Calculate MAR
            mar = calculate_mar(mouth_landmarks)

            # Calculate EAR
            left_ear = calculate_ear(left_eye_landmarks)
            right_ear = calculate_ear(right_eye_landmarks)
            ear = (left_ear + right_ear) / 2

            # Display MAR and EAR on the frame
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2)

            # Detect yawning
            if mar > YAWN_THRESHOLD:
                current_time = time.time()
                yawn_timestamps.append(current_time)

                # Remove old timestamps outside the time window
                yawn_timestamps = [t for t in yawn_timestamps if current_time - t <= TIME_WINDOW]

                # Display yawning detection message
                cv2.putText(frame, "Yawning Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2)

                # Trigger alert if multiple yawns are detected within the time window
                if len(yawn_timestamps) >= YAWN_ALERT_COUNT and not yawn_alerted:
                    cv2.putText(frame, "Take a Break!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2)
                    print("Yawn alert triggered!")  # Debug
                    engine.say("Yawning detected. Please take a break!")
                    engine.runAndWait()
                    yawn_alerted = True  # Prevent spam; reset after some time if needed

            else:
                yawn_alerted = False  # Reset flag

            # Detect drowsiness
            if ear < EYE_THRESHOLD:
                closed_eyes_frame_count += 1
                if closed_eyes_frame_count >= closed_eye_frame_threshold and not drowsy_alerted:
                    cv2.putText(frame, "Drowsiness Detected!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2)
                    print("Drowsiness alert triggered!")  # Debug
                    engine.say("Drowsiness detected. Please wake up!")
                    engine.runAndWait()
                    drowsy_alerted = True  # Prevent spam
            else:
                closed_eyes_frame_count = 0
                drowsy_alerted = False  # Reset flag

    # Display the frame
    cv2.imshow("Yawning and Drowsiness Detection", frame)

    # Manual voice test: Press 'v' in the window
    key = cv2.waitKey(10) & 0xFF
    if key == ord('v'):
        print("Manual voice test triggered!")
        engine.say("Voice working! This is a manual test.")
        engine.runAndWait()
    # Exit on 'q' key press
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
