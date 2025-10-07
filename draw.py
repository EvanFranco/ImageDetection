# ==============================
# AIR DRAWING USING HAND TRACKING
# ==============================
# Before running, install the following packages in your terminal:
# pip install opencv-python mediapipe numpy
# 
# This script uses your webcam and Mediapipe's Hand Tracking to:
#  - Detect your hand and its landmarks
#  - Allow you to draw in the air with your index finger
#  - Stop drawing when you close your fist
#  - Change drawing color when two fingers are together
# Press 'q' to quit the program

import cv2
import mediapipe as mp
import numpy as np

# ------------------------------
# Setup Mediapipe for hand tracking
# ------------------------------
mp_hands = mp.solutions.hands                    # Load hand tracking model
mp_draw = mp.solutions.drawing_utils             # Utility to draw landmarks and connections

# Initialize the Mediapipe Hands model with confidence thresholds
hands = mp_hands.Hands(
    min_detection_confidence=0.7,                # How sure Mediapipe must be to detect a hand
    min_tracking_confidence=0.7                  # How sure Mediapipe must be to track it
)

# ------------------------------
# Function: Detect if the hand is closed into a fist
# ------------------------------
def is_fist_closed(landmarks, w, h):
    """Check if most fingertips are close to the wrist, indicating a closed fist"""

    # Retrieve fingertip landmarks
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]
    ring_tip = landmarks.landmark[16]
    pinky_tip = landmarks.landmark[20]
    
    # Wrist landmark (used as a reference point)
    wrist = landmarks.landmark[0]
    
    # Calculate Euclidean distances (normalized coordinates) from fingertips to wrist
    thumb_dist = np.sqrt((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
    index_dist = np.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2)
    middle_dist = np.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
    ring_dist = np.sqrt((ring_tip.x - wrist.x)**2 + (ring_tip.y - wrist.y)**2)
    pinky_dist = np.sqrt((pinky_tip.x - wrist.x)**2 + (pinky_tip.y - wrist.y)**2)
    
    # Count how many fingers are close to the wrist
    close_fingers = sum([dist < 0.15 for dist in [thumb_dist, index_dist, middle_dist, ring_dist, pinky_dist]])
    
    # If 3 or more fingers are close, we assume the hand is closed (fist)
    return close_fingers >= 3

# ------------------------------
# Function: Check if two fingers (index + middle) are together
# ------------------------------
def are_two_fingers_together(landmarks, w, h):
    """Check if index and middle finger tips are close together"""

    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]
    
    # Distance between the two fingertips
    distance = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
    
    # Return True if they are very close together
    return distance < 0.05

# ------------------------------
# Try to automatically find a working webcam
# ------------------------------
cap = None
for i in range(5):  # Try indices 0–4 (in case your system has multiple cameras)
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret:
            cap = test_cap
            print(f"Camera found at index {i}")
            break
        else:
            test_cap.release()
    else:
        test_cap.release()

# If no camera found, give user instructions
if cap is None:
    print("No camera found! Please check:")
    print("1. Camera is connected")
    print("2. Camera is not being used by another application")
    print("3. Camera permissions are granted")
    print("4. Try running as administrator")
    exit()

# ------------------------------
# Configure camera properties
# ------------------------------
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Camera initialized successfully!")
print("Press 'q' to quit")

# Canvas (blank image) to draw on, same size as webcam feed
canvas = None

# ------------------------------
# MAIN LOOP: Process each webcam frame
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Initialize blank canvas if not already created
    if canvas is None:
        canvas = frame.copy() * 0  # Black image, same size as frame

    # Convert BGR (OpenCV default) to RGB for Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)  # Process frame for hand detection

    # If hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Extract index fingertip coordinates (for drawing)
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Check gestures
            fist_closed = is_fist_closed(hand_landmarks, w, h)
            two_fingers_together = are_two_fingers_together(hand_landmarks, w, h)

            # Change colors based on detected gesture
            if two_fingers_together:
                circle_color = (0, 0, 255)   # Red when two fingers together
                canvas_color = (0, 0, 255)
            else:
                circle_color = (0, 255, 0)   # Green when drawing normally
                canvas_color = (255, 0, 0)   # Blue drawing color

            # Draw fingertip marker on webcam feed
            cv2.circle(frame, (x, y), 8, circle_color, -1)

            # Only draw on canvas if fist is open
            if not fist_closed:
                cv2.circle(canvas, (x, y), 8, canvas_color, -1)

            # Draw landmarks and connections on hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display current gesture status text on screen
            if fist_closed:
                cv2.putText(frame, "FIST CLOSED - NOT DRAWING", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif two_fingers_together:
                cv2.putText(frame, "TWO FINGERS TOGETHER - RED MODE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "DRAWING MODE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ------------------------------
    # Combine the camera feed with the drawing canvas
    # ------------------------------
    combined = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    # Show the resulting window
    cv2.imshow("Draw in the Air", combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# Cleanup resources
# ------------------------------
cap.release()
cv2.destroyAllWindows()
