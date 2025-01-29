import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Screen resolution (change if necessary)
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(1)  # Use camera index 1 (as per your setup)

# Gesture settings
click_threshold = 40  # Distance threshold for pinch gesture

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB and process hand landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for index finger and thumb
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert normalized coordinates to screen coordinates
            ix, iy = int(index_finger.x * w), int(index_finger.y * h)
            sx, sy = int(index_finger.x * screen_width), int(index_finger.y * screen_height)

            # Move mouse to index finger position
            pyautogui.moveTo(sx, sy)

            # Check pinch gesture for click
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            distance = np.linalg.norm(np.array([ix, iy]) - np.array([thumb_x, thumb_y]))

            if distance < click_threshold:import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Open webcam with higher resolution
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use camera index 1 (as per your setup)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

click_threshold = 40  # Distance threshold for pinch gesture

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for a mirror effect
    h, w, _ = frame.shape

    # Convert to RGB and process hand landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get landmarks for index finger and thumb
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert to screen coordinates
            sx, sy = int(index_finger.x * screen_width), int(index_finger.y * screen_height)

            # Move mouse cursor
            pyautogui.moveTo(sx, sy, duration=0.05)  # Smoother movement

            # Pinch detection for click
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
            distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))

            if distance < click_threshold:
                pyautogui.click()

    # Show the frame (larger size)
    cv2.imshow("Hand Tracking", cv2.resize(frame, (800, 600)))  

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

                pyautogui.click()
                cv2.putText(frame, "Click!", (ix, iy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
