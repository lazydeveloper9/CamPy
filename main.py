import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import time
import numpy as np


def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.4
    )
    mp_drawing = mp.solutions.drawing_utils

    # Use DirectShow backend for better performance on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Get screen size using pynput (pyautogui removed)
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    smoothening = 9
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    last_gesture_time = 0
    gesture_debounce_time = 0.3

    INDEX_FINGER_TIP = 8
    THUMB_TIP = 4

    # --- FPS Calculation Variables ---
    prev_frame_time = 0
    new_frame_time = 0
    # ---------------------------------

    # Mouse controller and drag state
    mouse = Controller()
    is_dragging = False

    # Set a threshold for the pinch/drag gesture (normalized distance)
    GESTURE_DISTANCE_THRESHOLD = 0.05

    print("Hand gesture control active. Move your index finger to control the cursor.")
    print("Pinch index finger and thumb to drag. Release to stop drag. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        h, w, c = frame.shape

        if frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip_x = hand_landmarks.landmark[INDEX_FINGER_TIP].x
                index_finger_tip_y = hand_landmarks.landmark[INDEX_FINGER_TIP].y
                thumb_tip_x = hand_landmarks.landmark[THUMB_TIP].x
                thumb_tip_y = hand_landmarks.landmark[THUMB_TIP].y

                x_pixel = int(index_finger_tip_x * w)
                y_pixel = int(index_finger_tip_y * h)
                cv2.circle(frame, (x_pixel, y_pixel), 10, (255, 0, 255), cv2.FILLED)

                target_x = int(index_finger_tip_x * screen_width)
                target_y = int(index_finger_tip_y * screen_height)

                curr_x = prev_x + (target_x - prev_x) / smoothening
                curr_y = prev_y + (target_y - prev_y) / smoothening
                mouse.position = (curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Calculate normalized distance between thumb and index finger
                distance_between_fingers = ((thumb_tip_x - index_finger_tip_x) ** 2 + (thumb_tip_y - index_finger_tip_y) ** 2) ** 0.5
                current_time = time.time()

                # Drag logic: pinch to drag, release to stop drag
                if distance_between_fingers < GESTURE_DISTANCE_THRESHOLD:
                    if not is_dragging and (current_time - last_gesture_time > gesture_debounce_time):
                        print("Starting drag...")
                        mouse.press(Button.left)
                        is_dragging = True
                        last_gesture_time = current_time
                else:
                    if is_dragging and (current_time - last_gesture_time > gesture_debounce_time):
                        print("...Stopping drag")
                        mouse.release(Button.left)
                        is_dragging = False
                        last_gesture_time = current_time

        # --- FPS Calculation and Display ---
        new_frame_time = time.time()
        if new_frame_time == prev_frame_time:
            fps = 0
        else:
            fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow('Hand Gesture Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()