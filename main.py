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

    # Get device screen size
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    smoothening = 9  # Lower value for higher sensitivity
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    last_gesture_time = 0
    gesture_debounce_time = 0.6  # Increased debounce for more stable drag activation
    drag_hold_start = None
    drag_release_start = None
    MIN_PINCH_HOLD_TIME = 0.18  # seconds, must hold pinch this long to start drag
    MIN_RELEASE_HOLD_TIME = 0.18  # seconds, must release pinch this long to stop drag
    last_click_time = 0
    CLICK_DEBOUNCE_TIME = 0.5  # seconds
    PINKY_FINGER_MCP = 17  # Pinky finger metacarpal
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
    GESTURE_DISTANCE_THRESHOLD = 0.025 # Increased for more stable click and drag

    print("Hand gesture control active. Move your pinky finger to control the cursor.")
    print("Pinch pinky finger and thumb to drag. Release to stop drag. Press 'q' to quit.")



    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        # Resize frame to 720p height while maintaining aspect ratio
        target_height = 720
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        frame = cv2.resize(frame, (target_width, target_height))
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        h, w, c = frame.shape

    # Set frame to a smaller fixed size (e.g., 640x360)
    # (Already resized above, so just ensure target_height and target_width are set)

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
                pinky_finger_x = hand_landmarks.landmark[PINKY_FINGER_MCP].x # Adjusted for better pinch detection
                pinky_finger_y = hand_landmarks.landmark[PINKY_FINGER_MCP].y # Adjusted for better pinch detection
                # Map normalized hand coordinates to frame coordinates
                hand_x = int(pinky_finger_x * w)
                hand_y = int(pinky_finger_y * h)
                # thumb_x = int(thumb_tip_x * w)
                # thumb_y = int(thumb_tip_y * h)  
                # index_x = int(index_finger_tip_x * w)
                # index_y = int(index_finger_tip_y * h)
                # Draw a circle on the OpenCV frame for visualization
                cv2.circle(frame, (hand_x, hand_y), 8, (0, 0, 255), -1)
                # cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 0, 255), -1)
                # cv2.circle(frame, (index_x, index_y), 8, (0, 0, 255), -1)

                # Map only a central region of the frame to the full screen
                # E.g., use 20% margin on each side, so only 60% of the frame area is mapped
                margin_x = 0.3
                margin_y = 0.3
                min_x = margin_x
                max_x = 1.0 - margin_x
                min_y = margin_y
                max_y = 1.0 - margin_y
                # Clamp hand position to the active region
                rel_y = (pinky_finger_y - min_y) / (max_y - min_y)
                rel_x = (pinky_finger_x - min_x) / (max_x - min_x)
                rel_x = min(max(rel_x, 0.0), 1.0)
                rel_y = min(max(rel_y, 0.0), 1.0)
                target_x = int(rel_x * (screen_width - 1))
                target_y = int(rel_y * (screen_height - 1))

                # Calculate normalized distance between thumb and index finger
                distance_between_fingers = ((thumb_tip_x - index_finger_tip_x) ** 2 + (thumb_tip_y - index_finger_tip_y) ** 2) ** 0.5
                current_time = time.time()

                # Move cursor always (no box restriction)
                curr_x = prev_x + (target_x - prev_x) / smoothening
                curr_y = prev_y + (target_y - prev_y) / smoothening
                mouse.position = (curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Click logic: single click when pinch detected and not dragging
                if distance_between_fingers < GESTURE_DISTANCE_THRESHOLD:
                    if not is_dragging:
                        if current_time - last_click_time > CLICK_DEBOUNCE_TIME:
                            print("Click!")
                            mouse.click(Button.left)
                            last_click_time = current_time
                    # Drag logic: pinch to drag, release to stop drag
                    drag_release_start = None
                    if not is_dragging:
                        if drag_hold_start is None:
                            drag_hold_start = current_time
                        elif (current_time - drag_hold_start) > MIN_PINCH_HOLD_TIME:
                            print("Starting drag...")
                            mouse.press(Button.left)
                            is_dragging = True
                            last_gesture_time = current_time
                    else:
                        drag_hold_start = None  # Already dragging
                else:
                    # Pinch released
                    drag_hold_start = None
                    if is_dragging:
                        if drag_release_start is None:
                            drag_release_start = current_time
                        elif (current_time - drag_release_start) > MIN_RELEASE_HOLD_TIME:
                            print("...Stopping drag")
                            mouse.release(Button.left)
                            is_dragging = False
                            last_gesture_time = current_time
                    else:
                        drag_release_start = None  # Not dragging

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