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
    GESTURE_DISTANCE_THRESHOLD = 0.09  # Increased for more stable click and drag

    print("Hand gesture control active. Move your index finger to control the cursor.")
    print("Pinch index finger and thumb to drag. Release to stop drag. Press 'q' to quit.")



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

        # Calculate the box for the current frame size (OpenCV window)
        screen_aspect = screen_width / screen_height
        frame_aspect = w / h
        if screen_aspect > frame_aspect:
            box_width = w - 40
            box_height = int(box_width / screen_aspect)
        else:
            box_height = h - 40
            box_width = int(box_height * screen_aspect)
        box_x1 = (w - box_width) // 2
        box_y1 = (h - box_height) // 2
        box_x2 = box_x1 + box_width
        box_y2 = box_y1 + box_height

        # Draw the rectangle border on the OpenCV frame
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 3)

        if frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        result = hands.process(rgb_frame)

        hand_in_box = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip_x = hand_landmarks.landmark[INDEX_FINGER_TIP].x
                index_finger_tip_y = hand_landmarks.landmark[INDEX_FINGER_TIP].y
                thumb_tip_x = hand_landmarks.landmark[THUMB_TIP].x
                thumb_tip_y = hand_landmarks.landmark[THUMB_TIP].y

                # Map normalized hand coordinates to box coordinates (frame coordinates)
                hand_x = int(index_finger_tip_x * box_width) + box_x1
                hand_y = int(index_finger_tip_y * box_height) + box_y1

                # Draw a circle on the OpenCV frame for visualization
                cv2.circle(frame, (hand_x, hand_y), 8, (0, 255, 255), -1)

                # Check if hand is inside the box
                if box_x1 <= hand_x <= box_x2 and box_y1 <= hand_y <= box_y2:
                    hand_in_box = True
                    # Map hand position inside the box to screen coordinates
                    rel_x = (hand_x - box_x1) / box_width
                    rel_y = (hand_y - box_y1) / box_height
                    target_x = int(rel_x * screen_width)
                    target_y = int(rel_y * screen_height)
                else:
                    hand_in_box = False

                # Calculate normalized distance between thumb and index finger
                distance_between_fingers = ((thumb_tip_x - index_finger_tip_x) ** 2 + (thumb_tip_y - index_finger_tip_y) ** 2) ** 0.5
                current_time = time.time()

                # Only move cursor if hand is inside the box
                if hand_in_box:
                    curr_x = prev_x + (target_x - prev_x) / smoothening
                    curr_y = prev_y + (target_y - prev_y) / smoothening
                    mouse.position = (curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

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
                else:
                    # If hand is outside, stop drag if it was active
                    if is_dragging:
                        print("...Stopping drag (hand out of box)")
                        mouse.release(Button.left)
                        is_dragging = False

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