import cv2
import mediapipe as mp
import pyautogui # New import for cursor control
import time      # New import for slight delays

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Get screen dimensions for cursor mapping
    screen_width, screen_height = pyautogui.size()

    # Variables for smoothing cursor movement
    # These values might need adjustment based on your camera and preference
    smoothening = 9
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    # Variable to prevent multiple rapid clicks
    last_click_time = 0
    click_debounce_time = 0.3 # seconds

    # Define indices for specific hand landmarks (refer to MediaPipe Hand Landmarks diagram)
    # The tip of the index finger is landmark 8
    # The tip of the thumb is landmark 4
    INDEX_FINGER_TIP = 8
    THUMB_TIP = 4

    print("Hand gesture control active. Move your index finger to control the cursor.")
    print("Pinch index finger and thumb to click. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1) # Flip for natural mirror effect
        h, w, c = frame.shape # Get frame dimensions

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # --- Hand Landmark Processing and Cursor Control ---
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of index finger tip and thumb tip
                # Normalize coordinates (0 to 1) relative to frame size
                index_finger_tip_x = hand_landmarks.landmark[INDEX_FINGER_TIP].x
                index_finger_tip_y = hand_landmarks.landmark[INDEX_FINGER_TIP].y
                thumb_tip_x = hand_landmarks.landmark[THUMB_TIP].x
                thumb_tip_y = hand_landmarks.landmark[THUMB_TIP].y

                # Convert normalized coordinates to pixel coordinates on the frame
                # This helps visualize where the point is on the camera feed
                x_pixel = int(index_finger_tip_x * w)
                y_pixel = int(index_finger_tip_y * h)

                # Draw a circle at the index finger tip for visual feedback
                cv2.circle(frame, (x_pixel, y_pixel), 10, (255, 0, 255), cv2.FILLED)

                # Map index finger tip position to screen coordinates
                # Invert y-axis if needed (sometimes screen 0,0 is top-left, camera 0,0 is top-left)
                target_x = int(index_finger_tip_x * screen_width)
                target_y = int(index_finger_tip_y * screen_height)

                # Smoothening the cursor movement
                curr_x = prev_x + (target_x - prev_x) / smoothening
                curr_y = prev_y + (target_y - prev_y) / smoothening

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Check for "click" gesture (e.g., pinching thumb and index finger)
                # Calculate distance between thumb tip and index finger tip
                # You might need to adjust the 'click_distance_threshold' based on your hand size and camera
                click_distance = ((thumb_tip_x - index_finger_tip_x)**2 + (thumb_tip_y - index_finger_tip_y)**2)**0.5
                click_distance_threshold = 0.05 # Smaller value means closer together for a click

                current_time = time.time()
                if click_distance < click_distance_threshold and (current_time - last_click_time > click_debounce_time):
                    pyautogui.click()
                    print("Click!")
                    last_click_time = current_time


        # Display the resulting frame
        cv2.imshow('Hand Gesture Control', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture and destroy windows
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()