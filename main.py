import cv2
import mediapipe as mp
import pyautogui
import time

def main():
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    # Inside your main() function, where you initialize MediaPipe Hands
    hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6, # Slightly lower this if detection is too strict
    min_tracking_confidence=0.4   # Slightly lower this if tracking is frequently lost
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    screen_width, screen_height = pyautogui.size()

    smoothening = 9
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    last_click_time = 0
    click_debounce_time = 0.3

    INDEX_FINGER_TIP = 8
    THUMB_TIP = 4

    # --- FPS Calculation Variables ---
    prev_frame_time = 0
    new_frame_time = 0
    # ---------------------------------

    print("Hand gesture control active. Move your index finger to control the cursor.")
    print("Pinch index finger and thumb to click. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                click_distance = ((thumb_tip_x - index_finger_tip_x)**2 + (thumb_tip_y - index_finger_tip_y)**2)**0.5
                click_distance_threshold = 0.05

                current_time = time.time()
                if click_distance < click_distance_threshold and (current_time - last_click_time > click_debounce_time):
                    pyautogui.click()
                    print("Click!")
                    last_click_time = current_time

        # --- FPS Calculation and Display ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Convert FPS to an integer for display
        fps_text = f"FPS: {int(fps)}"

        # Display FPS on the frame
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        # Parameters: (image, text, (x, y), font, fontScale, color, thickness)
        # ---------------------------------

        cv2.imshow('Hand Gesture Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()