Hand Gesture Mouse Control
This Python application allows you to control your mouse cursor and perform clicks and drags using hand gestures, leveraging your webcam and MediaPipe.
Features
 * Cursor Control: Move your pinky finger to control the mouse cursor.
 * Click: Pinch your index finger and thumb together to perform a left-click.
 * Drag and Drop: Hold the pinch gesture to initiate a drag, and release the pinch to drop.
 * Visual Feedback: See your hand landmarks and a highlighted control point on the webcam feed.
 * Adjustable Sensitivity: Customize cursor smoothing for desired responsiveness.
How it Works
The application uses the following libraries and concepts:
 * OpenCV (cv2): Captures video from your webcam and displays the output.
 * MediaPipe (mediapipe): Detects and tracks hand landmarks in real-time.
 * pynput: Controls your mouse cursor.
 * Gesture Recognition:
   * The pinky finger's metacarpal position is used to map the hand's movement to the screen's cursor position.
   * A "pinch" gesture is detected by calculating the distance between the thumb tip and the index finger tip.
   * Timers are used to debounce clicks and to establish a sustained pinch for drag operations, preventing accidental clicks/drags.
 * Screen Mapping: A central region of the webcam feed is mapped to your entire screen for more intuitive control.
 * Smoothing: Cursor movement is smoothed to provide a more stable and less jumpy experience.
Getting Started
Prerequisites
Before running the application, ensure you have Python 3 installed. Then, install the necessary libraries using pip:
pip install opencv-python mediapipe pynput tkinter numpy

Running the Application
 * Save the code: Save the provided Python code as a .py file (e.g., hand_mouse.py).
 * Run from terminal: Open your terminal or command prompt, navigate to the directory where you saved the file, and run:
   python hand_mouse.py

Usage
 * Once the application starts, a window displaying your webcam feed will appear.
 * Move your pinky finger: Your mouse cursor will follow the movement of your pinky finger.
 * Click: Bring your thumb tip and index finger tip close together (a "pinch" gesture) to perform a left-click.
 * Drag: Maintain the pinch gesture for a short duration (approximately 0.18 seconds) to start dragging. Release the pinch to stop dragging.
 * Quit: Press the q key on your keyboard to exit the application.
Configuration
You can adjust several parameters in the main() function to fine-tune the application's behavior:
 * smoothening: Controls the cursor smoothing. A lower value makes the cursor more sensitive and less smooth, while a higher value makes it smoother but less responsive. (Default: 9)
 * gesture_debounce_time: Time in seconds to prevent rapid, accidental gesture activations. (Default: 0.6)
 * MIN_PINCH_HOLD_TIME: The minimum duration (in seconds) you must hold a pinch to initiate a drag. (Default: 0.18)
 * MIN_RELEASE_HOLD_TIME: The minimum duration (in seconds) you must hold a released pinch to stop dragging. (Default: 0.18)
 * CLICK_DEBOUNCE_TIME: Time in seconds to prevent multiple rapid clicks from a single pinch. (Default: 0.5)
 * GESTURE_DISTANCE_THRESHOLD: The normalized distance between the thumb and index finger to register a pinch. A smaller value requires a tighter pinch. (Default: 0.025)
 * margin_x, margin_y: These values define the central region of the webcam feed that maps to your screen. Increasing these values means a smaller active area in the webcam frame controls the full screen. (Default: 0.3 for both, meaning the central 40% of the frame is active).
 
