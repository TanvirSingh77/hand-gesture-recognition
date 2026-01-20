"""Main entry point for the hand gesture recognition system."""

import cv2
import time
import config
from src.camera import CameraCapture
from src.gesture_detection import GestureDetector
from src.gesture_classifier import GestureClassifier
from src.utils import draw_fps, draw_text


def main():
    """Run the real-time hand gesture recognition system."""
    
    # Initialize components
    camera = CameraCapture(camera_id=config.CAMERA_ID)
    detector = GestureDetector()
    classifier = GestureClassifier(model_path=config.MODEL_PATH)
    
    # FPS calculation tracking
    previous_frame_time = time.time()
    fps = 0
    
    try:
        camera.start()
        print("Starting gesture recognition system...")
        
        while True:
            ret, frame = camera.get_frame()
            
            if not ret:
                print("Failed to capture frame")
                break
            
            # Resize frame for consistency
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            
            # Detect hand gestures
            results = detector.detect(frame)
            
            # Draw landmarks if enabled
            if config.SHOW_LANDMARKS:
                frame = detector.draw_landmarks(frame, results)
            
            # Classify gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture_id, confidence = classifier.classify(hand_landmarks)
                    gesture_name = classifier.get_gesture_name(gesture_id)
                    
                    # Display gesture info
                    text = f"{gesture_name}: {confidence:.2f}"
                    frame = draw_text(frame, text, (30, 60), color=(0, 255, 0))
            
            # Calculate and display FPS
            if config.SHOW_FPS:
                current_frame_time = time.time()
                time_diff = current_frame_time - previous_frame_time
                if time_diff > 0:
                    fps = 1 / time_diff
                previous_frame_time = current_frame_time
                frame = draw_fps(frame, fps)
            
            # Show the frame
            cv2.imshow(config.WINDOW_NAME, frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
    
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
