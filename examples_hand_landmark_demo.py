"""Example usage of the HandLandmarkDetector module.

This script demonstrates how to use the hand landmark detection module
for real-time hand detection from webcam input.
"""

import cv2
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hand_landmarks import HandLandmarkDetector
from camera import CameraCapture
from utils import draw_text, draw_fps
import time


def draw_hand_landmarks(frame, landmarks, pixel_landmarks, handedness):
    """Draw hand landmarks on frame.
    
    Args:
        frame: Input video frame
        landmarks: Normalized landmarks (21, 2)
        pixel_landmarks: Pixel coordinates landmarks (21, 2)
        handedness: 'Left' or 'Right'
    
    Returns:
        Frame with landmarks drawn
    """
    # Define hand skeleton connections (pairs of landmark indices)
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    
    # Draw connections (skeleton)
    for start_idx, end_idx in HAND_CONNECTIONS:
        start_point = tuple(pixel_landmarks[start_idx])
        end_point = tuple(pixel_landmarks[end_idx])
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    # Draw landmarks (joints)
    for i, point in enumerate(pixel_landmarks):
        cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
        # Label key landmarks
        if i == 0:
            cv2.putText(frame, "W", (int(point[0]) + 10, int(point[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw handedness info
    h, w = frame.shape[:2]
    info_text = f"Hand: {handedness}"
    cv2.putText(frame, info_text, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame


def main():
    """Main example function."""
    # Initialize detector with optimized settings for real-time
    detector = HandLandmarkDetector(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        static_image_mode=False  # Optimize for video/tracking
    )
    
    # Initialize camera
    camera = CameraCapture(camera_id=0)
    camera.start()
    
    # FPS tracking
    frame_count = 0
    start_time = time.time()
    
    print("Hand Landmark Detection - Real-time Demo")
    print("=========================================")
    print("Press 'q' to exit")
    print("Press 's' to print landmark coordinates")
    print()
    
    try:
        while True:
            # Capture frame
            success, frame = camera.get_frame()
            if not success:
                print("Failed to read frame")
                break
            
            # Detect hand landmarks
            landmarks, handedness = detector.detect(frame)
            
            # Display results
            if landmarks is not None:
                # Get pixel coordinates for visualization
                h, w = frame.shape[:2]
                pixel_landmarks = detector.get_landmark_pixel_coordinates(
                    landmarks, w, h
                )
                
                # Draw landmarks on frame
                frame = draw_hand_landmarks(frame, landmarks, pixel_landmarks, handedness)
                
                # Calculate some useful metrics
                thumb_index_dist = detector.calculate_landmark_distance(
                    landmarks,
                    HandLandmarkDetector.THUMB_TIP,
                    HandLandmarkDetector.INDEX_TIP
                )
                dist_text = f"Thumb-Index dist: {thumb_index_dist:.3f}"
                cv2.putText(frame, dist_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            else:
                # No hand detected
                cv2.putText(frame, "No hand detected", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                frame = draw_fps(frame, fps)
            
            # Display frame
            cv2.imshow("Hand Landmark Detection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and landmarks is not None:
                # Print landmark coordinates
                print(f"\nDetected {handedness} Hand Landmarks:")
                print(f"{'Index':<6} {'X':<10} {'Y':<10}")
                print("-" * 26)
                for i, (x, y) in enumerate(landmarks):
                    print(f"{i:<6} {x:<10.4f} {y:<10.4f}")
                print()
    
    finally:
        # Cleanup
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        print("\nDemo ended")


if __name__ == "__main__":
    main()
