"""Data collection script for hand gesture recognition.

This script captures hand landmarks from webcam frames and organizes them
by gesture class for training machine learning models. Features include:
- Real-time hand detection with visual feedback
- Keyboard controls for start/stop recording
- Automatic data organization by gesture class
- JSON-based landmark data storage
- Sample counter and statistics
"""

import cv2
import json
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import os

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hand_landmarks import HandLandmarkDetector
from camera import CameraCapture


class GestureDataCollector:
    """Collects and manages hand landmark data for gesture recognition."""
    
    def __init__(self, data_dir: str = "data/collected_gestures"):
        """Initialize the data collector.
        
        Args:
            data_dir: Directory to store collected gesture data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector
        self.detector = HandLandmarkDetector(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        # State tracking
        self.current_gesture = None
        self.is_recording = False
        self.samples_collected = {}
        self.current_sample_frames = []
        
        # Load existing sample counts
        self._load_sample_counts()
    
    def _load_sample_counts(self):
        """Load existing sample counts from saved data."""
        for gesture_dir in self.data_dir.iterdir():
            if gesture_dir.is_dir():
                gesture_name = gesture_dir.name
                sample_count = len(list(gesture_dir.glob("sample_*.json")))
                self.samples_collected[gesture_name] = sample_count
    
    def set_gesture(self, gesture_name: str):
        """Set the current gesture to collect.
        
        Args:
            gesture_name: Name of the gesture class
        """
        self.current_gesture = gesture_name.lower().replace(" ", "_")
        
        # Create gesture directory if it doesn't exist
        gesture_dir = self.data_dir / self.current_gesture
        gesture_dir.mkdir(exist_ok=True)
        
        # Initialize sample count if not exists
        if self.current_gesture not in self.samples_collected:
            self.samples_collected[self.current_gesture] = 0
    
    def start_recording(self):
        """Start recording a new gesture sample."""
        if self.current_gesture is None:
            return False
        
        self.is_recording = True
        self.current_sample_frames = []
        return True
    
    def stop_recording(self) -> bool:
        """Stop recording and save the gesture sample.
        
        Returns:
            True if sample was saved successfully
        """
        if not self.is_recording or not self.current_sample_frames:
            self.is_recording = False
            return False
        
        self.is_recording = False
        
        # Save collected landmarks
        sample_num = self.samples_collected[self.current_gesture]
        success = self._save_sample(self.current_sample_frames, sample_num)
        
        if success:
            self.samples_collected[self.current_gesture] += 1
            self.current_sample_frames = []
            return True
        
        return False
    
    def add_frame(self, landmarks: np.ndarray, handedness: str):
        """Add a frame to the current recording.
        
        Args:
            landmarks: Hand landmarks (21, 2) array
            handedness: 'Left' or 'Right'
        """
        if self.is_recording and landmarks is not None:
            frame_data = {
                "landmarks": landmarks.tolist(),
                "handedness": handedness
            }
            self.current_sample_frames.append(frame_data)
    
    def _save_sample(self, frames: list, sample_num: int) -> bool:
        """Save a gesture sample to disk.
        
        Args:
            frames: List of frame data
            sample_num: Sample number for naming
        
        Returns:
            True if save was successful
        """
        try:
            gesture_dir = self.data_dir / self.current_gesture
            
            # Create sample data structure
            sample_data = {
                "gesture": self.current_gesture,
                "sample_num": sample_num,
                "timestamp": datetime.now().isoformat(),
                "frame_count": len(frames),
                "frames": frames,
                "metadata": {
                    "detector_version": "1.0",
                    "frame_size": None,
                    "confidence": 0.7
                }
            }
            
            # Save to JSON file
            sample_file = gesture_dir / f"sample_{sample_num:05d}.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error saving sample: {e}")
            return False
    
    def get_statistics(self) -> dict:
        """Get collection statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": sum(self.samples_collected.values()),
            "gestures": self.samples_collected.copy(),
            "current_gesture": self.current_gesture
        }
        return stats
    
    def close(self):
        """Clean up resources."""
        self.detector.close()


def draw_ui(frame, collector: GestureDataCollector, recording_frames: int):
    """Draw UI elements on frame.
    
    Args:
        frame: Video frame
        collector: GestureDataCollector instance
        recording_frames: Number of frames in current recording
    
    Returns:
        Frame with UI drawn
    """
    h, w = frame.shape[:2]
    
    # Title bar
    cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
    cv2.putText(frame, "Hand Gesture Data Collector", (10, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Right side info panel
    panel_x = w - 320
    cv2.rectangle(frame, (panel_x - 10, 60), (w, h), (30, 30, 30), -1)
    
    y_offset = 90
    line_height = 30
    
    # Current gesture
    gesture_text = collector.current_gesture if collector.current_gesture else "None"
    cv2.putText(frame, f"Gesture: {gesture_text}", (panel_x, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    y_offset += line_height
    
    # Recording status
    if collector.is_recording:
        status_text = f"REC: {recording_frames} frames"
        status_color = (0, 0, 255)
        cv2.circle(frame, (panel_x + 5, y_offset - 8), 5, status_color, -1)
    else:
        status_text = "STOPPED"
        status_color = (100, 100, 100)
    
    cv2.putText(frame, status_text, (panel_x + 20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
    y_offset += line_height
    
    # Sample counts
    cv2.putText(frame, "Samples:", (panel_x, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += line_height
    
    for gesture, count in collector.samples_collected.items():
        display_gesture = gesture.replace("_", " ").title()
        count_text = f"  {display_gesture}: {count}"
        cv2.putText(frame, count_text, (panel_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
    
    # Instructions
    y_offset = h - 150
    cv2.putText(frame, "Controls:", (panel_x, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y_offset += 25
    
    instructions = [
        "1-9: Set gesture",
        "SPACE: Start/Stop",
        "R: Reset current",
        "S: Show stats",
        "Q: Quit"
    ]
    
    for instruction in instructions:
        cv2.putText(frame, instruction, (panel_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y_offset += 20
    
    return frame


def draw_landmarks(frame, landmarks, handedness):
    """Draw hand landmarks on frame.
    
    Args:
        frame: Video frame
        landmarks: Normalized landmarks (21, 2)
        handedness: 'Left' or 'Right'
    
    Returns:
        Frame with landmarks drawn
    """
    h, w = frame.shape[:2]
    pixel_landmarks = np.array(landmarks)
    pixel_landmarks[:, 0] *= w
    pixel_landmarks[:, 1] *= h
    pixel_landmarks = pixel_landmarks.astype(np.int32)
    
    # Hand connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    
    # Draw connections
    for start_idx, end_idx in connections:
        start_point = tuple(pixel_landmarks[start_idx])
        end_point = tuple(pixel_landmarks[end_idx])
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    # Draw landmarks
    for i, point in enumerate(pixel_landmarks):
        cv2.circle(frame, tuple(point), 4, (0, 0, 255), -1)
    
    # Draw hand label
    if handedness:
        label_y = pixel_landmarks[0][1] - 20
        label_x = pixel_landmarks[0][0]
        cv2.putText(frame, handedness, (label_x, max(0, label_y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return frame


def main():
    """Main data collection loop."""
    print("\n" + "="*60)
    print("Hand Gesture Data Collector")
    print("="*60)
    print("\nUsage:")
    print("  1-9: Select gesture class (1=Thumbs Up, 2=Peace, etc.)")
    print("  SPACE: Start/Stop recording")
    print("  R: Reset current gesture samples")
    print("  S: Show statistics")
    print("  Q: Quit")
    print("\nStarting webcam...\n")
    
    # Initialize collector and camera
    collector = GestureDataCollector()
    camera = CameraCapture(camera_id=0)
    
    try:
        camera.start()
    except RuntimeError as e:
        print(f"Error: {e}")
        collector.close()
        return
    
    # Gesture mappings
    gesture_map = {
        '1': 'thumbs_up',
        '2': 'peace',
        '3': 'ok',
        '4': 'fist',
        '5': 'open_hand',
        '6': 'point',
        '7': 'rock',
        '8': 'love',
        '9': 'custom'
    }
    
    # Statistics
    fps_counter = 0
    start_time = None
    
    print("Ready to collect data. Press '1'-'9' to select a gesture.\n")
    
    try:
        while True:
            ret, frame = camera.get_frame()
            if not ret:
                print("Error reading frame")
                break
            
            # Detect hand landmarks
            landmarks, handedness = collector.detector.detect(frame)
            
            # Draw landmarks if detected
            if landmarks is not None:
                frame = draw_landmarks(frame, landmarks, handedness)
                
                # Add frame to recording if active
                if collector.is_recording:
                    collector.add_frame(landmarks, handedness)
            
            # Draw UI
            frame = draw_ui(frame, collector, len(collector.current_sample_frames))
            
            # Display frame
            cv2.imshow("Hand Gesture Data Collector", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nExiting...")
                break
            
            elif key == ord(' '):  # Space to start/stop
                if collector.current_gesture is None:
                    print("Please select a gesture first (1-9)")
                elif collector.is_recording:
                    if collector.stop_recording():
                        sample_num = collector.samples_collected[collector.current_gesture] - 1
                        print(f"✓ Saved {collector.current_gesture} sample #{sample_num}")
                    else:
                        print("✗ Failed to save sample (no frames collected)")
                else:
                    if collector.start_recording():
                        print(f"▶ Recording {collector.current_gesture}... (press SPACE to stop)")
            
            elif key in [ord(str(i)) for i in range(1, 10)]:
                # Select gesture
                gesture_key = chr(key)
                if gesture_key in gesture_map:
                    gesture_name = gesture_map[gesture_key]
                    collector.set_gesture(gesture_name)
                    print(f"✓ Selected gesture: {gesture_name}")
            
            elif key == ord('r'):  # Reset current gesture
                if collector.current_gesture:
                    collector.samples_collected[collector.current_gesture] = 0
                    gesture_dir = collector.data_dir / collector.current_gesture
                    # Delete sample files
                    for f in gesture_dir.glob("sample_*.json"):
                        f.unlink()
                    print(f"✓ Reset {collector.current_gesture} samples")
            
            elif key == ord('s'):  # Show statistics
                stats = collector.get_statistics()
                print("\n" + "-"*40)
                print("Collection Statistics:")
                print(f"  Total samples: {stats['total_samples']}")
                print(f"  Current gesture: {stats['current_gesture']}")
                print("\n  Samples per gesture:")
                for gesture, count in stats['gestures'].items():
                    print(f"    {gesture}: {count}")
                print("-"*40 + "\n")
            
            fps_counter += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        camera.release()
        collector.close()
        cv2.destroyAllWindows()
        
        # Save final statistics
        stats = collector.get_statistics()
        print("\n" + "="*60)
        print("Final Statistics:")
        print(f"  Total samples collected: {stats['total_samples']}")
        print("\n  Breakdown by gesture:")
        for gesture, count in stats['gestures'].items():
            display_name = gesture.replace("_", " ").title()
            print(f"    {display_name}: {count} samples")
        print(f"\n  Data saved to: {collector.data_dir}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
