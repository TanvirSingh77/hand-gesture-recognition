"""Visualization script for collected gesture data.

This script allows you to:
- View collected gesture samples
- Replay recordings of hand movements
- Visualize landmark data
- Compare gestures
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from typing import List

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_utils import GestureDataLoader


class GestureVisualizer:
    """Visualize collected gesture data."""
    
    def __init__(self, data_dir: str = "data/collected_gestures"):
        """Initialize the visualizer.
        
        Args:
            data_dir: Directory containing collected gesture data
        """
        self.loader = GestureDataLoader(data_dir)
        self.frame_width = 640
        self.frame_height = 480
    
    def draw_landmarks_on_blank(self, 
                               landmarks: np.ndarray,
                               handedness: str = None) -> np.ndarray:
        """Draw landmarks on a blank canvas.
        
        Args:
            landmarks: Normalized landmarks (21, 2)
            handedness: Optional hand side label
        
        Returns:
            Image with drawn landmarks
        """
        # Create blank canvas
        frame = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 255
        
        # Convert to pixel coordinates
        pixel_landmarks = landmarks.copy()
        pixel_landmarks[:, 0] *= self.frame_width
        pixel_landmarks[:, 1] *= self.frame_height
        pixel_landmarks = pixel_landmarks.astype(np.int32)
        
        # Hand skeleton connections
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
            cv2.line(frame, start_point, end_point, (0, 200, 0), 2)
        
        # Draw landmarks
        for i, point in enumerate(pixel_landmarks):
            cv2.circle(frame, tuple(point), 5, (0, 0, 200), -1)
            cv2.circle(frame, tuple(point), 5, (200, 200, 200), 1)
        
        # Draw wrist label
        cv2.circle(frame, tuple(pixel_landmarks[0]), 7, (255, 0, 0), -1)
        
        # Draw handedness if provided
        if handedness:
            cv2.putText(frame, handedness, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return frame
    
    def replay_sample(self, gesture_name: str, sample_idx: int, speed: int = 50):
        """Replay a gesture sample with visualization.
        
        Args:
            gesture_name: Name of the gesture
            sample_idx: Index of the sample
            speed: Playback speed (1-100), default 50
        """
        samples = self.loader.get_gesture_samples(gesture_name)
        
        if sample_idx >= len(samples):
            print(f"Sample {sample_idx} not found for gesture {gesture_name}")
            return
        
        sample = samples[sample_idx]
        frames = sample.get("frames", [])
        
        if not frames:
            print("No frames in sample")
            return
        
        print(f"\nReplaying {gesture_name} - Sample {sample_idx}")
        print(f"Total frames: {len(frames)}")
        print("Press 'q' to stop, SPACE to pause\n")
        
        frame_idx = 0
        paused = False
        
        while frame_idx < len(frames):
            frame_data = frames[frame_idx]
            landmarks = np.array(frame_data["landmarks"], dtype=np.float32)
            handedness = frame_data.get("handedness", "Unknown")
            
            # Draw visualization
            display_frame = self.draw_landmarks_on_blank(landmarks, handedness)
            
            # Add info text
            cv2.putText(display_frame, f"Frame: {frame_idx + 1}/{len(frames)}",
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(display_frame, f"Sample: {gesture_name} #{sample_idx}",
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Show frame
            cv2.imshow("Gesture Replay", display_frame)
            
            # Handle input
            delay_ms = max(1, 101 - speed)  # Convert speed to delay
            key = cv2.waitKey(delay_ms) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            
            if not paused:
                frame_idx += 1
        
        cv2.destroyAllWindows()
    
    def compare_samples(self, gesture_name: str, sample_indices: List[int]):
        """Compare multiple samples by showing them side-by-side.
        
        Args:
            gesture_name: Name of the gesture
            sample_indices: List of sample indices to compare
        """
        samples = self.loader.get_gesture_samples(gesture_name)
        
        if not samples:
            print(f"No samples found for gesture {gesture_name}")
            return
        
        # Validate indices
        valid_indices = [i for i in sample_indices if i < len(samples)]
        if not valid_indices:
            print("No valid sample indices")
            return
        
        print(f"\nComparing {gesture_name} samples: {valid_indices}")
        print("Press 'q' to exit\n")
        
        max_frames = max(
            len(samples[i].get("frames", [])) 
            for i in valid_indices
        )
        
        for frame_idx in range(max_frames):
            # Create composite image
            cols = []
            
            for sample_idx in valid_indices:
                sample = samples[sample_idx]
                frames = sample.get("frames", [])
                
                if frame_idx < len(frames):
                    frame_data = frames[frame_idx]
                    landmarks = np.array(frame_data["landmarks"], dtype=np.float32)
                    handedness = frame_data.get("handedness", "")
                    
                    img = self.draw_landmarks_on_blank(landmarks, f"S{sample_idx}")
                else:
                    img = np.ones((self.frame_height, self.frame_width, 3), 
                                 dtype=np.uint8) * 200
                    cv2.putText(img, "No more frames", (100, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                cols.append(img)
            
            # Concatenate side-by-side
            composite = np.hstack(cols)
            
            # Add frame counter
            cv2.putText(composite, f"Frame: {frame_idx + 1}/{max_frames}",
                       (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            cv2.imshow("Gesture Comparison", composite)
            
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def show_gesture_summary(self, gesture_name: str):
        """Show summary of a gesture with first frame from each sample.
        
        Args:
            gesture_name: Name of the gesture
        """
        samples = self.loader.get_gesture_samples(gesture_name)
        
        if not samples:
            print(f"No samples found for gesture {gesture_name}")
            return
        
        print(f"\nGesture Summary: {gesture_name}")
        print(f"Total samples: {len(samples)}\n")
        
        # Calculate grid size
        cols = min(3, len(samples))
        rows = (len(samples) + cols - 1) // cols
        
        grid_images = []
        
        for sample_idx, sample in enumerate(samples):
            frames = sample.get("frames", [])
            
            if frames:
                frame_data = frames[0]
                landmarks = np.array(frame_data["landmarks"], dtype=np.float32)
                handedness = frame_data.get("handedness", "")
                
                img = self.draw_landmarks_on_blank(landmarks, f"#{sample_idx}")
            else:
                img = np.ones((self.frame_height, self.frame_width, 3), 
                             dtype=np.uint8) * 200
            
            # Add text info
            cv2.putText(img, f"Sample {sample_idx}", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            grid_images.append(img)
        
        # Create grid
        display = self._create_grid(grid_images, cols)
        
        cv2.imshow(f"Gesture Summary: {gesture_name}", display)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _create_grid(self, images: List[np.ndarray], cols: int) -> np.ndarray:
        """Create a grid of images.
        
        Args:
            images: List of images
            cols: Number of columns
        
        Returns:
            Grid image
        """
        rows = (len(images) + cols - 1) // cols
        
        # Pad with blank images if needed
        while len(images) < rows * cols:
            blank = np.ones((self.frame_height, self.frame_width, 3), 
                          dtype=np.uint8) * 200
            images.append(blank)
        
        # Create rows
        row_images = []
        for row_idx in range(rows):
            start_idx = row_idx * cols
            end_idx = min(start_idx + cols, len(images))
            row_images.append(np.hstack(images[start_idx:end_idx]))
        
        # Stack rows
        return np.vstack(row_images)


def main():
    """Main visualization interface."""
    print("\n" + "="*60)
    print("Gesture Data Visualizer")
    print("="*60)
    
    visualizer = GestureVisualizer()
    
    # Print available gestures
    stats = visualizer.loader.get_statistics()
    print(f"\nAvailable gestures ({len(stats['gestures'])} total):")
    for gesture_name in sorted(stats['gestures'].keys()):
        sample_count = stats['gestures'][gesture_name]['sample_count']
        print(f"  • {gesture_name}: {sample_count} samples")
    
    print("\n" + "="*60)
    print("Commands:")
    print("  v <gesture>         - View gesture summary")
    print("  r <gesture> <num>   - Replay sample")
    print("  c <gesture> <nums>  - Compare samples (e.g., 'c peace 0 1 2')")
    print("  s                   - Show statistics")
    print("  q                   - Quit")
    print("="*60 + "\n")
    
    while True:
        try:
            command = input("Enter command: ").strip().split()
            
            if not command:
                continue
            
            if command[0].lower() == 'q':
                print("Exiting...")
                break
            
            elif command[0].lower() == 'v':
                if len(command) < 2:
                    print("Usage: v <gesture_name>")
                    continue
                
                gesture_name = command[1]
                visualizer.show_gesture_summary(gesture_name)
            
            elif command[0].lower() == 'r':
                if len(command) < 3:
                    print("Usage: r <gesture_name> <sample_number>")
                    continue
                
                gesture_name = command[1]
                sample_num = int(command[2])
                visualizer.replay_sample(gesture_name, sample_num)
            
            elif command[0].lower() == 'c':
                if len(command) < 3:
                    print("Usage: c <gesture_name> <sample1> [sample2] [sample3] ...")
                    continue
                
                gesture_name = command[1]
                sample_indices = [int(x) for x in command[2:]]
                visualizer.compare_samples(gesture_name, sample_indices)
            
            elif command[0].lower() == 's':
                visualizer.loader.print_statistics()
            
            else:
                print("Unknown command")
        
        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
