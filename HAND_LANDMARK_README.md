# Hand Landmark Detection Module

A production-ready, reusable Python module for detecting hand landmarks in real-time using MediaPipe. This module provides a clean, optimized interface for extracting 21 hand landmarks from video frames.

## Features

✅ **Real-time Performance**
- Optimized for single hand detection
- Tracking-based temporal consistency
- Configurable confidence thresholds
- ~30+ FPS on standard hardware

✅ **Accurate Landmark Detection**
- Detects all 21 hand landmarks per hand
- Returns normalized (x, y) coordinates
- Named constants for easy landmark access
- Z-coordinates available if needed

✅ **Clean Code Design**
- Full type hints for IDE support
- Comprehensive docstrings
- Error handling and validation
- Context manager support

✅ **Easy Integration**
- Simple 3-line detection API
- Utility methods for common operations
- Compatible with OpenCV
- Works with existing gesture classifier

## Installation

Ensure you have the required dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages:
- `mediapipe>=0.10.9` - Hand detection
- `opencv-python>=4.8.1` - Frame handling
- `numpy>=1.24.3` - Numerical operations

## Quick Start

```python
from src.hand_landmarks import HandLandmarkDetector
import cv2

# Initialize detector
detector = HandLandmarkDetector()

# Capture frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Detect landmarks
landmarks, handedness = detector.detect(frame)

if landmarks is not None:
    print(f"Detected {handedness} hand")
    print(f"21 landmarks shape: {landmarks.shape}")  # (21, 2)
    print(f"First landmark (wrist): {landmarks[0]}")
```

## Hand Landmarks

The module detects 21 hand landmarks in this order:

```
0: Wrist

Thumb (1-4):
1: CMC, 2: MCP, 3: IP, 4: Tip

Index Finger (5-8):
5: MCP, 6: PIP, 7: DIP, 8: Tip

Middle Finger (9-12):
9: MCP, 10: PIP, 11: DIP, 12: Tip

Ring Finger (13-16):
13: MCP, 14: PIP, 15: DIP, 16: Tip

Pinky (17-20):
17: MCP, 18: PIP, 19: DIP, 20: Tip
```

Access landmarks using named constants:

```python
detector = HandLandmarkDetector()
landmarks, _ = detector.detect(frame)

if landmarks is not None:
    wrist = landmarks[detector.WRIST]           # Index 0
    thumb_tip = landmarks[detector.THUMB_TIP]   # Index 4
    index_tip = landmarks[detector.INDEX_TIP]   # Index 8
```

## Coordinate System

Landmarks are returned as **normalized coordinates** (0.0 to 1.0):

- **x-axis**: 0.0 (left) → 1.0 (right)
- **y-axis**: 0.0 (top) → 1.0 (bottom)
- **z-axis**: Relative depth (optional, not included in basic 2D detection)

### Convert to Pixel Coordinates

```python
# Get normalized landmarks
landmarks, _ = detector.detect(frame)

# Convert to pixel coordinates for visualization
h, w = frame.shape[:2]
pixel_landmarks = detector.get_landmark_pixel_coordinates(landmarks, w, h)

# Now use pixel_landmarks with cv2.circle(), cv2.line(), etc.
cv2.circle(frame, tuple(pixel_landmarks[0]), 5, (0, 255, 0), -1)
```

## Common Operations

### 1. Detect Pinching Gesture

```python
distance = detector.calculate_landmark_distance(
    landmarks,
    detector.THUMB_TIP,
    detector.INDEX_TIP
)

if distance < 0.05:  # Fingers close together
    print("Pinch detected!")
```

### 2. Get Hand Bounding Box

```python
x_min, y_min, x_max, y_max = detector.get_hand_bounding_box(
    landmarks, w, h, padding=0.1
)

# Crop hand region
hand_region = frame[y_min:y_max, x_min:x_max]
```

### 3. Extract Features for ML

```python
# Flatten for ML models
features = landmarks.flatten()  # Shape: (42,)

# Or create custom features
features = [
    detector.calculate_landmark_distance(landmarks, 4, 8),  # thumb-index
    detector.calculate_landmark_distance(landmarks, 8, 12),  # index-middle
    # Add more distances...
]
```

### 4. Real-time Detection Loop

```python
with HandLandmarkDetector() as detector:
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks, handedness = detector.detect(frame)
        
        if landmarks is not None:
            # Draw landmarks
            h, w = frame.shape[:2]
            px_lm = detector.get_landmark_pixel_coordinates(landmarks, w, h)
            
            for point in px_lm:
                cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
        
        cv2.imshow('Hand Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

## API Reference

### HandLandmarkDetector Class

#### Constructor

```python
detector = HandLandmarkDetector(
    min_detection_confidence=0.5,    # Initial detection threshold
    min_tracking_confidence=0.5,     # Tracking threshold
    static_image_mode=False          # Use tracking (video) vs static (image)
)
```

**Parameters:**
- `min_detection_confidence` (float): 0.0-1.0, higher = stricter
- `min_tracking_confidence` (float): 0.0-1.0, only used in video mode
- `static_image_mode` (bool): True for single images, False for video

#### Core Methods

```python
# Detect hand landmarks
landmarks, handedness = detector.detect(frame)
# Returns: (np.ndarray[21,2], str) or (None, None)

# Convert to pixel coordinates
px_landmarks = detector.get_landmark_pixel_coordinates(landmarks, width, height)
# Returns: np.ndarray[21,2] in pixel space

# Calculate distance between landmarks
distance = detector.calculate_landmark_distance(landmarks, idx1, idx2)
# Returns: float in normalized space

# Get bounding box
x_min, y_min, x_max, y_max = detector.get_hand_bounding_box(
    landmarks, width, height, padding=0.1
)
# Returns: tuple of ints (pixel coordinates)

# Cleanup
detector.close()
```

#### Landmark Constants

```python
# Finger tips
detector.THUMB_TIP    # Index 4
detector.INDEX_TIP    # Index 8
detector.MIDDLE_TIP   # Index 12
detector.RING_TIP     # Index 16
detector.PINKY_TIP    # Index 20

# Joints
detector.THUMB_MCP, detector.THUMB_IP, ...
detector.INDEX_MCP, detector.INDEX_PIP, detector.INDEX_DIP, ...
# ... and so on for all fingers
```

## Performance Optimization

### Best Practices

1. **Reuse detector instance** - Initialize once, use for entire session
2. **Use context manager** - Automatically manages resources
3. **Set appropriate thresholds** - Balance accuracy vs speed
4. **Resize frames if needed** - Smaller frames = faster processing
5. **Use tracking mode** - `static_image_mode=False` for video

### Configuration for Different Scenarios

```python
# Fast/lenient (more detections, lower accuracy)
detector = HandLandmarkDetector(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

# Balanced (default)
detector = HandLandmarkDetector(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Strict/accurate (fewer but accurate detections)
detector = HandLandmarkDetector(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.8
)
```

## Integration with Gesture Classifier

The hand landmarks are designed to integrate seamlessly with the gesture classifier:

```python
from src.hand_landmarks import HandLandmarkDetector
from src.gesture_classifier import GestureClassifier

detector = HandLandmarkDetector()
classifier = GestureClassifier()

landmarks, hand = detector.detect(frame)
if landmarks is not None:
    # Prepare features
    features = landmarks.flatten()  # (42,)
    
    # Classify gesture
    gesture = classifier.predict(features)
    print(f"Gesture: {gesture}")
```

## Testing

Run the unit tests:

```bash
pytest tests/test_hand_landmarks.py -v
```

Tests cover:
- Initialization and configuration
- Landmark detection
- Coordinate conversions
- Distance calculations
- Error handling
- Resource management

## Example Scripts

### Real-time Detection Demo

```bash
python examples_hand_landmark_demo.py
```

Features:
- Live webcam hand detection
- Draws landmarks and skeleton
- Shows hand side (left/right)
- Calculates finger distances
- Displays FPS
- Press 's' to print coordinates, 'q' to quit

## Troubleshooting

### No hand detected

1. Ensure good lighting
2. Keep hand visible and not too far from camera
3. Lower detection threshold:
   ```python
   detector = HandLandmarkDetector(min_detection_confidence=0.5)
   ```

### Slow performance

1. Reduce frame size
2. Lower confidence thresholds
3. Use `static_image_mode=False` for video
4. Ensure GPU acceleration if available

### Jittery detections

1. Increase `min_tracking_confidence`
2. Use temporal smoothing on landmarks
3. Ensure consistent lighting

## Architecture

```
hand_landmarks.py
├── HandLandmarkDetector
│   ├── __init__()           # Initialize with thresholds
│   ├── detect()             # Main detection method
│   ├── get_landmark_pixel_coordinates()  # Coord conversion
│   ├── calculate_landmark_distance()      # Distance metric
│   ├── get_hand_bounding_box()            # Bounding box
│   └── close()              # Resource cleanup
```

## Clean Code Principles Applied

- ✅ **Single Responsibility** - Each method does one thing
- ✅ **Type Hints** - Full type annotations for clarity
- ✅ **Error Handling** - Validates inputs and provides clear errors
- ✅ **Documentation** - Comprehensive docstrings with examples
- ✅ **Constants** - Named landmark indices for maintainability
- ✅ **Resource Management** - Context manager support
- ✅ **Testability** - Modular design with unit tests

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mediapipe | 0.10.9+ | Hand landmark detection |
| opencv-python | 4.8.1+ | Frame handling |
| numpy | 1.24.3+ | Numerical operations |

## License

This module is part of the hand gesture recognition system.

## References

- [MediaPipe Hand Documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [MediaPipe GitHub](https://github.com/google/mediapipe)
