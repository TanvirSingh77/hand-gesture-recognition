# Hand Landmark Detection - Quick Reference

## 1. Installation
```bash
pip install -r requirements.txt
```

## 2. Basic Usage (3 lines!)
```python
from src.hand_landmarks import HandLandmarkDetector

detector = HandLandmarkDetector()
landmarks, handedness = detector.detect(frame)  # frame is numpy array
```

## 3. Key Return Values

| Name | Type | Description |
|------|------|-------------|
| `landmarks` | numpy array (21, 2) | Normalized (x, y) coords: 0.0 to 1.0 |
| `handedness` | str | 'Left' or 'Right' |

## 4. Hand Landmarks (21 total)

```
INDEX  FINGER           INDEX  FINGER
  0    Wrist              10   Middle PIP
  1    Thumb CMC          11   Middle DIP
  2    Thumb MCP          12   Middle TIP
  3    Thumb IP           13   Ring MCP
  4    Thumb TIP          14   Ring PIP
  5    Index MCP          15   Ring DIP
  6    Index PIP          16   Ring TIP
  7    Index DIP          17   Pinky MCP
  8    Index TIP          18   Pinky PIP
  9    Middle MCP         19   Pinky DIP
                          20   Pinky TIP
```

## 5. Named Landmark Access

```python
detector.WRIST           # 0
detector.THUMB_TIP       # 4
detector.INDEX_TIP       # 8
detector.MIDDLE_TIP      # 12
detector.RING_TIP        # 16
detector.PINKY_TIP       # 20
```

## 6. Common Operations

### Convert to Pixel Coordinates
```python
h, w = frame.shape[:2]
pixel_landmarks = detector.get_landmark_pixel_coordinates(landmarks, w, h)
```

### Calculate Distance Between Landmarks
```python
distance = detector.calculate_landmark_distance(
    landmarks,
    detector.THUMB_TIP,
    detector.INDEX_TIP
)
```

### Get Bounding Box
```python
x_min, y_min, x_max, y_max = detector.get_hand_bounding_box(
    landmarks, w, h, padding=0.1
)
```

### Draw Landmarks
```python
h, w = frame.shape[:2]
px = detector.get_landmark_pixel_coordinates(landmarks, w, h)

for point in px:
    cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)

cv2.imshow("Hand", frame)
```

## 7. Real-time Video Loop

```python
from src.hand_landmarks import HandLandmarkDetector
import cv2

with HandLandmarkDetector() as detector:
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks, hand = detector.detect(frame)
        
        if landmarks is not None:
            print(f"Detected {hand} hand")
            # Process landmarks...
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

## 8. Gesture Detection Example

```python
# Pinch detection
distance = detector.calculate_landmark_distance(
    landmarks, detector.THUMB_TIP, detector.INDEX_TIP
)
if distance < 0.05:
    print("PINCH!")

# Open hand detection
spread = detector.calculate_landmark_distance(
    landmarks, detector.INDEX_MCP, detector.PINKY_MCP
)
if spread > 0.3:
    print("OPEN HAND!")
```

## 9. Initialization Options

```python
# Default (balanced)
detector = HandLandmarkDetector()

# Lenient (more detections)
detector = HandLandmarkDetector(min_detection_confidence=0.5)

# Strict (accurate)
detector = HandLandmarkDetector(min_detection_confidence=0.9)

# For static images
detector = HandLandmarkDetector(static_image_mode=True)
```

## 10. Important Notes

✅ Always check if `landmarks is not None` before using  
✅ Use context manager for auto cleanup: `with HandLandmarkDetector() as detector:`  
✅ Landmarks are normalized (0.0-1.0), convert for pixel operations  
✅ Real-time requires video frames, not single images  
✅ Works best with good lighting  

## 11. Error Handling

```python
try:
    landmarks, hand = detector.detect(frame)
    if landmarks is None:
        print("No hand detected")
    else:
        print(f"Detected {hand} hand")
except ValueError as e:
    print(f"Invalid input: {e}")
```

## 12. Performance Tips

- Keep frames at 640x480 or smaller
- Use `static_image_mode=False` for video
- Reuse detector instance (don't reinitialize)
- Set confidence to 0.7 for good balance
- Expected: 30+ FPS on standard hardware

## 13. Testing

```bash
pytest tests/test_hand_landmarks.py -v
```

## 14. Documentation Files

| File | Purpose |
|------|---------|
| HAND_LANDMARK_README.md | Getting started guide |
| HAND_LANDMARK_API.md | Complete API reference |
| HAND_LANDMARK_BEST_PRACTICES.md | Design patterns & guidelines |
| IMPLEMENTATION_SUMMARY.md | Implementation details |
| This file | Quick reference |

## 15. Demo Script

```bash
python examples_hand_landmark_demo.py
```
- 's' = print landmark coordinates
- 'q' = quit

## 16. Troubleshooting

| Issue | Solution |
|-------|----------|
| No hand detected | Lower detection threshold or improve lighting |
| Slow performance | Reduce frame size or lower confidence |
| Jittery detection | Increase tracking confidence or add smoothing |
| "No attribute" error | Make sure MediaPipe is installed: `pip install mediapipe` |

## 17. Common Patterns

### Feature Extraction for ML
```python
features = landmarks.flatten()  # (42,) array
# or
features = [
    detector.calculate_landmark_distance(landmarks, 4, 8),
    detector.calculate_landmark_distance(landmarks, 8, 12),
]
```

### Extract Hand Region
```python
x_min, y_min, x_max, y_max = detector.get_hand_bounding_box(landmarks, w, h)
hand_region = frame[y_min:y_max, x_min:x_max]
```

## 18. Landmark Constants Reference

```python
# Wrist
detector.WRIST = 0

# Thumb
detector.THUMB_CMC = 1
detector.THUMB_MCP = 2
detector.THUMB_IP = 3
detector.THUMB_TIP = 4

# Index Finger
detector.INDEX_MCP = 5
detector.INDEX_PIP = 6
detector.INDEX_DIP = 7
detector.INDEX_TIP = 8

# Middle Finger
detector.MIDDLE_MCP = 9
detector.MIDDLE_PIP = 10
detector.MIDDLE_DIP = 11
detector.MIDDLE_TIP = 12

# Ring Finger
detector.RING_MCP = 13
detector.RING_PIP = 14
detector.RING_DIP = 15
detector.RING_TIP = 16

# Pinky
detector.PINKY_MCP = 17
detector.PINKY_PIP = 18
detector.PINKY_DIP = 19
detector.PINKY_TIP = 20

# All landmarks
detector.NUM_LANDMARKS = 21
```

## 19. Next Steps

1. Read [HAND_LANDMARK_README.md](HAND_LANDMARK_README.md) for detailed guide
2. Run `python examples_hand_landmark_demo.py` to see it in action
3. Check [HAND_LANDMARK_API.md](HAND_LANDMARK_API.md) for complete API
4. Review [HAND_LANDMARK_BEST_PRACTICES.md](HAND_LANDMARK_BEST_PRACTICES.md) for advanced patterns
5. Run `pytest tests/test_hand_landmarks.py -v` to verify installation

---
**Last Updated**: January 20, 2026  
**Implementation Status**: ✅ Complete and Production-Ready
