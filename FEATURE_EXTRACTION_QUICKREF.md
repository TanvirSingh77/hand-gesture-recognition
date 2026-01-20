# Feature Engineering Module - Quick Reference

## 46-Feature Gesture Feature Vector

### Feature Breakdown

```
Total Features: 46

├── Inter-joint Distances (21 features)
│   ├── Wrist to Fingertips (5)
│   ├── Finger Inter-joint Segments (12)
│   └── MCP Joint Spread (4)
│
├── Joint Angles (15 features)
│   └── 3 angles × 5 fingers (Thumb, Index, Middle, Ring, Pinky)
│
├── Hand Span Metrics (4 features)
│   ├── Bounding box width
│   ├── Bounding box height
│   ├── Aspect ratio
│   └── Maximum reach from wrist
│
└── Relative Positions (6 features)
    ├── Wrist relative Y
    ├── Thumb tip (X, Y)
    ├── Index tip (X, Y)
    └── Pinky tip X
```

## Quick Start

```python
from src.hand_landmarks import HandLandmarkDetector
from src.feature_extractor import HandGestureFeatureExtractor

# Initialize
detector = HandLandmarkDetector()
extractor = HandGestureFeatureExtractor(normalize=True)

# Extract
frame = cv2.imread("gesture.jpg")
landmarks, handedness = detector.detect(frame)
features = extractor.extract(landmarks)  # shape: (46,)
```

## Feature Categories Explained

### 1. Inter-Joint Distances (Normalized)
**What:** Euclidean distances between hand joints, normalized by hand size

**Why:** 
- Captures hand geometry and spread
- Scale-invariant (same gesture at different distances = same features)
- Indicates finger opening/closing

**Example values:**
- **High:** Fingers extended away from each other
- **Low:** Fingers bent or close together

---

### 2. Joint Angles
**What:** Angles between finger segments in degrees [0°, 180°]

**Why:**
- Captures finger bending/flexion state
- Key differentiator between gestures
- Angle-aware feature representation

**Example values:**
- **180°:** Finger fully extended
- **90°:** Finger at right angle
- **0°:** Finger fully bent

**Common patterns:**
- Peace sign: Index & middle ~180°, others ~90°
- Fist: All angles small (~45-90°)
- Open hand: All angles ~180°

---

### 3. Hand Span Metrics
**What:** Overall hand size and shape properties

**Features:**
- **Width:** Horizontal extent of hand
- **Height:** Vertical extent of hand
- **Aspect ratio:** Width/Height (>1 = wider than tall)
- **Max reach:** Distance from wrist to furthest point

**Why:** Distinguishes between different hand sizes and overall gesture scale

---

### 4. Relative Positions
**What:** Finger tip positions within hand bounding box [0, 1]

**Why:** 
- Captures relative positioning of fingers
- Useful for asymmetric gestures
- Rotation-invariant representation

**Example:**
- Thumbs up: Thumb tip at high Y position
- Thumbs down: Thumb tip at low Y position

---

## Feature Names (Complete List)

### Inter-Joint Distances (21)
```
1. wrist_to_thumb_tip
2. wrist_to_index_tip
3. wrist_to_middle_tip
4. wrist_to_ring_tip
5. wrist_to_pinky_tip
6. thumb_cmc_mcp_dist
7. thumb_mcp_ip_dist
8. thumb_ip_tip_dist
9. index_mcp_pip_dist
10. index_pip_dip_dist
11. index_dip_tip_dist
12. middle_mcp_pip_dist
13. middle_pip_dip_dist
14. middle_dip_tip_dist
15. ring_mcp_pip_dist
16. ring_pip_dip_dist
17. ring_dip_tip_dist
18. pinky_mcp_pip_dist
19. pinky_pip_dip_dist
20. pinky_dip_tip_dist
21. thumb_index_mcp_dist
22. index_middle_mcp_dist
23. middle_ring_mcp_dist
24. ring_pinky_mcp_dist
```

### Joint Angles (15)
```
25. thumb_mcp_angle
26. thumb_ip_angle
27. thumb_overall_angle
28. index_pip_angle
29. index_dip_angle
30. index_overall_angle
31. middle_pip_angle
32. middle_dip_angle
33. middle_overall_angle
34. ring_pip_angle
35. ring_dip_angle
36. ring_overall_angle
37. pinky_pip_angle
38. pinky_dip_angle
39. pinky_overall_angle
```

### Hand Span Metrics (4)
```
40. hand_bbox_width
41. hand_bbox_height
42. hand_aspect_ratio
43. max_reach_from_wrist
```

### Relative Positions (6)
```
44. wrist_relative_y
45. thumb_tip_relative_x
46. thumb_tip_relative_y
47. index_tip_relative_x
48. index_tip_relative_y
49. pinky_tip_relative_x
```

## Usage Examples

### Example 1: Extract with Breakdown
```python
result = extractor.extract(landmarks, return_dict=True)

print(f"Distances: {result['distances'].shape}")     # (21,)
print(f"Angles: {result['angles'].shape}")           # (15,)
print(f"Spans: {result['spans'].shape}")             # (4,)
print(f"Positions: {result['positions'].shape}")     # (6,)
print(f"Hand span: {result['hand_span']}")           # scalar
```

### Example 2: Get Feature Names
```python
names = extractor.get_feature_names()
for i, name in enumerate(names, 1):
    print(f"{i:2d}. {name}")
```

### Example 3: Process Video
```python
from examples_feature_extraction import RealtimeFeatureExtractor

extractor = RealtimeFeatureExtractor()
results = extractor.process_video("video.mp4", display=True)

# Extract feature matrix
features = np.array([f for f, _ in results if f is not None])
print(f"Shape: {features.shape}")  # (n_frames, 46)
```

## Common Gesture Patterns

### Peace Sign
- **Distances:** Index & middle wrist-to-tip HIGH
- **Angles:** Index & middle ~180°, others <120°
- **Spans:** Medium width
- **Key feature:** Two high angles, three low angles

### Thumbs Up
- **Distances:** Thumb wrist-to-tip HIGH, others LOW
- **Angles:** Thumb ~180°, others ~90°
- **Positions:** Thumb tip at high Y coordinate
- **Key feature:** High thumb angle, low other angles

### Fist
- **Distances:** All LOW
- **Angles:** All <120°
- **Spans:** Small width/height
- **Key feature:** All features small

### Open Hand
- **Distances:** All HIGH
- **Angles:** All ~180°
- **Spans:** Large width/height
- **Key feature:** All features large

### OK Sign
- **Distances:** Thumb-Index LOW, others HIGH
- **Angles:** Thumb-Index bent, others extended
- **Key feature:** Mixed high and low angles

## Properties

### Scale Invariance ✓
Same gesture at different distances = same feature values

### Rotation Sensitivity
Features capture hand orientation (intentional for some gestures)

### Computation Time
~5-10ms per frame on modern hardware

### Feature Range
- **Distances:** [0, ~0.7]
- **Angles:** [0, 180]°
- **Spans:** Variable
- **Positions:** [0, 1]

## Integration Points

### With Classifiers
```python
X, y = load_features_and_labels()
clf = RandomForestClassifier()
clf.fit(X, y)
y_pred = clf.predict(new_features)
```

### With Neural Networks
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(46,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_gestures, activation='softmax')
])
```

### With Data Storage
```python
# Features are lightweight
features_csv = "features.csv"  # ~184 bytes per frame
features_npy = "features.npy"  # Efficient NumPy format
```

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| All zeros | Hand not detected | Improve lighting, check confidence threshold |
| High variance | Camera too far | Move closer, increase detection confidence |
| NaN values | Invalid landmarks | Validate input, check for hand presence |
| Low accuracy | Insufficient features | Try data augmentation, collect more samples |

## File Locations

```
project/
├── src/
│   └── feature_extractor.py          # Main module
├── examples_feature_extraction.py     # Integration examples
├── tests/
│   └── test_feature_extractor.py     # 30+ unit tests
└── FEATURE_ENGINEERING_GUIDE.md       # Full documentation
```

## Testing

Run unit tests:
```bash
pytest tests/test_feature_extractor.py -v
```

Run demo:
```bash
python examples_feature_extraction.py
```
