# Data Collection Guide

Complete guide for collecting hand gesture data for training ML models.

## Quick Start

### 1. Run the Data Collection Script

```bash
python data_collection.py
```

### 2. Select a Gesture (Press 1-9)

Available gestures:
- **1** = Thumbs Up
- **2** = Peace
- **3** = OK
- **4** = Fist
- **5** = Open Hand
- **6** = Point
- **7** = Rock
- **8** = Love
- **9** = Custom

### 3. Record Samples

- **SPACE** = Start/Stop recording
- **S** = Show statistics
- **R** = Reset gesture
- **Q** = Quit

### 4. Visualize Data

```bash
python visualize_data.py
```

---

## Detailed Workflow

### Step 1: Start Collection

```bash
python data_collection.py
```

You'll see:
- Live webcam feed
- Hand landmarks drawn in real-time
- Info panel showing gesture and recording status
- Keyboard controls at the bottom

### Step 2: Select a Gesture

Press **1-9** to select a gesture:

```
Press '1' for Thumbs Up
Press '2' for Peace
Press '3' for OK
...
```

Console will confirm: `✓ Selected gesture: thumbs_up`

### Step 3: Position Your Hand

Make sure:
- Your entire hand is visible
- Good lighting on your hand
- Camera is 1-2 feet away
- You see the green hand skeleton on screen

### Step 4: Record a Sample

1. Press **SPACE** to start recording
   - Console: `▶ Recording thumbs_up... (press SPACE to stop)`
   - Red "REC" indicator appears on screen

2. Slowly move your hand (2-5 seconds)
   - Frame counter shows: `REC: 45 frames`
   - Green lines show hand skeleton
   - Red dots show finger joints

3. Press **SPACE** to stop recording
   - Console: `✓ Saved thumbs_up sample #0`
   - Sample counter increments

### Step 5: Collect Multiple Samples

Repeat step 4 to collect more samples:
- Different hand positions
- Different speeds of movement
- From different angles

Aim for 20-30 samples per gesture for good training data.

### Step 6: Switch to Another Gesture

Press a different number (1-9) and repeat steps 3-5.

### Step 7: Check Statistics

Press **S** to see collection progress:

```
------------------------------------------
Collection Statistics:
  Total samples: 15
  Current gesture: thumbs_up

  Samples per gesture:
    thumbs_up: 5
    peace: 10
------------------------------------------
```

### Step 8: Exit and Save

Press **Q** to quit. Data is automatically saved.

---

## Data Organization

Data is saved in `data/collected_gestures/`:

```
data/collected_gestures/
├── thumbs_up/
│   ├── sample_00000.json
│   ├── sample_00001.json
│   └── sample_00002.json
├── peace/
│   ├── sample_00000.json
│   ├── sample_00001.json
│   └── ...
└── ok/
    └── ...
```

Each JSON file contains:
- 21 hand landmarks per frame
- Handedness (Left/Right)
- Frame count
- Timestamp
- Metadata

### Sample File Structure

```json
{
  "gesture": "thumbs_up",
  "sample_num": 0,
  "timestamp": "2026-01-20T10:30:45.123456",
  "frame_count": 45,
  "frames": [
    {
      "landmarks": [
        [0.5, 0.3],
        [0.52, 0.25],
        ...
      ],
      "handedness": "Right"
    },
    ...
  ],
  "metadata": {
    "detector_version": "1.0",
    "confidence": 0.7
  }
}
```

---

## Keyboard Controls Summary

| Key | Action |
|-----|--------|
| **1-9** | Select gesture class |
| **SPACE** | Start/Stop recording |
| **R** | Reset current gesture samples |
| **S** | Show statistics |
| **Q** | Quit and save |

---

## Visualizing Collected Data

### View Gesture Summary

```bash
python visualize_data.py

# In the visualizer:
v thumbs_up
```

Shows all samples from a gesture in a grid.

### Replay a Single Sample

```bash
r thumbs_up 0
```

- **Q** = Stop replay
- **SPACE** = Pause/Resume
- Playback speed adjustable (1-100)

### Compare Multiple Samples

```bash
c peace 0 1 2
```

Shows 3 samples side-by-side with frame-by-frame comparison.

### View Statistics

```bash
s
```

Shows total samples and breakdown by gesture.

---

## Data Collection Tips

### Getting Good Samples

1. **Lighting**: Ensure good lighting on your hands
2. **Distance**: Keep hands 1-2 feet from camera
3. **Visibility**: Make sure entire hand is visible
4. **Variation**: Collect from different angles
5. **Movement**: Perform gesture slowly for more frames
6. **Background**: Use plain background for better detection

### Sample Guidelines

**Minimum**: 10 samples per gesture  
**Recommended**: 20-30 samples per gesture  
**Ideal**: 50+ samples per gesture  

More samples = better model training

### Gesture Performance

- **Easy gestures**: Thumbs up, peace, open hand
- **Medium gestures**: OK, point, fist
- **Hard gestures**: Rock, love, custom

Collect more samples for harder gestures.

### Hand Variations

Collect different variations:
- Left hand and right hand
- Different hand sizes
- Different speeds of movement
- Different distances from camera
- Different lighting conditions

---

## Exporting Data for Training

After collecting data, you can export it in different formats:

### Export to CSV

```python
from src.data_utils import GestureDataLoader

loader = GestureDataLoader()
loader.export_to_csv("data/landmarks.csv")
```

Creates a CSV file with:
- One row per gesture sample
- Columns for gesture class and landmark coordinates
- Ready for scikit-learn or pandas

### Export to NumPy

```python
loader.export_to_numpy("data/landmarks.npz")
```

Creates `.npz` file with:
- `X`: Feature matrix (n_samples, 42)
- `y`: Labels (n_samples,)
- `gesture_names`: Class names
- `gesture_ids`: Class IDs

Ready for TensorFlow, PyTorch, or scikit-learn.

### Get Feature Vectors

```python
X, y, gesture_names = loader.get_feature_vectors(
    aggregate_frames="first"
)

print(f"Features shape: {X.shape}")  # (n_samples, 42)
print(f"Labels shape: {y.shape}")    # (n_samples,)
print(f"Gestures: {gesture_names}")  # ['ok', 'peace', 'thumbs_up']
```

Aggregate methods:
- `"first"`: Use first frame only
- `"mean"`: Average across frames
- `"flatten"`: Concatenate all frames

---

## Data Augmentation

Increase training data without collecting more samples:

```python
from src.data_utils import augment_landmarks

landmarks = np.array(...)  # (21, 2)

augmented = augment_landmarks(
    landmarks,
    flip=True,      # Horizontal flip
    rotate=True,    # ±10° rotations
    scale=True      # ±10% scaling
)

# Returns list of 7 variations from 1 sample
```

---

## Troubleshooting

### No hand detected

- **Problem**: "No hand detected" message
- **Solution**: 
  - Ensure good lighting
  - Move hand closer to camera
  - Ensure entire hand is visible
  - Check camera is working

### Poor detection quality

- **Problem**: Landmarks jump around
- **Solution**:
  - Improve lighting
  - Reduce background clutter
  - Keep hand more still when starting

### Can't select gesture

- **Problem**: Pressing 1-9 doesn't work
- **Solution**:
  - Click on the video window first
  - Make sure webcam is active

### Samples not saving

- **Problem**: Error message when trying to save
- **Solution**:
  - Check write permissions to `data/` folder
  - Ensure disk space available
  - Try creating `data/collected_gestures/` manually

### Slow performance

- **Problem**: Low FPS or laggy
- **Solution**:
  - Close other applications
  - Reduce other processes
  - Try lower camera resolution

---

## Next Steps

### 1. Collect Sufficient Data

- Aim for 20-30 samples per gesture
- Multiple hand variations
- Different lighting/distances

### 2. Export Data

```python
from src.data_utils import GestureDataLoader

loader = GestureDataLoader()
X, y, names = loader.get_feature_vectors()
```

### 3. Train a Classifier

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
accuracy = model.score(X, y)
```

### 4. Evaluate Performance

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

---

## Example Workflow

```bash
# 1. Start collection
python data_collection.py

# In the script:
# Press 1 for thumbs_up
# SPACE to start, move hand, SPACE to stop
# Repeat 20-30 times
# Press 2 for peace
# Repeat for each gesture
# Press Q to exit

# 2. Visualize data
python visualize_data.py
# Command: v thumbs_up
# Command: r peace 0
# Command: c ok 0 1 2

# 3. Export for training
python -c "
from src.data_utils import GestureDataLoader
loader = GestureDataLoader()
X, y, names = loader.get_feature_vectors()
print(f'Collected {len(X)} samples')
print(f'Classes: {names}')
"

# 4. Train model
python train_gesture_model.py
```

---

## File Locations

| File | Purpose |
|------|---------|
| `data_collection.py` | Main collection script |
| `visualize_data.py` | View and replay samples |
| `src/data_utils.py` | Data loading utilities |
| `data/collected_gestures/` | Saved gesture data |

---

## Questions?

Refer to these files for more information:
- [HAND_LANDMARK_README.md](HAND_LANDMARK_README.md) - Hand detection details
- [HAND_LANDMARK_API.md](HAND_LANDMARK_API.md) - API reference
- [src/data_utils.py](src/data_utils.py) - Source code with examples

