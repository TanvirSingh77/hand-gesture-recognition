# Data Collection - Quick Start

Get started collecting hand gesture data in 3 minutes.

## Installation

```bash
pip install -r requirements.txt
```

## 1. Start Collection (1 minute)

```bash
python data_collection.py
```

You'll see a live webcam feed with hand landmarks and a control panel.

## 2. Select a Gesture (10 seconds)

Press a number to select:

```
1 = Thumbs Up
2 = Peace
3 = OK
4 = Fist
5 = Open Hand
6 = Point
7 = Rock
8 = Love
9 = Custom
```

Console shows: `✓ Selected gesture: thumbs_up`

## 3. Record Samples (2 minutes)

1. **Position your hand** in the camera view (you'll see green skeleton)
2. **Press SPACE** to start recording
3. **Move your hand slowly** for 2-5 seconds
4. **Press SPACE** to stop
5. **Repeat** 20-30 times for best results

Console shows:
- `▶ Recording thumbs_up...`
- `✓ Saved thumbs_up sample #0`

## 4. Collect More Gestures

1. **Press 2** to switch to Peace
2. **Repeat steps 3-4** for each gesture

## 5. Exit

Press **Q** to quit. Data automatically saves to `data/collected_gestures/`

## Statistics

Press **S** to see collection progress:

```
Total samples: 45
thumbs_up: 15
peace: 15
ok: 15
```

## View Your Data

After collection:

```bash
python visualize_data.py

# Then:
v thumbs_up      # View all samples
r peace 0        # Replay specific sample
s                # Show statistics
```

## Export for Training

```python
from src.data_utils import GestureDataLoader

loader = GestureDataLoader()
X, y, names = loader.get_feature_vectors()
print(f"Ready to train with {len(X)} samples")
```

## Train a Model

```bash
python train_examples.py

# Select option 3 for Random Forest
# or option 5 for Neural Network
```

---

## Commands Cheat Sheet

### Data Collection
| Key | Action |
|-----|--------|
| 1-9 | Select gesture |
| SPACE | Start/Stop record |
| S | Show stats |
| R | Reset |
| Q | Quit |

### Visualization
```
v <gesture>     - View gesture
r <gesture> <#> - Replay sample
c <gesture> <#> - Compare samples
s               - Statistics
q               - Quit
```

---

## Troubleshooting

**No hand detected?**
- Check lighting
- Move closer to camera
- Make sure whole hand is visible

**Slow performance?**
- Close other apps
- Try lower resolution

**Data not saving?**
- Check folder permissions
- Ensure disk space

---

## Next: Train & Use

After collecting enough data (~50 samples minimum):

```python
from src.data_utils import GestureDataLoader
from sklearn.ensemble import RandomForestClassifier

loader = GestureDataLoader()
X, y, names = loader.get_feature_vectors()

model = RandomForestClassifier()
model.fit(X, y)

print(f"Model accuracy: {model.score(X, y):.2%}")
```

---

## Tips

✓ Collect 20-30 samples per gesture minimum  
✓ Use different hand angles  
✓ Include both left and right hands  
✓ Good lighting is important  
✓ Smooth, slow movements work best  

---

**Ready?** Run: `python data_collection.py`
