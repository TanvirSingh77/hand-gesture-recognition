# Neural Network Quick Reference

## One-Page Cheat Sheet for Gesture Classification

### Installation & Setup
```bash
pip install tensorflow>=2.14.0 numpy opencv-python mediapipe

python train_gesture_model.py --architecture balanced --epochs 100
```

### Model Creation & Training
```python
from src.gesture_model import GestureClassificationModel
import numpy as np

# Create model
model = GestureClassificationModel(num_gestures=5, architecture="lightweight")

# Build & compile
model.build(verbose=True)
model.compile(learning_rate=0.001)

# Train
history = model.train(
    X_train, y_train,      # Features (N,46), Labels (N,5) one-hot
    X_val, y_val,
    epochs=100,
    batch_size=32,
    class_weight_strategy="balanced"
)

# Evaluate
metrics = model.evaluate(X_test, y_test)

# Save
model.save_model("models/gesture_classifier.h5")
```

### Making Predictions
```python
# Load model
model = GestureClassificationModel.load_model("models/gesture_classifier.h5")

# Single prediction
pred = model.predict(np.array([features]))  # Shape: (1, 46)
gesture = np.argmax(pred[0])
confidence = pred[0][gesture]

# Batch with confidence
results = model.predict_batch_with_confidence(
    features,  # Shape: (N, 46)
    confidence_threshold=0.5,
    return_top_k=3
)
# results[0] = {
#     "class_id": 2,
#     "confidence": 0.95,
#     "above_threshold": True,
#     "top_k": [{"class_id": 2, "confidence": 0.95}, ...]
# }
```

### Training Script
```bash
# Basic training
python train_gesture_model.py

# Custom configuration
python train_gesture_model.py \
    --architecture powerful \
    --epochs 200 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --demo
```

### Examples
```bash
# Real-time gesture recognition
python examples_gesture_classification.py --mode realtime

# Batch predictions
python examples_gesture_classification.py --mode batch

# Model comparison
python examples_gesture_classification.py --mode comparison

# Feature space analysis
python examples_gesture_classification.py --mode analysis
```

### Testing
```bash
pytest tests/test_gesture_model.py -v
pytest tests/test_gesture_model.py::TestModelTraining -v
```

### Architecture Comparison

| Feature | Lightweight | Balanced | Powerful |
|---------|-----------|----------|---------|
| Hidden Layers | 2 | 3 | 3 |
| Layer Size | 64→32 | 128→64→32 | 256→128→64 |
| Parameters | 18K | 50K | 100K |
| Inference | 2-5ms | 5-10ms | 10-20ms |
| Accuracy | Good | Better | Best |

### Common Patterns

**Prepare Data (46 features, one-hot labels)**
```python
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=5)
y_val = to_categorical(y_val, num_classes=5)
```

**Handle Imbalanced Data**
```python
model.train(
    X_train, y_train,
    X_val, y_val,
    class_weight_strategy="balanced"  # Automatic weighting
)
```

**Train with Early Stopping**
```python
# Early stopping automatically enabled
# Stops if val_loss doesn't improve for 15 epochs
# Saves best model automatically
history = model.train(X_train, y_train, X_val, y_val, epochs=1000)
```

**Real-Time Integration**
```python
from src.hand_landmarks import HandLandmarkDetector
from src.feature_extractor import HandGestureFeatureExtractor
import cv2

detector = HandLandmarkDetector()
extractor = HandGestureFeatureExtractor()
model = GestureClassificationModel.load_model("models/gesture_classifier.h5")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    success, landmarks = detector.detect(frame)
    if success:
        features = extractor.extract(landmarks)
        if features is not None:
            pred = model.predict(np.array([features]))[0]
            gesture = np.argmax(pred)
            print(f"Gesture: {gesture}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
```

### Model Information
```python
info = model.get_model_info()
print(info)
# {
#     "status": "built",
#     "model_name": "gesture_classifier",
#     "architecture": "lightweight",
#     "input_features": 46,
#     "num_gestures": 5,
#     "total_parameters": 18234,
#     "trainable_parameters": 18234,
#     "training_metadata": {...}
# }
```

### Performance Tips

| Problem | Solution |
|---------|----------|
| Slow training | Increase batch_size, use lightweight |
| Poor accuracy | More data, use powerful architecture, more epochs |
| Overfitting | More dropout, less complex, more data |
| Memory issues | Smaller batch_size, lightweight architecture |
| Inference slow | Use lightweight, increase batch_size |

### Error Handling
```python
try:
    model = GestureClassificationModel.load_model("models/model.h5")
except FileNotFoundError:
    print("Model not found. Train first: python train_gesture_model.py")

try:
    predictions = model.predict(features)
except ValueError as e:
    print(f"Feature dimension mismatch: {e}")
```

### File Locations
```
models/gesture_classifier.h5          # Trained model
models/gesture_classifier_metadata.json # Training info
datasets/train_features.npy           # Training data
datasets/train_labels.npy
datasets/val_features.npy
datasets/val_labels.npy
```

### Command Cheatsheet
```bash
# Training
python train_gesture_model.py --help
python train_gesture_model.py --architecture powerful
python train_gesture_model.py --epochs 200 --batch_size 16

# Examples
python examples_gesture_classification.py --mode predict
python examples_gesture_classification.py --mode realtime
python examples_gesture_classification.py --mode batch
python examples_gesture_classification.py --mode comparison

# Testing
pytest tests/test_gesture_model.py -v
pytest tests/ -v
```

### Troubleshooting
```python
# Check model is loaded correctly
try:
    model = GestureClassificationModel.load_model("models/gesture_classifier.h5")
    assert model.model is not None
    print(f"✓ Model loaded: {model.num_gestures} gestures, {model.input_features} features")
except Exception as e:
    print(f"✗ Error loading model: {e}")

# Verify predictions make sense
pred = model.predict(features)
assert pred.shape == (len(features), model.num_gestures)
assert np.allclose(pred.sum(axis=1), 1.0)  # Probabilities sum to 1
print(f"✓ Predictions valid")

# Check data format
assert X_train.shape[1] == 46  # Must have 46 features
assert y_train.shape[1] == num_gestures  # One-hot encoded
print(f"✓ Data format correct")
```

### More Info
- Full docs: [GESTURE_CLASSIFICATION_GUIDE.md](GESTURE_CLASSIFICATION_GUIDE.md)
- Examples: [examples_gesture_classification.py](examples_gesture_classification.py)
- Tests: [tests/test_gesture_model.py](tests/test_gesture_model.py)
- Training: [train_gesture_model.py](train_gesture_model.py)

---

**Quick Start:** `python train_gesture_model.py && python examples_gesture_classification.py --mode realtime`
