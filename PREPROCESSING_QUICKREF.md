# Preprocessing Pipeline - Quick Reference

## 30-Second Overview

```python
from src.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(random_seed=42)
X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()
pipeline.save_datasets("processed_data")
```

**What it does:**
- Loads raw gesture landmarks from JSON files
- Extracts 46 features per sample
- Normalizes features
- Splits into training/validation/test sets
- Saves with reproducibility metadata

---

## Key Concepts

### Reproducibility
```python
# Same seed = identical results
pipeline1 = PreprocessingPipeline(random_seed=42)
X1, y1 = pipeline1.process()[:4]

pipeline2 = PreprocessingPipeline(random_seed=42)
X2, y2 = pipeline2.process()[:4]

# X1 == X2 and y1 == y2
```

### Configuration
```python
from src.preprocessing import PreprocessingConfig

config = PreprocessingConfig(
    data_dir="data/collected_gestures",
    random_seed=42,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    normalize_method="standard"  # or "minmax"
)

pipeline = PreprocessingPipeline(config)
```

### Default Splits
- **Training:** 70% (model learns from this)
- **Validation:** 15% (tune hyperparameters)
- **Test:** 15% (final evaluation)

---

## Common Tasks

### Task 1: Basic Preprocessing
```python
pipeline = PreprocessingPipeline(random_seed=42)
X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()
pipeline.save_datasets()
```

### Task 2: Custom Splits
```python
pipeline = PreprocessingPipeline(
    train_split=0.8,
    val_split=0.1,
    test_split=0.1
)
```

### Task 3: Different Normalization
```python
pipeline = PreprocessingPipeline(
    normalize_method="minmax"  # StandardScaler by default
)
```

### Task 4: Load Saved Data
```python
X_train, X_val, X_test, y_train, y_val, y_test, metadata = \
    PreprocessingPipeline.load_datasets("processed_data")
```

### Task 5: Apply Preprocessing to New Data
```python
scaler = PreprocessingPipeline.load_scaler("processed_data")
new_features_normalized = scaler.transform(new_features)
```

### Task 6: Train Model
```python
from sklearn.ensemble import RandomForestClassifier

X_train, X_val, _, y_train, y_val, _, _ = pipeline.process()

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(f"Validation accuracy: {clf.score(X_val, y_val):.4f}")
```

### Task 7: Check Data Stats
```python
pipeline = PreprocessingPipeline()
_, _, _, y_train, _, _, metadata = pipeline.process()

print(f"Classes: {metadata.gesture_names}")
print(f"Training samples: {metadata.train_size}")
print(f"Samples per gesture: {metadata.samples_per_gesture}")
```

---

## Data Flow

```
Raw Gesture Data (JSON)
         ↓
    [Load Data]
    21 landmarks × frames
         ↓
[Feature Engineering]
    46 features
         ↓
 [Normalization]
StandardScaler or MinMaxScaler
         ↓
 [Data Splitting]
Train (70%) / Val (15%) / Test (15%)
         ↓
  [Saved Datasets]
.npy files + scaler + metadata
```

---

## Output Format

### Saved Files
```
processed_data/
├── X_train.npy      # (n_train, 46) float32
├── X_val.npy        # (n_val, 46) float32
├── X_test.npy       # (n_test, 46) float32
├── y_train.npy      # (n_train,) int64
├── y_val.npy        # (n_val,) int64
├── y_test.npy       # (n_test,) int64
├── scaler.pkl       # Fitted scaler
├── metadata.json    # Complete info
└── config.json      # Configuration
```

### Data Shapes
- `X_train`: (1000, 46) = 1000 samples, 46 features
- `y_train`: (1000,) = 1000 labels (class indices 0-4)
- `metadata.gesture_names`: ["peace", "ok", "thumbs_up", ...]

---

## Normalization Methods

### StandardScaler (Default)
```
x_normalized = (x - mean) / std

Result: Mean ≈ 0, Std ≈ 1, Range ≈ [-3, 3]
Best for: Most ML algorithms
```

### MinMaxScaler
```
x_normalized = (x - min) / (max - min)

Result: Min = 0, Max = 1, Range [0, 1]
Best for: Neural networks, bounded data
```

---

## Configuration Parameters

```python
PreprocessingConfig(
    data_dir="data/collected_gestures",    # Input directory
    output_dir="processed_data",            # Output directory
    random_seed=42,                         # For reproducibility
    train_split=0.7,                        # Training proportion
    val_split=0.15,                         # Validation proportion
    test_split=0.15,                        # Test proportion
    normalize_method="standard",            # "standard" or "minmax"
    aggregation_method="mean",              # "first", "mean", "flatten"
    min_samples_per_gesture=5,              # Minimum samples per class
    verbose=True                            # Print progress
)
```

---

## Returns from process()

```python
X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()

# X_train: (n_train, 46) - training features
# X_val: (n_val, 46) - validation features
# X_test: (n_test, 46) - test features
# y_train: (n_train,) - training labels
# y_val: (n_val,) - validation labels
# y_test: (n_test,) - test labels
# metadata: PreprocessingMetadata object

# Access metadata
metadata.gesture_names       # ["peace", "ok", "thumbs_up", ...]
metadata.n_features         # 46
metadata.n_gestures         # 5
metadata.train_size         # 1050
metadata.samples_per_gesture # {"peace": 300, ...}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No gesture directories found" | Check data_dir path and directory structure |
| "No valid samples loaded" | Verify JSON format and landmark shapes |
| Different results each run | Add `random_seed=42` parameter |
| Features have NaN values | Check input landmarks are valid (no degenerate poses) |
| Out of memory | Reduce data size or use chunked processing |

---

## Common Patterns

### Pattern 1: Train-Test-Only
```python
pipeline = PreprocessingPipeline(
    train_split=0.8,
    val_split=0.0,
    test_split=0.2
)
```

### Pattern 2: Training-Only (for cross-validation)
```python
pipeline = PreprocessingPipeline(
    train_split=1.0,
    val_split=0.0,
    test_split=0.0
)
```

### Pattern 3: Deterministic Research
```python
pipeline = PreprocessingPipeline(
    random_seed=42,  # Fixed seed
    normalize_method="standard",
    verbose=True  # Track everything
)
```

### Pattern 4: Production Inference
```python
scaler = PreprocessingPipeline.load_scaler("processed_data")
new_normalized = scaler.transform(new_features)
prediction = model.predict(new_normalized)
```

---

## Feature Ranges After Normalization

### StandardScaler
- Mean: 0
- Std: 1
- Typical range: [-3, 3]
- No strict bounds

### MinMaxScaler
- Min: 0
- Max: 1
- Strict range: [0, 1]
- Outliers clipped

---

## Class Distribution

**Stratified split example:**

```
Before split (all data):
  peace: 30% | ok: 25% | thumbs_up: 25% | rock: 10% | love: 10%

After split:
  Training: 70%
    peace: 30% | ok: 25% | thumbs_up: 25% | rock: 10% | love: 10%
  
  Validation: 15%
    peace: 30% | ok: 25% | thumbs_up: 25% | rock: 10% | love: 10%
  
  Test: 15%
    peace: 30% | ok: 25% | thumbs_up: 25% | rock: 10% | love: 10%
```

---

## Performance Notes

| Task | Time | Notes |
|------|------|-------|
| Load 1000 samples | 1-2s | Includes JSON parsing |
| Feature extraction | 50-100ms | 46 features per sample |
| Normalization | <100ms | Fit + transform |
| Total preprocessing | 5-30s | Depends on data size |

---

## File Size Reference

```
1000 samples × 46 features × 4 bytes (float32)
= 184 KB per dataset

Typical project:
  X_train (70%): ~128 KB
  X_val (15%): ~27 KB
  X_test (15%): ~27 KB
  Metadata + scaler: ~50 KB
  Total: ~230 KB
```

---

## Testing

### Run tests
```bash
pytest tests/test_preprocessing.py -v
```

### Run examples
```bash
python examples_preprocessing_pipeline.py
```

---

## Integration Checklist

- [ ] Data collected in `data/collected_gestures`
- [ ] Preprocessing pipeline created
- [ ] Random seed specified
- [ ] Datasets saved
- [ ] Scaler loaded for inference
- [ ] Metadata tracked
- [ ] Model training ready

---

## Quick Debugging

```python
# Check if pipeline works
pipeline = PreprocessingPipeline(verbose=True)

# See what's loaded
X_raw, y_raw, names = pipeline._load_raw_data()
print(f"Loaded {len(X_raw)} samples from {len(names)} classes")

# Check features
X_feat = pipeline._apply_feature_engineering(X_raw)
print(f"Features: {X_feat.shape}, NaN: {np.isnan(X_feat).sum()}")

# Check normalization
X_norm = pipeline._normalize_features(X_feat, fit=True)
print(f"After norm - Min: {X_norm.min():.4f}, Max: {X_norm.max():.4f}")

# Full pipeline
result = pipeline.process()
print("✓ All steps successful!")
```

---

## Next Steps

1. **Collect data**: `python data_collection.py`
2. **Preprocess**: Use this module
3. **Train**: Use preprocessed data with ML models
4. **Deploy**: Load scaler and use in inference

---

## API Summary

```python
# Initialize
pipeline = PreprocessingPipeline(random_seed=42, ...)

# Process
X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()

# Save
pipeline.save_datasets("output_dir")

# Load
X_train, _, _, y_train, _, _, metadata = PreprocessingPipeline.load_datasets("output_dir")

# Load scaler
scaler = PreprocessingPipeline.load_scaler("output_dir")
```

Done! Ready for ML pipeline! 🎉
