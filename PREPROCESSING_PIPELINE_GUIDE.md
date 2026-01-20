# Preprocessing Pipeline - Complete Guide

## Overview

The preprocessing pipeline provides a complete, reproducible workflow for preparing hand gesture data for machine learning:

1. **Load** raw landmark data from JSON files
2. **Extract** features using the feature engineering module
3. **Normalize** features for ML models
4. **Split** data into training, validation, and test sets
5. **Save** processed datasets with full metadata

All steps are fully deterministic and reproducible with fixed random seeds.

---

## Key Features

### ✓ Reproducibility
- Fixed random seeds for identical results across runs
- Stratified splitting maintains class distribution
- Complete metadata tracking

### ✓ Comprehensive Data Loading
- Loads from `data/collected_gestures` directory structure
- Handles JSON format from data collection module
- Validates landmark data automatically

### ✓ Integrated Feature Engineering
- Uses HandGestureFeatureExtractor automatically
- Converts 21-point landmarks to 46 features
- Handles frame-level data efficiently

### ✓ Flexible Normalization
- StandardScaler (default): Zero mean, unit variance
- MinMaxScaler: Range [0, 1]
- Fitted scaler saved for applying to new data

### ✓ Smart Data Splitting
- Stratified splitting preserves class distribution
- Configurable train/val/test ratios
- No data leakage between splits

### ✓ Full Metadata Tracking
- Stores preprocessing configuration
- Saves scaler parameters
- Tracks data splits and class distribution
- Records processing timestamp

---

## Architecture

```
PreprocessingPipeline
├── load_raw_data()           # Load from JSON files
├── apply_feature_engineering() # Extract features
├── normalize_features()       # Standardize/MinMax
├── split_data()               # Train/val/test split
├── create_metadata()          # Track everything
└── save_datasets()            # Persist to disk

PreprocessingConfig
├── Data paths
├── Random seed
├── Split ratios
├── Normalization method
└── Filtering thresholds

PreprocessingMetadata
├── Data shapes and names
├── Split information
├── Scaler parameters
└── Processing details
```

---

## Quick Start

### Basic Usage

```python
from src.preprocessing import PreprocessingPipeline

# Create pipeline
pipeline = PreprocessingPipeline(
    data_dir="data/collected_gestures",
    random_seed=42
)

# Process data
X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()

# Save
pipeline.save_datasets("processed_data")
```

### Custom Configuration

```python
from src.preprocessing import PreprocessingConfig, PreprocessingPipeline

# Configure
config = PreprocessingConfig(
    data_dir="data/collected_gestures",
    output_dir="processed_data",
    random_seed=42,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    normalize_method="standard",  # or "minmax"
    verbose=True
)

# Run
pipeline = PreprocessingPipeline(config)
X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()
```

### Using Kwargs

```python
# Pass config as kwargs
pipeline = PreprocessingPipeline(
    data_dir="data/collected_gestures",
    random_seed=42,
    train_split=0.8,
    verbose=True
)
```

---

## Configuration Options

### PreprocessingConfig Parameters

```python
class PreprocessingConfig:
    # Data paths
    data_dir: str = "data/collected_gestures"
    output_dir: str = "processed_data"
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Train/validation/test split (must sum to 1.0)
    train_split: float = 0.7      # 70% training
    val_split: float = 0.15       # 15% validation
    test_split: float = 0.15      # 15% test
    
    # Feature normalization
    normalize_method: str = "standard"  # "standard" or "minmax"
    
    # Feature aggregation (from feature extractor)
    aggregation_method: str = "mean"    # "first", "mean", "flatten"
    
    # Data filtering
    min_samples_per_gesture: int = 5    # Skip gestures with fewer samples
    
    # Verbosity
    verbose: bool = True
```

---

## Detailed Workflow

### Step 1: Load Raw Data

```python
X_raw, y_raw, gesture_names = pipeline._load_raw_data()
# X_raw shape: (n_samples, 21, 2) - raw landmarks
# y_raw shape: (n_samples,) - gesture class indices
# gesture_names: list of class names
```

**Handles:**
- Reading JSON files from gesture directories
- Validating landmark shapes (21 points, 2D)
- Filtering by minimum samples per gesture
- Creating label indices

### Step 2: Apply Feature Engineering

```python
X_features = pipeline._apply_feature_engineering(X_raw)
# X_features shape: (n_samples, 46) - extracted features
```

**Extracts:**
- 21 inter-joint distances (normalized)
- 15 joint angles (degrees)
- 4 hand span metrics
- 6 relative positions

### Step 3: Normalize Features

```python
X_normalized = pipeline._normalize_features(X_features, fit=True)
# X_normalized shape: (n_samples, 46) - normalized features
```

**Methods:**

**StandardScaler (default):**
```
x_normalized = (x - mean) / std
Result: Mean ≈ 0, Std ≈ 1
```

**MinMaxScaler:**
```
x_normalized = (x - min) / (max - min)
Result: Range [0, 1]
```

### Step 4: Split Data

```python
pipeline._split_data(X_normalized, y_raw, gesture_names)
# Sets: X_train, X_val, X_test
#       y_train, y_val, y_test
```

**Ensures:**
- Stratified split maintains class distribution
- No overlap between splits
- Correct proportions (70%, 15%, 15%)

### Step 5: Create Metadata

```python
pipeline._create_metadata(gesture_names)
# Creates PreprocessingMetadata with:
# - Data shapes
# - Split information
# - Scaler parameters
# - Class names and counts
```

---

## Output Format

### Saved Files

When calling `pipeline.save_datasets("output_dir")`:

```
output_dir/
├── X_train.npy              # (n_train, 46) float32
├── X_val.npy                # (n_val, 46) float32 [optional]
├── X_test.npy               # (n_test, 46) float32 [optional]
├── y_train.npy              # (n_train,) int64
├── y_val.npy                # (n_val,) int64 [optional]
├── y_test.npy               # (n_test,) int64 [optional]
├── scaler.pkl               # Fitted scaler object
├── metadata.json            # Complete preprocessing metadata
└── config.json              # Configuration used
```

### Metadata JSON Structure

```json
{
  "n_samples": 1500,
  "n_features": 46,
  "n_gestures": 5,
  "gesture_names": ["peace", "ok", "thumbs_up", "rock", "love"],
  "train_size": 1050,
  "val_size": 225,
  "test_size": 225,
  "normalize_method": "standard",
  "scaler_mean": [...],      // Mean of each feature
  "scaler_scale": [...],     // Std of each feature
  "feature_names": ["wrist_to_thumb_tip", ...],
  "aggregation_method": "mean",
  "processed_timestamp": "2024-01-20T12:34:56.789",
  "random_seed": 42,
  "samples_per_gesture": {
    "peace": 300,
    "ok": 300,
    ...
  }
}
```

---

## Reproducibility

### Same Seed = Identical Results

```python
# Run 1
pipeline1 = PreprocessingPipeline(random_seed=42)
X1, y1 = pipeline1.process()[:4]

# Run 2
pipeline2 = PreprocessingPipeline(random_seed=42)
X2, y2 = pipeline2.process()[:4]

# X1 == X2 and y1 == y2 (byte-for-byte identical)
```

### Different Seed = Different Split

```python
# Different seed produces different train/val/test split
pipeline3 = PreprocessingPipeline(random_seed=123)
X3, y3 = pipeline3.process()[:4]

# X1 != X3 and y1 != y3 (different samples in training set)
```

**Why Important:**
- Compare models fairly (same data)
- Reproduce research results
- Debug issues systematically
- Share datasets reliably

---

## Loading and Using Saved Data

### Load All Datasets

```python
X_train, X_val, X_test, y_train, y_val, y_test, metadata = (
    PreprocessingPipeline.load_datasets("processed_data")
)

print(f"Training shape: {X_train.shape}")
print(f"Classes: {metadata.gesture_names}")
```

### Load Scaler Only

```python
scaler = PreprocessingPipeline.load_scaler("processed_data")

# Apply to new features
new_features_normalized = scaler.transform(new_features)
```

### Access Metadata

```python
metadata = PreprocessingPipeline.load_datasets("processed_data")[6]

print(f"Total samples: {metadata.n_samples}")
print(f"Features: {metadata.n_features}")
print(f"Classes: {metadata.gesture_names}")
print(f"Scaler mean: {metadata.scaler_mean[:5]}")
```

---

## Usage Examples

### Example 1: Train a Model

```python
from sklearn.ensemble import RandomForestClassifier

# Preprocess
pipeline = PreprocessingPipeline(random_seed=42)
X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()

# Train
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate
score = clf.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")
```

### Example 2: Different Splits

```python
# More aggressive training (90% train, no test)
pipeline = PreprocessingPipeline(
    train_split=0.9,
    val_split=0.1,
    test_split=0.0
)
X_train, X_val, _, y_train, y_val, _, metadata = pipeline.process()
```

### Example 3: Custom Normalization

```python
# Use MinMax normalization instead
pipeline = PreprocessingPipeline(
    normalize_method="minmax",
    random_seed=42
)
X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()

# Features now in range [0, 1]
print(f"Min: {X_train.min():.4f}, Max: {X_train.max():.4f}")
```

### Example 4: Inference with Saved Preprocessing

```python
# Training phase
pipeline = PreprocessingPipeline(random_seed=42)
X_train, _, _, y_train, _, _, _ = pipeline.process()
pipeline.save_datasets()

# Inference phase (later)
from src.feature_extractor import HandGestureFeatureExtractor

extractor = HandGestureFeatureExtractor()
scaler = PreprocessingPipeline.load_scaler("processed_data")

# Process new frame
frame = capture_frame()
landmarks = detector.detect(frame)
features = extractor.extract(landmarks)  # (46,)
features_normalized = scaler.transform([features])  # (1, 46)

# Use with model
prediction = model.predict(features_normalized)
```

---

## Data Splits Explained

### Stratified Splitting

The pipeline uses stratified splitting from scikit-learn to ensure:

```
Original distribution:
  Class A: 30%
  Class B: 50%
  Class C: 20%

Training set (70%):
  Class A: 30%
  Class B: 50%
  Class C: 20%

Validation set (15%):
  Class A: 30%
  Class B: 50%
  Class C: 20%

Test set (15%):
  Class A: 30%
  Class B: 50%
  Class C: 20%
```

**Benefits:**
- Balanced evaluation metrics
- Reliable cross-dataset comparison
- Prevents biased performance estimates

---

## Normalization Methods

### StandardScaler (Default)

```
z = (x - mean(X)) / std(X)

Properties:
- Center at 0
- Scale to unit variance
- Suitable for: Normal distributions
- Algorithm preference: Most ML algorithms

After normalization:
- Mean ≈ 0
- Std ≈ 1
- Range ≈ [-3, 3]
```

### MinMaxScaler

```
x_norm = (x - min(X)) / (max(X) - min(X))

Properties:
- Range [0, 1]
- Preserves original distribution shape
- Suitable for: Bounded data, neural networks
- More stable: Unaffected by outliers less

After normalization:
- Min = 0
- Max = 1
- All values in [0, 1]
```

---

## Common Patterns

### Pattern 1: Separate Training and Testing

```python
# Save training data only (for model development)
pipeline = PreprocessingPipeline(
    train_split=1.0,  # All training
    val_split=0.0,
    test_split=0.0
)
X, _, _, y, _, _, _ = pipeline.process()
pipeline.save_datasets("training_only")

# Later: Load and use
X_all, _, _, y_all, _, _, _ = PreprocessingPipeline.load_datasets("training_only")
```

### Pattern 2: Cross-Validation Ready

```python
# Get one split for stratification check
pipeline = PreprocessingPipeline(random_seed=42)
X_train, X_val, X_test, y_train, y_val, y_test, _ = pipeline.process()

# Use all data for cross-validation
X_all = np.vstack([X_train, X_val, X_test])
y_all = np.hstack([y_train, y_val, y_test])

# sklearn cross-validation handles splits
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_all, y_all, cv=5)
```

### Pattern 3: Reproducible Experiments

```python
# Document configuration
config = PreprocessingConfig(
    random_seed=42,
    train_split=0.8,
    normalize_method="standard"
)

# Save config
pipeline = PreprocessingPipeline(config)
pipeline.process()
pipeline.save_datasets()

# Later: Reproduce exactly
loaded_config = json.load(open("processed_data/config.json"))
pipeline2 = PreprocessingPipeline(**loaded_config)
```

---

## Troubleshooting

### Issue: "No gesture directories found"

**Cause:** Data directory empty or wrong path

**Solution:**
```python
# Check path
pipeline = PreprocessingPipeline(
    data_dir="data/collected_gestures",
    verbose=True  # Shows loading progress
)

# Or verify manually
import os
print(os.listdir("data/collected_gestures"))
```

### Issue: "No valid samples loaded"

**Cause:** Files are empty or have wrong format

**Solution:**
```python
# Check file format
import json
with open("data/collected_gestures/peace/sample_00000.json") as f:
    data = json.load(f)
    print(data.keys())  # Should have 'landmarks' key
    print(np.array(data['landmarks']).shape)  # Should be (n_frames, 21, 2)
```

### Issue: Features have NaN or Inf values

**Cause:** Invalid landmarks (degenerate hand poses)

**Solution:**
```python
# Check for NaN in loaded data
X_train, _, _, _, _, _, _ = pipeline.process()
print(f"NaN values: {np.isnan(X_train).sum()}")
print(f"Inf values: {np.isinf(X_train).sum()}")

# Filter out invalid samples in data collection
```

### Issue: Different results with different runs

**Cause:** Random seed not set or different seed used

**Solution:**
```python
# Always specify random_seed
pipeline = PreprocessingPipeline(
    random_seed=42  # IMPORTANT: Explicit seed
)
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Data loading time | 1-5 seconds (depends on data size) |
| Feature engineering | 5-20ms per sample |
| Total preprocessing | 10-60 seconds for typical dataset |
| Memory usage | ~1-2GB for typical dataset |
| Saved dataset size | ~100MB for 10K samples, 46 features |

---

## Best Practices

### ✓ Always Use Fixed Seeds

```python
# Good
pipeline = PreprocessingPipeline(random_seed=42)

# Avoid
pipeline = PreprocessingPipeline()  # Uses random seed each time
```

### ✓ Save Preprocessed Data

```python
# Always save after preprocessing
X_train, X_val, X_test, y_train, y_val, y_test, _ = pipeline.process()
pipeline.save_datasets("processed_data")

# Reuse instead of reprocessing
X_train, _, _, y_train, _, _, _ = PreprocessingPipeline.load_datasets("processed_data")
```

### ✓ Document Configuration

```python
# Save configuration with results
config = PreprocessingConfig(random_seed=42, ...)
pipeline = PreprocessingPipeline(config)
pipeline.process()
pipeline.save_datasets()  # Saves config.json automatically
```

### ✓ Validate Data After Preprocessing

```python
# Check for issues
X_train, _, _, y_train, _, _, _ = pipeline.process()
assert not np.isnan(X_train).any()  # No NaN
assert not np.isinf(X_train).any()  # No Inf
assert X_train.min() >= -10  # Reasonable range
assert X_train.max() <= 10
```

---

## Integration with ML Pipeline

```python
# Complete workflow
from src.preprocessing import PreprocessingPipeline
from src.feature_extractor import HandGestureFeatureExtractor
from sklearn.ensemble import RandomForestClassifier

# 1. Preprocess
pipeline = PreprocessingPipeline(random_seed=42)
X_train, X_val, _, y_train, y_val, _, metadata = pipeline.process()
pipeline.save_datasets()

# 2. Train
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 3. Evaluate
train_score = clf.score(X_train, y_train)
val_score = clf.score(X_val, y_val)

print(f"Training accuracy: {train_score:.4f}")
print(f"Validation accuracy: {val_score:.4f}")

# 4. Deploy
scaler = PreprocessingPipeline.load_scaler("processed_data")
# Use scaler.transform() on new features before prediction
```

---

## API Reference

### PreprocessingPipeline

```python
process() -> Tuple
    Execute complete preprocessing pipeline
    Returns: (X_train, X_val, X_test, y_train, y_val, y_test, metadata)

save_datasets(output_dir) -> str
    Save preprocessed datasets to disk
    Returns: Path to output directory

load_datasets(output_dir) -> Tuple [static]
    Load previously saved datasets
    Returns: (X_train, X_val, X_test, y_train, y_val, y_test, metadata)

load_scaler(output_dir) [static]
    Load fitted scaler
    Returns: Fitted scaler object
```

---

## Version History

**v1.0.0 - Initial Release**
- Complete preprocessing pipeline
- Stratified train/val/test splitting
- StandardScaler and MinMaxScaler support
- Metadata tracking and saving
- Full reproducibility with random seeds
- Comprehensive documentation

---

## Next Steps

1. **Collect data** using `data_collection.py`
2. **Preprocess** using this module
3. **Train models** using preprocessed data
4. **Deploy** with saved scaler for inference

Example:
```bash
# Step 1: Collect
python data_collection.py

# Step 2: Preprocess
python examples_preprocessing_pipeline.py

# Step 3: Train
python train_examples.py

# Step 4: Deploy
# Use in gesture_classifier.py
```
