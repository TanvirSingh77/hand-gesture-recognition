# Model Evaluation Script Guide

## Overview

The `evaluate_gesture_model.py` script provides comprehensive evaluation of trained gesture classification models with detailed metrics, confusion matrix, and result logging.

---

## Features

✅ **Accuracy Metrics**
- Overall accuracy on validation data
- Per-class accuracy, precision, recall, F1 score
- Detailed classification statistics

✅ **Confusion Matrix**
- Full confusion matrix calculation
- Normalized confusion matrix display
- Text-based matrix visualization
- Optional graphical visualization (PNG)

✅ **Results Logging**
- Console output with detailed metrics
- Text file logging (.txt)
- JSON summary export (.json)
- Confusion matrix plot (requires matplotlib)

✅ **Performance Analysis**
- Per-class performance breakdown
- Best/worst performing gesture classes
- Class distribution analysis
- Sample count tracking

---

## Installation

Evaluation script is included with the project. Optional visualization requires:

```bash
pip install matplotlib seaborn
```

---

## Usage

### Basic Evaluation

```bash
python evaluate_gesture_model.py \
    --model models/gesture_classifier.h5 \
    --data datasets/val_features.npy \
    --labels datasets/val_labels.npy
```

### With Output Logging

```bash
python evaluate_gesture_model.py \
    --model models/gesture_classifier.h5 \
    --data datasets/val_features.npy \
    --labels datasets/val_labels.npy \
    --output eval_results.txt
```

### With Confusion Matrix Plot

```bash
python evaluate_gesture_model.py \
    --model models/gesture_classifier.h5 \
    --data datasets/val_features.npy \
    --labels datasets/val_labels.npy \
    --output eval_results.txt \
    --plot confusion_matrix.png
```

### Help

```bash
python evaluate_gesture_model.py --help
```

---

## Command-Line Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--model` | Yes | - | Path to trained model (.h5 file) |
| `--data` | Yes | - | Path to validation features (.npy file) |
| `--labels` | Yes | - | Path to validation labels (.npy file) |
| `--output` | No | eval_results.txt | Path to save evaluation results |
| `--plot` | No | - | Path to save confusion matrix plot (PNG) |
| `--batch_size` | No | 32 | Batch size for predictions |
| `--verbose` | No | True | Print to console |

---

## Output Files

### 1. eval_results.txt
**Text report with:**
- Model information
- Overall accuracy
- Per-class metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Summary statistics
- Class distribution

**Example:**
```
================================================================================
Model Evaluation Results
Generated: 2024-01-20 15:30:45
================================================================================

================================================================================
EVALUATION METRICS
================================================================================

OVERALL ACCURACY: 0.8756 (87.56%)
Total Samples: 200

PER-CLASS METRICS:
------------------------------------------------------------------------------------------
Class     Accuracy        Precision       Recall          F1 Score        Samples        
------------------------------------------------------------------------------------------
Class 0   0.9000          0.9000          0.9000          0.9000          40             
Class 1   0.8500          0.8500          0.8500          0.8500          40             
Class 2   0.8000          0.8000          0.8000          0.8000          40             
Class 3   0.9000          0.9000          0.9000          0.9000          40             
Class 4   0.8500          0.8500          0.8500          0.8500          40             
------------------------------------------------------------------------------------------

================================================================================
CONFUSION MATRIX
================================================================================

Confusion Matrix:
Actual \ Predicted   Class0    Class1    Class2    Class3    Class4
-------------------------------------------------------------------
       Class0            36         2         1         1         0
       Class1             1        34         2         2         1
       Class2             2         1        32         3         2
       Class3             1         0         2        36         1
       Class4             1         2         2         1        34
```

### 2. eval_results_summary.json
**JSON summary containing:**
```json
{
  "model": "models/gesture_classifier.h5",
  "evaluation_date": "2024-01-20T15:30:45.123456",
  "overall_accuracy": 0.8756,
  "total_samples": 200,
  "per_class_metrics": {
    "0": {
      "accuracy": 0.9,
      "precision": 0.9,
      "recall": 0.9,
      "f1_score": 0.9,
      "samples": 40
    },
    ...
  },
  "confusion_matrix": [[36, 2, 1, 1, 0], ...]
}
```

### 3. confusion_matrix.png (Optional)
Heatmap visualization of confusion matrix with:
- Normalized values (colors)
- Actual counts (numbers)
- Class labels
- Color scale

---

## Evaluation Metrics Explained

### Overall Accuracy
- Percentage of correct predictions
- Formula: `correct_predictions / total_predictions`
- Range: 0.0 to 1.0

### Per-Class Metrics

**Accuracy (Recall)**
- Portion of samples correctly classified for this class
- Formula: `TP / (TP + FN)`
- TP = True Positives, FN = False Negatives

**Precision**
- Accuracy of positive predictions for this class
- Formula: `TP / (TP + FP)`
- FP = False Positives

**Recall**
- Same as Accuracy (TP / (TP + FN))

**F1 Score**
- Harmonic mean of precision and recall
- Formula: `2 * (precision * recall) / (precision + recall)`
- Better for imbalanced datasets

---

## Python API

### Load and Evaluate Programmatically

```python
from evaluate_gesture_model import (
    load_data,
    compute_metrics,
    compute_confusion_matrix,
    format_metrics_report
)
from src.gesture_model import GestureClassificationModel
import numpy as np

# Load model and data
model = GestureClassificationModel.load_model("models/gesture_classifier.h5")
val_features, val_labels = load_data(
    "datasets/val_features.npy",
    "datasets/val_labels.npy"
)

# Make predictions
predictions = model.predict(val_features, batch_size=32)
predicted_classes = np.argmax(predictions, axis=1)

# Compute metrics
metrics = compute_metrics(val_labels, predicted_classes, model.num_gestures)
print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")

# Compute confusion matrix
cm = compute_confusion_matrix(val_labels, predicted_classes, model.num_gestures)
print(f"Confusion Matrix Shape: {cm.shape}")
```

---

## Interpretation Guide

### High Accuracy (>90%)
✅ Model generalizes well to validation data  
✅ Ready for deployment  
✅ Consider production use

### Medium Accuracy (70-90%)
⚠ Model needs improvement  
⚠ Collect more training data  
⚠ Try different architecture
⚠ Tune hyperparameters

### Low Accuracy (<70%)
❌ Model underperforms  
❌ Significant data quality issues likely  
❌ Retrain with better data  
❌ Check feature engineering

### Imbalanced Per-Class Accuracy
⚠ Some classes perform better than others  
⚠ Possible class imbalance in training data  
⚠ Use class weights during training  
⚠ Collect more data for underperforming classes

### Confusion Matrix Analysis

**High diagonal values:**
✅ Model distinguishes classes well

**Off-diagonal patterns:**
- Similar values in row/column: Classes are confused
- Solution: Collect more diverse training data

**Specific class confusion:**
- Pattern in specific row/column: Particular gesture confused with another
- Solution: Improve training data for that gesture

---

## Example Workflow

### 1. Train Model
```bash
python train_gesture_model.py --architecture balanced --epochs 100
```

### 2. Evaluate Model
```bash
python evaluate_gesture_model.py \
    --model models/gesture_classifier.h5 \
    --data datasets/val_features.npy \
    --labels datasets/val_labels.npy \
    --output eval_results.txt \
    --plot confusion_matrix.png
```

### 3. Review Results
```bash
cat eval_results.txt
```

### 4. Analyze JSON Summary
```bash
python -m json.tool eval_results_summary.json
```

### 5. View Confusion Matrix Plot
```bash
# Open in image viewer
open confusion_matrix.png  # macOS
display confusion_matrix.png  # Linux
start confusion_matrix.png  # Windows
```

---

## Integration with Training

After training, evaluation should be part of the workflow:

```python
from src.gesture_model import GestureClassificationModel
from evaluate_gesture_model import load_data, compute_metrics, compute_confusion_matrix
import numpy as np

# Train model
model = GestureClassificationModel(num_gestures=5, architecture="balanced")
model.build()
model.compile()
history = model.train(X_train, y_train, X_val, y_val, epochs=100)
model.save_model("models/gesture_classifier.h5")

# Evaluate
val_features, val_labels = load_data("datasets/val_features.npy", "datasets/val_labels.npy")
predictions = model.predict(val_features)
predicted_classes = np.argmax(predictions, axis=1)

metrics = compute_metrics(val_labels, predicted_classes, model.num_gestures)
cm = compute_confusion_matrix(val_labels, predicted_classes, model.num_gestures)

print(f"Final Accuracy: {metrics['overall_accuracy']:.4f}")
```

---

## Troubleshooting

### "Model file not found"
**Solution:** Train model first
```bash
python train_gesture_model.py
```

### "Data files not found"
**Solution:** Preprocess data first
```bash
python examples_preprocessing_pipeline.py
```

### "No module named 'matplotlib'"
**Solution:** Install optional dependencies
```bash
pip install matplotlib seaborn
```

### "Memory error with large batch size"
**Solution:** Reduce batch size
```bash
python evaluate_gesture_model.py --model ... --batch_size 16
```

### "Low accuracy results"
**Checklist:**
- ✓ Model trained correctly?
- ✓ Validation data is representative?
- ✓ Features normalized properly?
- ✓ Labels are one-hot encoded?
- ✓ Model architecture appropriate?

---

## Advanced Usage

### Custom Metrics Calculation

```python
from evaluate_gesture_model import compute_metrics, compute_confusion_matrix
import numpy as np

# Compute metrics for specific class
metrics = compute_metrics(val_labels, predicted_classes, 5)

# Get F1 score for class 2
f1_class_2 = metrics['per_class'][2]['f1_score']
print(f"Class 2 F1 Score: {f1_class_2:.4f}")

# Get confusion matrix
cm = compute_confusion_matrix(val_labels, predicted_classes, 5)

# Analyze specific confusion
confusion_0_to_1 = cm[0, 1]  # How often class 0 is confused with class 1
print(f"Class 0 confused as class 1: {confusion_0_to_1} times")
```

### Batch Evaluation

```python
import os
from evaluate_gesture_model import main
import sys

# Evaluate multiple models
models = [
    ("models/gesture_classifier_lightweight.h5", "eval_lightweight.txt"),
    ("models/gesture_classifier_balanced.h5", "eval_balanced.txt"),
    ("models/gesture_classifier_powerful.h5", "eval_powerful.txt"),
]

for model_path, output_path in models:
    sys.argv = [
        "evaluate_gesture_model.py",
        "--model", model_path,
        "--data", "datasets/val_features.npy",
        "--labels", "datasets/val_labels.npy",
        "--output", output_path
    ]
    main()
```

---

## Output Example

**Console Output:**
```
================================================================================
EVALUATION METRICS
================================================================================

OVERALL ACCURACY: 0.8756 (87.56%)
Total Samples: 200

PER-CLASS METRICS:
Class 0: Accuracy 0.9000, Precision 0.9000, F1 0.9000 (40 samples)
Class 1: Accuracy 0.8500, Precision 0.8500, F1 0.8500 (40 samples)
...

================================================================================
CONFUSION MATRIX
================================================================================

Confusion Matrix:
[36  2  1  1  0]
[ 1 34  2  2  1]
...

================================================================================
SUMMARY STATISTICS
================================================================================

Correct Predictions: 175/200
Accuracy: 0.8750 (87.50%)

Per-Class Summary:
  Best Performance: Class 0 (0.9000 accuracy)
  Worst Performance: Class 2 (0.8000 accuracy)

Class Distribution:
  Class 0: 40 samples (20.0%)
  Class 1: 40 samples (20.0%)
...

================================================================================
EVALUATION COMPLETE
================================================================================

Results saved to: eval_results.txt
Confusion matrix plot saved to: confusion_matrix.png
Summary JSON saved to: eval_results_summary.json
```

---

## Best Practices

1. **Always Evaluate on Validation Data**
   - Never evaluate on training data (will overestimate accuracy)
   - Use separate validation set

2. **Monitor Per-Class Accuracy**
   - Some classes may need improvement
   - Collect more data for underperforming classes

3. **Save Results**
   - Always save text and JSON for analysis
   - Keep confusion matrix plots for documentation

4. **Compare Models**
   - Evaluate all architecture variants
   - Compare results side-by-side

5. **Iterate**
   - If accuracy is low, collect more data
   - Try different architectures
   - Tune hyperparameters

---

## File Reference

| Function | Purpose |
|----------|---------|
| `load_data()` | Load features and labels from .npy files |
| `compute_confusion_matrix()` | Calculate confusion matrix |
| `compute_metrics()` | Calculate accuracy, precision, recall, F1 |
| `format_confusion_matrix_str()` | Format matrix as string |
| `format_metrics_report()` | Format metrics as readable report |
| `visualize_confusion_matrix()` | Create PNG visualization |
| `FormattedLogger` | Custom logger class |

---

## See Also

- [Training Script](train_gesture_model.py)
- [Neural Network Guide](GESTURE_CLASSIFICATION_GUIDE.md)
- [Usage Examples](examples_gesture_classification.py)

---

**Status:** ✅ Production-Ready  
**Last Updated:** January 20, 2026
