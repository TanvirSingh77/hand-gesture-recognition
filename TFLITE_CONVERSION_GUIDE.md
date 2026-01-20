# TensorFlow Lite Conversion Guide

## Overview

This guide explains how to convert trained TensorFlow gesture classification models to TensorFlow Lite format with multiple quantization strategies for optimal deployment on mobile and edge devices.

---

## Why TensorFlow Lite?

### Size Reduction
- **Original (.h5)**: 2-4 MB
- **TFLite with quantization**: 200-800 KB (60-95% smaller)

### Speed Improvement
- **CPU**: 30-50% faster (dynamic range), 2-3x faster (full integer)
- **Accelerators**: 10-100x faster on Edge TPU, specialized hardware

### Device Compatibility
- Mobile: iOS, Android
- Edge: Raspberry Pi, ARM devices
- Microcontrollers: Arduino, embedded systems
- Accelerators: Google Edge TPU, GPU, TPU

### Reduced Latency
- Typical inference: 5-20ms on mobile (vs 50-100ms for full model)
- Real-time gesture recognition feasible

---

## Quantization Methods

### 1. Float32 (No Quantization)

**What it does:**
- Keeps all weights and activations as float32 (32-bit floating point)
- Direct conversion without compression
- Baseline for comparison

**Why use it:**
- ✓ Highest accuracy (100% preserved)
- ✓ Simplest conversion
- ✓ Good for cloud inference

**Why NOT:**
- ✗ Largest file size (~2-4 MB)
- ✗ Slowest inference
- ✗ Not suitable for mobile/edge

**Use cases:**
- Cloud-based inference (AWS Lambda, Google Cloud)
- When storage/latency not critical
- Baseline accuracy measurement

**Expected results:**
- Size reduction: 0% (reference point)
- Accuracy preservation: 100%
- Inference speed: Normal (baseline)

---

### 2. Dynamic Range Quantization

**What it does:**
- Quantizes **weights** to 8-bit integers
- Keeps **activations** as float32
- Automatic range calibration

**Why use it:**
- ✓ ~75% size reduction (best general choice)
- ✓ Minimal accuracy loss (1-3%)
- ✓ 30-50% faster on CPU
- ✓ Works on all devices
- ✓ No representative dataset needed

**Why NOT:**
- ✗ Not optimized for specialized hardware (TPU, etc.)
- ✗ Activations still float32 (memory overhead)

**Use cases:**
- **Mobile phones** (iOS, Android)
- **Most common use case**
- When you need good balance of size/speed/accuracy
- Deployment on general-purpose devices

**How it works:**
```
Original weights: [1.23, -0.45, 2.67, ...]  (float32)
↓ Quantization
Quantized weights: [125, 46, 271, ...]  (int8)
↓ During inference
Activations computed in float32, weights dequantized from int8
```

**Expected results:**
- Size reduction: 70-75%
- Accuracy preservation: 98-99%
- Inference speed: 30-50% faster
- Memory usage: Moderate

---

### 3. Float16 Quantization

**What it does:**
- Converts all operations to float16 (16-bit floating point)
- Intermediate precision between float32 and int8
- Good accuracy preservation

**Why use it:**
- ✓ ~50% size reduction
- ✓ Very high accuracy (99.5%+)
- ✓ 40-60% faster
- ✓ Works on modern GPUs/mobile chips

**Why NOT:**
- ✗ Requires float16 hardware support
- ✗ Not optimal for edge devices without float16
- ✗ Larger than int8 quantization

**Use cases:**
- iOS devices (Apple Neural Engine)
- Modern Android phones (Snapdragon 8xx)
- GPUs with float16 support
- When accuracy is critical (>99%)

**How it works:**
```
Original (float32): [1.23456789, 2.34567890, ...]
↓ Convert to float16
Quantized (float16): [1.2346, 2.3457, ...]
↓ Inference in float16 throughout
Result: 50% size reduction with <1% accuracy loss
```

**Expected results:**
- Size reduction: 50%
- Accuracy preservation: 99.5-99.9%
- Inference speed: 40-60% faster
- Memory usage: Low-moderate

---

### 4. Full Integer Quantization (8-bit)

**What it does:**
- Quantizes **both weights AND activations** to 8-bit integers
- Requires representative dataset for calibration
- Input/output can be float32 (automatically quantized/dequantized)

**Why use it:**
- ✓ Maximum size reduction (75-80%)
- ✓ Fastest inference (2-3x on quantized hardware)
- ✓ Best for Edge TPU, ARM devices
- ✓ Lowest memory footprint

**Why NOT:**
- ✗ Requires representative dataset for calibration
- ✗ Largest accuracy loss (95-98% preserved)
- ✗ Requires quantization-aware hardware
- ✗ More complex setup

**Use cases:**
- **Edge TPU** (Google Coral)
- **Raspberry Pi**, ARM boards
- **Microcontrollers** with limited memory
- **Embedded systems**
- When storage is critical constraint

**Calibration process:**
```
1. Load representative dataset (validation features)
2. Run forward pass to collect activation ranges
3. Calibrate quantization parameters for each layer
4. Quantize weights and activations to int8
5. Update quantization parameters in model
```

**Expected results:**
- Size reduction: 75-80% (best)
- Accuracy preservation: 95-98%
- Inference speed: 2-3x faster (quantized hardware)
- Memory usage: Minimal

---

## Quick Decision Tree

```
┌─ What's your deployment target?
│
├─ Mobile Phone (iPhone/Android)
│  └─ Use: Dynamic Range Quantization
│     (75% smaller, 30-50% faster, 99% accurate)
│
├─ Edge TPU / Google Coral
│  └─ Use: Full Integer Quantization
│     (80% smaller, 2-3x faster, 97% accurate)
│
├─ Raspberry Pi / ARM Board
│  └─ Use: Full Integer Quantization
│     (80% smaller, much faster, 97% accurate)
│
├─ Cloud Server (AWS, Google Cloud)
│  └─ Use: Float32 (No quantization)
│     (100% accurate, size not critical)
│
├─ Modern GPU / Apple Neural Engine
│  └─ Use: Float16 Quantization
│     (50% smaller, 40-60% faster, 99.5% accurate)
│
└─ Microcontroller (Arduino, etc)
   └─ Use: Full Integer Quantization (with careful memory budgeting)
```

---

## Usage Examples

### Basic Conversion (All Methods)

```bash
python convert_to_tflite.py --model models/gesture_classifier.h5
```

**Output:**
- `models/gesture_classifier_float32.tflite`
- `models/gesture_classifier_dynamic_range.tflite`
- `models/gesture_classifier_float16.tflite`
- `models/gesture_classifier_int8.tflite`
- `models/conversion_results.json`

### Specific Method

```bash
# Dynamic range only (most common)
python convert_to_tflite.py --model models/gesture_classifier.h5 \
    --method dynamic_range

# Full integer with representative data
python convert_to_tflite.py --model models/gesture_classifier.h5 \
    --method full_integer \
    --data datasets/val_features.npy
```

### Custom Output Directory

```bash
python convert_to_tflite.py --model models/gesture_classifier.h5 \
    --output converted_models
```

---

## Understanding the Results

### Console Output Example

```
================================================================================
CONVERSION METHOD 2: Dynamic Range Quantization
================================================================================
Converting model (Dynamic Range Quantization)...
  - Weights quantized to 8-bit integers
  - Activations remain float32
  - Expected size reduction: ~75%
✓ Conversion successful
  Saved to: models/gesture_classifier_dynamic_range.tflite
  File size: 512,345 bytes (0.49 MB)
  Size reduction: 75.2%

================================================================================
MODEL SIZE COMPARISON
================================================================================

Original Model (.h5):
  File size: 2,048,576 bytes (1.95 MB)

TensorFlow Lite Conversions:

  Float32:
    File size: 2,056,789 bytes (1.96 MB)
    [Almost no reduction - baseline]

  Dynamic Range Quantization:
    File size: 512,345 bytes (0.49 MB)
    Size reduction: 75.2%
    Speed vs original: 30-50% faster
    Accuracy preservation: 98-99%

  Float16 Quantization:
    File size: 1,024,567 bytes (0.98 MB)
    Size reduction: 50.1%
    Speed vs original: 40-60% faster
    Accuracy preservation: 99.5-99.9%

  Full Integer Quantization (8-bit):
    File size: 409,872 bytes (0.39 MB)
    Size reduction: 80.0%
    Speed vs original: 2-3x faster (quantized hardware)
    Accuracy preservation: 95-98%
```

### JSON Results File

```json
{
  "timestamp": "2024-01-20T15:45:30.123456",
  "original_model": {
    "path": "models/gesture_classifier.h5",
    "size_bytes": 2048576,
    "size_mb": 1.95,
    "input_shape": [null, 46],
    "output_shape": [null, 5],
    "parameters": 15234
  },
  "conversions": {
    "dynamic_range": {
      "method": "Dynamic Range Quantization",
      "size_bytes": 512345,
      "size_reduction_percent": 75.2,
      "accuracy_preservation": "98-99%"
    },
    "full_integer": {
      "method": "Full Integer Quantization (8-bit)",
      "size_bytes": 409872,
      "size_reduction_percent": 80.0,
      "accuracy_preservation": "95-98%"
    }
  }
}
```

---

## Why Each Optimization is Applied

### Weight Quantization (8-bit)
**Why:** 
- Weights are typically 30-40% of model size
- Same weights used for all samples
- Can be calibrated once during conversion
- Minimal accuracy impact

**How it works:**
```
Range: weights from -2.5 to 2.5
Map to: -128 to 127 (int8)
Original: 1.23 (float32) → Quantized: 63 (int8)
Savings: 4 bytes → 1 byte per weight (75% reduction)
```

### Activation Quantization (8-bit) - Full Integer Only
**Why:**
- Reduces memory bandwidth during inference
- Enables faster integer operations on hardware
- Critical for edge TPU and ARM NEON

**Trade-off:**
- Must calibrate per layer (needs representative data)
- Can lose accuracy if calibration data not representative
- Requires quantization-aware hardware

### Float16 Quantization
**Why:**
- Modern GPUs have fast float16 operations
- Still maintains good accuracy (vs int8)
- Supported on modern mobile chips
- Natural middle ground between float32 and int8

**Advantages:**
- No calibration needed (deterministic conversion)
- Simpler than int8 quantization
- Better accuracy with minimal size penalty

### Why Not Quantize Further (Lower Bit-Width)?

**4-bit, 2-bit, 1-bit quantization:**
- Theoretical size: Even smaller
- **But:** Accuracy loss is too severe (>10-20% accuracy drop)
- Mobile apps need >95% accuracy minimum
- Not supported by standard TFLite

**When you might need it:**
- Extreme edge cases (IoT with <1MB storage)
- Non-critical applications (where 80% accuracy acceptable)
- Research/experimental purposes

---

## Performance Metrics Explained

### Size Reduction %
**Formula:** `(1 - quantized_size / original_size) * 100`

**Example:**
- Original: 2.0 MB
- Quantized: 0.5 MB
- Reduction: `(1 - 0.5/2.0) * 100 = 75%`

### Accuracy Preservation
**Definition:** Percentage of validation accuracy maintained after quantization

**Typical ranges:**
- Dynamic Range: 98-99% (1-2% loss)
- Float16: 99.5-99.9% (<1% loss)
- Full Integer: 95-98% (2-5% loss)

**Loss causes:**
- Rounding errors in quantization
- Activation distribution mismatch during calibration
- Cumulative effects through multiple layers

### Inference Speed Improvement

**Dynamic Range (CPU):**
- Smaller model fits better in CPU cache
- Fewer memory transfers needed
- Typical: 30-50% faster on modern CPUs

**Float16 (GPU):**
- Native GPU support for float16 ops
- Less bandwidth required
- Typical: 40-60% faster on GPUs

**Full Integer (Quantized Hardware):**
- Integer operations are fastest on all hardware
- Edge TPU: 10-100x faster
- ARM NEON: 2-3x faster
- Typical: 2-3x faster on quantized accelerators

---

## Conversion Workflow for Different Devices

### For iOS App

```bash
# 1. Convert with dynamic range (good balance)
python convert_to_tflite.py --model models/gesture_classifier.h5 \
    --method dynamic_range

# 2. Output: gesture_classifier_dynamic_range.tflite (~0.5 MB)

# 3. Add to Xcode project
# 4. Load using TensorFlow Lite iOS SDK
```

### For Android App

```bash
# 1. Convert all methods and benchmark
python convert_to_tflite.py --model models/gesture_classifier.h5

# 2. Choose based on target device:
#    - Modern phones (2020+): dynamic_range.tflite
#    - Older phones: float32.tflite

# 3. Add to Android project
# 4. Load using TensorFlow Lite Android SDK
```

### For Raspberry Pi

```bash
# 1. Convert with full integer quantization
python convert_to_tflite.py --model models/gesture_classifier.h5 \
    --method full_integer \
    --data datasets/val_features.npy

# 2. Output: gesture_classifier_int8.tflite (~0.39 MB)

# 3. Copy to Raspberry Pi
# 4. Load using TensorFlow Lite Python interpreter
```

### For Google Edge TPU (Coral)

```bash
# 1. Convert with full integer quantization
python convert_to_tflite.py --model models/gesture_classifier.h5 \
    --method full_integer \
    --data datasets/val_features.npy

# 2. Compile for Edge TPU using:
#    edgetpu_compiler gesture_classifier_int8.tflite

# 3. Output: gesture_classifier_int8_edgetpu.tflite
```

---

## Troubleshooting

### Model Conversion Fails

**Issue:** "ValueError: Unsupported operation"

**Solution:**
- Not all TensorFlow operations supported in TFLite
- Convert problematic layers to supported ops
- See TensorFlow Lite ops list

### Accuracy Drops Significantly After Quantization

**Issue:** Quantized model has <90% accuracy on validation data

**Solutions:**
1. Use representative dataset for calibration
2. Verify representative data is from same distribution
3. Try less aggressive quantization (float16 instead of int8)
4. Retrain model with quantization-aware training

### Model File Not Found

**Issue:** "FileNotFoundError: Model file not found"

**Solution:**
```bash
# Verify model exists
ls -la models/gesture_classifier.h5

# Train model first if missing
python train_gesture_model.py --architecture balanced
```

### Representative Data Issues

**Issue:** "ValueError: Representative dataset has wrong shape"

**Solution:**
```python
# Representative data must match model input shape
# For gesture model: (batch_size, 46) where 46 is feature count

data = np.load("datasets/val_features.npy")
print(data.shape)  # Should be (num_samples, 46)
```

---

## Best Practices

1. **Always test quantized models**
   - Measure accuracy on validation data
   - Verify on target device if possible
   - Accept <2% accuracy loss for mobile

2. **Use representative data for int8**
   - Real validation data best
   - Generate synthetic if unavailable
   - Must match training data distribution

3. **Choose quantization per use case**
   - Mobile → Dynamic Range
   - Edge TPU → Full Integer
   - Cloud → Float32
   - Premium devices → Float16

4. **Compare all methods**
   - Don't assume one is best
   - Benchmark on target device
   - Trade off size vs accuracy vs speed

5. **Monitor model size**
   - Original models can grow large
   - TFLite keeps them manageable
   - Track model size in version control

6. **Document quantization choice**
   - Note which method chosen
   - Record expected accuracy loss
   - Document target device requirements

---

## Resources

- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [Model Optimization Guide](https://www.tensorflow.org/lite/guide/model_optimization)
- [Quantization Documentation](https://www.tensorflow.org/lite/guide/quantization)
- [TensorFlow Lite Interpreter API](https://www.tensorflow.org/lite/guide/inference)

---

## File Reference

**Main Conversion Script:** [convert_to_tflite.py](convert_to_tflite.py)

**Key Methods:**
- `convert_float32()` - Baseline conversion
- `convert_dynamic_range()` - Weight quantization
- `convert_float16()` - Half precision floating point
- `convert_full_integer()` - Full 8-bit quantization
- `convert_all()` - All methods

**Usage:**
```bash
python convert_to_tflite.py --model <path> [--method] [--data] [--output]
```

---

**Status:** ✅ Complete & Production-Ready  
**Last Updated:** January 20, 2026
