"""
TensorFlow Lite Model Usage Examples

Demonstrates how to load and use converted TFLite models for real-time inference.
Shows both single predictions and batch processing on mobile/edge devices.
"""

import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import tensorflow as tf
except ImportError:
    tf = None


class TFLiteInference:
    """
    Wrapper for TensorFlow Lite model inference.
    Handles model loading, prediction, and performance monitoring.
    """

    def __init__(self, model_path: str, num_threads: int = 4):
        """
        Initialize TFLite interpreter.

        Args:
            model_path: Path to .tflite model file
            num_threads: Number of CPU threads (for mobile: 1-4, default: 4)
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Create interpreter with hardware acceleration options
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads,
        )

        # Allocate tensors
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Model info
        self.model_path = model_path
        self.num_threads = num_threads

        print(f"✓ Model loaded: {model_path}")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Output shape: {self.output_details[0]['shape']}")

    def predict_single(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Predict gesture for single feature vector.

        Args:
            features: Feature vector of shape (46,) for gesture model

        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Convert to model input type
        input_dtype = self.input_details[0]["dtype"]
        input_data = features.astype(input_dtype)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        predictions = output_data[0]

        # Get predicted class and confidence
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]

        return int(predicted_class), float(confidence)

    def predict_batch(
        self, features_batch: np.ndarray, batch_size: int = 32
    ) -> List[Tuple[int, float]]:
        """
        Predict gestures for batch of feature vectors.

        Args:
            features_batch: Batch of features, shape (num_samples, 46)
            batch_size: Process batch size for memory efficiency

        Returns:
            List of (predicted_class, confidence) tuples
        """
        results = []
        num_samples = len(features_batch)

        for i in range(0, num_samples, batch_size):
            batch_data = features_batch[i : i + batch_size]

            if batch_data.ndim == 1:
                batch_data = batch_data.reshape(1, -1)

            # Convert to model input type
            input_dtype = self.input_details[0]["dtype"]
            batch_data = batch_data.astype(input_dtype)

            # Ensure correct batch size (pad if needed)
            expected_shape = self.input_details[0]["shape"]
            if batch_data.shape[0] < expected_shape[0]:
                padding = np.zeros(
                    (expected_shape[0] - batch_data.shape[0], expected_shape[1])
                )
                batch_data = np.vstack([batch_data, padding])

            # Set input and run inference
            self.interpreter.set_tensor(self.input_details[0]["index"], batch_data)
            self.interpreter.invoke()

            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

            # Process results
            for j, predictions in enumerate(output_data[: len(features_batch[i : i + batch_size])]):
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class]
                results.append((int(predicted_class), float(confidence)))

        return results

    def predict_with_timing(
        self, features: np.ndarray, iterations: int = 100
    ) -> Tuple[Tuple[int, float], float]:
        """
        Predict and measure inference time.

        Args:
            features: Feature vector
            iterations: Number of iterations for timing

        Returns:
            Tuple of ((predicted_class, confidence), avg_inference_time_ms)
        """
        # Warmup
        _ = self.predict_single(features)

        # Measure inference time
        start_time = time.time()
        for _ in range(iterations):
            result = self.predict_single(features)
        end_time = time.time()

        total_time = (end_time - start_time) * 1000  # Convert to ms
        avg_time = total_time / iterations

        return result, avg_time

    def get_top_k_predictions(
        self, features: np.ndarray, k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Get top-k predictions with confidence scores.

        Args:
            features: Feature vector
            k: Number of top predictions

        Returns:
            List of (class, confidence) sorted by confidence
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        input_dtype = self.input_details[0]["dtype"]
        input_data = features.astype(input_dtype)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        predictions = output_data[0]

        # Get top-k
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        results = [(int(idx), float(predictions[idx])) for idx in top_k_indices]

        return results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


def example_basic_inference():
    """Example 1: Basic single prediction."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Single Prediction")
    print("=" * 80)

    # Initialize model
    model = TFLiteInference("models/gesture_classifier_dynamic_range.tflite")

    # Create sample features (46-dimensional vector)
    features = np.random.randn(46).astype(np.float32)
    features = features / (np.max(np.abs(features)) + 1e-8)

    # Predict
    gesture_class, confidence = model.predict_single(features)

    print(f"\nPredicted gesture class: {gesture_class}")
    print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")


def example_batch_prediction():
    """Example 2: Batch prediction."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch Prediction")
    print("=" * 80)

    model = TFLiteInference("models/gesture_classifier_dynamic_range.tflite")

    # Create batch of samples
    num_samples = 100
    features_batch = np.random.randn(num_samples, 46).astype(np.float32)
    features_batch = features_batch / (np.max(np.abs(features_batch)) + 1e-8)

    print(f"\nPredicting on {num_samples} samples...")

    start_time = time.time()
    results = model.predict_batch(features_batch, batch_size=32)
    inference_time = (time.time() - start_time) * 1000

    print(f"Inference time: {inference_time:.2f}ms")
    print(f"Average per sample: {inference_time / num_samples:.2f}ms")

    # Display first 10 results
    print("\nFirst 10 predictions:")
    for i, (gesture_class, confidence) in enumerate(results[:10]):
        print(f"  Sample {i}: Class {gesture_class}, Confidence {confidence:.4f}")


def example_timing_comparison():
    """Example 3: Compare inference time across different quantization methods."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Quantization Method Comparison")
    print("=" * 80)

    models_to_test = [
        ("models/gesture_classifier_float32.tflite", "Float32"),
        ("models/gesture_classifier_dynamic_range.tflite", "Dynamic Range"),
        ("models/gesture_classifier_float16.tflite", "Float16"),
        ("models/gesture_classifier_int8.tflite", "Full Integer (8-bit)"),
    ]

    # Create test features
    test_features = np.random.randn(46).astype(np.float32)
    test_features = test_features / (np.max(np.abs(test_features)) + 1e-8)

    results_summary = []

    for model_path, method_name in models_to_test:
        if not Path(model_path).exists():
            print(f"\n⚠ Model not found: {model_path}")
            continue

        try:
            model = TFLiteInference(model_path, num_threads=4)

            # Get file size
            model_size = Path(model_path).stat().st_size

            # Measure inference time
            _, avg_time = model.predict_with_timing(test_features, iterations=100)

            results_summary.append(
                (method_name, model_size, avg_time, model_path)
            )

            print(f"\n{method_name}:")
            print(f"  File size: {model_size / 1024:.1f} KB")
            print(f"  Avg inference time: {avg_time:.2f}ms")

        except Exception as e:
            print(f"\n✗ Error testing {method_name}: {e}")

    # Summary table
    if results_summary:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Method':<25} {'Size (KB)':<15} {'Inference (ms)':<15}")
        print("-" * 55)

        # Sort by inference time (fast is better)
        sorted_results = sorted(results_summary, key=lambda x: x[2])

        for method, size, time_ms, _ in sorted_results:
            print(f"{method:<25} {size / 1024:>13.1f} {time_ms:>13.2f}")


def example_top_k_predictions():
    """Example 4: Get top-3 gesture predictions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Top-K Predictions")
    print("=" * 80)

    model = TFLiteInference("models/gesture_classifier_dynamic_range.tflite")

    # Create test features
    features = np.random.randn(46).astype(np.float32)
    features = features / (np.max(np.abs(features)) + 1e-8)

    # Get top-3 predictions
    top_k = model.get_top_k_predictions(features, k=3)

    print(f"\nTop 3 gesture predictions:")
    for rank, (gesture_class, confidence) in enumerate(top_k, 1):
        print(f"  {rank}. Class {gesture_class}: {confidence:.4f} ({confidence * 100:.2f}%)")


def example_real_time_simulation():
    """Example 5: Simulate real-time gesture recognition."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Real-Time Simulation")
    print("=" * 80)

    model = TFLiteInference("models/gesture_classifier_dynamic_range.tflite")

    print("\nSimulating real-time gesture recognition (10 frames)...")
    print("-" * 80)

    gesture_names = ["Palm", "Fist", "Peace", "OK", "Thumbs Up"]  # Example names

    frame_times = []

    for frame_id in range(10):
        # Generate features for this frame
        features = np.random.randn(46).astype(np.float32)
        features = features / (np.max(np.abs(features)) + 1e-8)

        # Predict
        start = time.time()
        gesture_class, confidence = model.predict_single(features)
        frame_time = (time.time() - start) * 1000

        frame_times.append(frame_time)

        gesture_name = gesture_names[gesture_class % len(gesture_names)]
        print(f"Frame {frame_id:2d}: {gesture_name:<12} ({confidence:.2%}) {frame_time:.2f}ms")

    print("-" * 80)
    print(f"Average frame time: {np.mean(frame_times):.2f}ms")
    print(f"FPS: {1000 / np.mean(frame_times):.1f}")
    print(f"Min/Max time: {np.min(frame_times):.2f}ms / {np.max(frame_times):.2f}ms")


def example_model_size_comparison():
    """Example 6: Compare model sizes on disk."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Model Size Comparison")
    print("=" * 80)

    models = [
        ("models/gesture_classifier.h5", "TensorFlow (.h5)"),
        ("models/gesture_classifier_float32.tflite", "TFLite Float32"),
        ("models/gesture_classifier_dynamic_range.tflite", "TFLite Dynamic Range"),
        ("models/gesture_classifier_float16.tflite", "TFLite Float16"),
        ("models/gesture_classifier_int8.tflite", "TFLite Int8"),
    ]

    print("\nModel file sizes:")
    print("-" * 60)
    print(f"{'Model':<30} {'Size (KB)':<15} {'Size (MB)':<15}")
    print("-" * 60)

    sizes = []

    for model_path, model_name in models:
        if Path(model_path).exists():
            size_kb = Path(model_path).stat().st_size / 1024
            size_mb = size_kb / 1024
            sizes.append((model_name, size_kb, size_mb))
            print(f"{model_name:<30} {size_kb:>13.1f} {size_mb:>13.3f}")
        else:
            print(f"{model_name:<30} {'N/A':<15} {'N/A':<15}")

    if sizes:
        print("-" * 60)
        original_size = sizes[0][2]  # Original .h5 size
        print(f"\nSize Reductions (vs original {original_size:.2f}MB):")
        for model_name, size_kb, size_mb in sizes[1:]:
            reduction_pct = (1 - size_mb / original_size) * 100
            print(f"  {model_name:<28} {reduction_pct:>5.1f}%")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("TensorFlow Lite Inference Examples")
    print("=" * 80)

    examples = [
        (example_basic_inference, "Basic single prediction"),
        (example_batch_prediction, "Batch prediction"),
        (example_timing_comparison, "Quantization method timing comparison"),
        (example_top_k_predictions, "Top-K predictions"),
        (example_real_time_simulation, "Real-time simulation"),
        (example_model_size_comparison, "Model size comparison"),
    ]

    print("\nAvailable examples:")
    for i, (func, description) in enumerate(examples, 1):
        print(f"  {i}. {description}")

    print("\nRunning all examples...\n")

    for func, description in examples:
        try:
            func()
        except FileNotFoundError as e:
            print(f"\n⚠ Skipping example: {e}")
        except Exception as e:
            print(f"\n✗ Error in example: {e}")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
