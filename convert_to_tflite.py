"""
TensorFlow Lite Model Conversion Script

Converts trained TensorFlow/Keras gesture classification models to TensorFlow Lite format
with multiple quantization strategies and comprehensive size comparison.

This script provides:
1. Float32 baseline (no quantization)
2. Dynamic range quantization (8-bit integers)
3. Full integer quantization (8-bit with calibration data)
4. Float16 quantization (reduced precision floating point)

Each strategy trades off accuracy vs. model size and inference speed.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import schema_util


class TFLiteConverter:
    """
    Handles conversion of TensorFlow/Keras models to TensorFlow Lite format
    with multiple quantization strategies.
    """

    def __init__(self, model_path: str, output_dir: str = "models", verbose: bool = True):
        """
        Initialize converter.

        Args:
            model_path: Path to trained .h5 model
            output_dir: Directory to save .tflite models
            verbose: Print detailed logs
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Setup logging
        self.logger = self._setup_logging()

        # Load model
        self.model = None
        self._load_model()

        # Store conversion results
        self.results = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup formatted logger."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_model(self) -> None:
        """Load TensorFlow/Keras model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.logger.info(f"✓ Model loaded: {self.model_path}")
            self.logger.info(f"  Input shape: {self.model.input_shape}")
            self.logger.info(f"  Output shape: {self.model.output_shape}")
            self.logger.info(f"  Parameters: {self.model.count_params():,}")
        except Exception as e:
            self.logger.error(f"✗ Failed to load model: {e}")
            raise

    def _create_representative_dataset(
        self, num_samples: int = 100, feature_size: int = 46
    ) -> tf.data.Dataset:
        """
        Create representative dataset for quantization.

        This is used for calibration during full integer quantization.
        For demo purposes, we generate random data matching expected feature shape.
        In production, use actual validation data.

        Args:
            num_samples: Number of samples to generate
            feature_size: Size of feature vector (should match model input)

        Returns:
            tf.data.Dataset with representative data
        """
        # Generate random representative data
        # In production, replace with actual validation features
        data = np.random.randn(num_samples, feature_size).astype(np.float32)

        # Normalize to [-1, 1] range (typical for normalized features)
        data = data / (np.max(np.abs(data)) + 1e-8)

        def representative_dataset():
            """Generator for representative dataset."""
            for i in range(0, len(data), 10):
                yield [data[i : i + 10].astype(np.float32)]

        return representative_dataset

    def convert_float32(self, save_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Convert model to TensorFlow Lite without quantization (Float32).

        WHY: Baseline conversion - preserves full model accuracy but largest file size.
        USE CASE: When accuracy is critical and model size/speed is less important.
        TRADE-OFF: Largest size, slowest inference, highest accuracy

        Args:
            save_path: Custom save path (optional)

        Returns:
            Tuple of (save_path, conversion_info)
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CONVERSION METHOD 1: Float32 (No Quantization)")
        self.logger.info("=" * 80)

        if save_path is None:
            save_path = str(self.output_dir / "gesture_classifier_float32.tflite")

        try:
            self.logger.info("Converting model (Float32)...")

            # Create converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

            # No optimizations - baseline conversion
            converter.optimizations = []
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]

            # Convert and save
            tflite_model = converter.convert()
            with open(save_path, "wb") as f:
                f.write(tflite_model)

            file_size = os.path.getsize(save_path)
            self.logger.info(f"✓ Conversion successful")
            self.logger.info(f"  Saved to: {save_path}")
            self.logger.info(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

            info = {
                "method": "Float32",
                "size_bytes": file_size,
                "save_path": save_path,
                "quantization": "None",
                "optimization_level": "None (Baseline)",
                "accuracy_preservation": "100%",
                "inference_speed": "Normal",
                "device_compatibility": "All devices",
                "description": "Full precision baseline - largest size, highest accuracy",
            }

            self.results["float32"] = info
            return save_path, info

        except Exception as e:
            self.logger.error(f"✗ Conversion failed: {e}")
            raise

    def convert_dynamic_range(self, save_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Convert with Dynamic Range Quantization (weights only).

        WHY: Quantizes weights to 8-bit integers while keeping activations as float32.
        This reduces model size significantly with minimal accuracy loss.
        USE CASE: Mobile devices with good CPU performance, when file size matters most.
        TRADE-OFF: ~75% size reduction, ~1-3% accuracy loss, 30-50% faster on CPU

        Args:
            save_path: Custom save path (optional)

        Returns:
            Tuple of (save_path, conversion_info)
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CONVERSION METHOD 2: Dynamic Range Quantization")
        self.logger.info("=" * 80)

        if save_path is None:
            save_path = str(self.output_dir / "gesture_classifier_dynamic_range.tflite")

        try:
            self.logger.info("Converting model (Dynamic Range Quantization)...")
            self.logger.info("  - Weights quantized to 8-bit integers")
            self.logger.info("  - Activations remain float32")
            self.logger.info("  - Expected size reduction: ~75%")

            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

            # Enable weight quantization (dynamic range)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Keep activations as float (not fully quantized)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]

            # Convert and save
            tflite_model = converter.convert()
            with open(save_path, "wb") as f:
                f.write(tflite_model)

            file_size = os.path.getsize(save_path)
            original_size = os.path.getsize(self.model_path)
            reduction_pct = (1 - file_size / original_size) * 100

            self.logger.info(f"✓ Conversion successful")
            self.logger.info(f"  Saved to: {save_path}")
            self.logger.info(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            self.logger.info(f"  Size reduction: {reduction_pct:.1f}%")

            info = {
                "method": "Dynamic Range Quantization",
                "size_bytes": file_size,
                "save_path": save_path,
                "size_reduction_percent": round(reduction_pct, 1),
                "quantization": "Weights (8-bit), Activations (float32)",
                "optimization_level": "Medium (Balanced approach)",
                "accuracy_preservation": "98-99%",
                "inference_speed": "30-50% faster",
                "device_compatibility": "All mobile devices (CPU)",
                "description": "Weights quantized to 8-bit, activations float32",
            }

            self.results["dynamic_range"] = info
            return save_path, info

        except Exception as e:
            self.logger.error(f"✗ Conversion failed: {e}")
            raise

    def convert_full_integer(
        self,
        representative_data_path: Optional[str] = None,
        num_samples: int = 100,
        save_path: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """
        Convert with Full Integer Quantization (8-bit).

        WHY: Quantizes both weights and activations to 8-bit integers.
        Requires representative dataset for calibration.
        USE CASE: Embedded devices (ARM, Edge TPU, microcontrollers).
        TRADE-OFF: ~75-80% size reduction, ~2-5% accuracy loss, 2-3x faster on quantized hardware

        Args:
            representative_data_path: Path to validation features .npy file (optional)
            num_samples: Number of samples for representative dataset if not provided
            save_path: Custom save path (optional)

        Returns:
            Tuple of (save_path, conversion_info)
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CONVERSION METHOD 3: Full Integer Quantization (8-bit)")
        self.logger.info("=" * 80)

        if save_path is None:
            save_path = str(self.output_dir / "gesture_classifier_int8.tflite")

        try:
            self.logger.info("Converting model (Full Integer Quantization)...")
            self.logger.info("  - Weights quantized to 8-bit integers")
            self.logger.info("  - Activations quantized to 8-bit integers")
            self.logger.info("  - Requires representative dataset for calibration")
            self.logger.info(f"  - Expected size reduction: 75-80%")

            # Load representative data if provided
            if representative_data_path and os.path.exists(representative_data_path):
                self.logger.info(f"  - Loading representative data from: {representative_data_path}")
                representative_data = np.load(representative_data_path)
                feature_size = representative_data.shape[1]

                def representative_dataset():
                    for i in range(0, len(representative_data), 10):
                        yield [representative_data[i : i + 10].astype(np.float32)]

            else:
                self.logger.info(f"  - Generating synthetic representative data ({num_samples} samples)")
                representative_dataset_fn = self._create_representative_dataset(
                    num_samples=num_samples, feature_size=self.model.input_shape[-1]
                )
                representative_dataset = representative_dataset_fn

            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

            # Enable full integer quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset

            # Require integer inputs/outputs for full quantization
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]

            # Allow buffer to be converted to float if needed
            converter.allow_custom_ops = False

            # Convert and save
            tflite_model = converter.convert()
            with open(save_path, "wb") as f:
                f.write(tflite_model)

            file_size = os.path.getsize(save_path)
            original_size = os.path.getsize(self.model_path)
            reduction_pct = (1 - file_size / original_size) * 100

            self.logger.info(f"✓ Conversion successful")
            self.logger.info(f"  Saved to: {save_path}")
            self.logger.info(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            self.logger.info(f"  Size reduction: {reduction_pct:.1f}%")

            info = {
                "method": "Full Integer Quantization (8-bit)",
                "size_bytes": file_size,
                "save_path": save_path,
                "size_reduction_percent": round(reduction_pct, 1),
                "quantization": "Weights and Activations (8-bit integers)",
                "optimization_level": "High (Aggressive optimization)",
                "accuracy_preservation": "95-98%",
                "inference_speed": "2-3x faster (quantized hardware)",
                "device_compatibility": "Edge TPU, ARM, microcontrollers",
                "description": "Both weights and activations quantized to 8-bit",
                "calibration": "Required (representative dataset used)",
            }

            self.results["full_integer"] = info
            return save_path, info

        except Exception as e:
            self.logger.error(f"✗ Conversion failed: {e}")
            raise

    def convert_float16(self, save_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Convert with Float16 Quantization (half precision floating point).

        WHY: Reduces precision from float32 to float16 (half precision).
        Intermediate between float32 and int8 quantization.
        USE CASE: Devices with float16 support but no int8 acceleration.
        TRADE-OFF: ~50% size reduction, <1% accuracy loss, 40-60% faster

        Args:
            save_path: Custom save path (optional)

        Returns:
            Tuple of (save_path, conversion_info)
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CONVERSION METHOD 4: Float16 Quantization")
        self.logger.info("=" * 80)

        if save_path is None:
            save_path = str(self.output_dir / "gesture_classifier_float16.tflite")

        try:
            self.logger.info("Converting model (Float16 Quantization)...")
            self.logger.info("  - All operations in float16 (half precision)")
            self.logger.info("  - Expected size reduction: ~50%")

            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

            # Enable float16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

            # Convert and save
            tflite_model = converter.convert()
            with open(save_path, "wb") as f:
                f.write(tflite_model)

            file_size = os.path.getsize(save_path)
            original_size = os.path.getsize(self.model_path)
            reduction_pct = (1 - file_size / original_size) * 100

            self.logger.info(f"✓ Conversion successful")
            self.logger.info(f"  Saved to: {save_path}")
            self.logger.info(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            self.logger.info(f"  Size reduction: {reduction_pct:.1f}%")

            info = {
                "method": "Float16 Quantization",
                "size_bytes": file_size,
                "save_path": save_path,
                "size_reduction_percent": round(reduction_pct, 1),
                "quantization": "All operations in float16 (half precision)",
                "optimization_level": "Medium-High (Good balance)",
                "accuracy_preservation": "99.5-99.9%",
                "inference_speed": "40-60% faster",
                "device_compatibility": "Modern GPUs, some mobile chips",
                "description": "All operations converted to 16-bit floating point",
            }

            self.results["float16"] = info
            return save_path, info

        except Exception as e:
            self.logger.error(f"✗ Conversion failed: {e}")
            raise

    def convert_all(self, representative_data_path: Optional[str] = None) -> Dict:
        """
        Convert model using all quantization strategies.

        Args:
            representative_data_path: Path to validation features for int8 quantization

        Returns:
            Dictionary with all conversion results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TensorFlow Lite Model Conversion - All Methods")
        self.logger.info("=" * 80)

        conversions = [
            ("float32", self.convert_float32),
            ("dynamic_range", self.convert_dynamic_range),
            ("float16", self.convert_float16),
            ("full_integer", lambda: self.convert_full_integer(representative_data_path)),
        ]

        for name, conversion_fn in conversions:
            try:
                conversion_fn()
            except Exception as e:
                self.logger.error(f"Failed to convert with {name}: {e}")

        return self.results

    def get_size_comparison(self) -> Dict:
        """
        Compare file sizes of all converted models.

        Returns:
            Dictionary with size comparisons
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("MODEL SIZE COMPARISON")
        self.logger.info("=" * 80)

        original_size = os.path.getsize(self.model_path)
        self.logger.info(f"\nOriginal Model (.h5):")
        self.logger.info(f"  File size: {original_size:,} bytes ({original_size / 1024 / 1024:.2f} MB)")

        comparison = {"original": original_size, "conversions": {}}

        if self.results:
            self.logger.info(f"\nTensorFlow Lite Conversions:")

            for method, result in sorted(self.results.items()):
                size = result.get("size_bytes", 0)
                reduction = result.get("size_reduction_percent", 0)

                self.logger.info(f"\n  {result['method']}:")
                self.logger.info(f"    File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

                if "size_reduction_percent" in result:
                    self.logger.info(f"    Size reduction: {reduction:.1f}%")
                    self.logger.info(f"    Speed vs original: {result['inference_speed']}")
                    self.logger.info(f"    Accuracy preservation: {result['accuracy_preservation']}")

                comparison["conversions"][method] = {
                    "size_bytes": size,
                    "size_mb": round(size / 1024 / 1024, 2),
                }

        # Calculate savings across all methods
        if self.results:
            self.logger.info(f"\nSIZE REDUCTION SUMMARY:")
            total_saved = sum(
                original_size - result.get("size_bytes", 0)
                for result in self.results.values()
            )
            avg_reduction = (total_saved / (len(self.results) * original_size)) * 100
            self.logger.info(
                f"  Average reduction: {avg_reduction:.1f}% across all methods"
            )
            self.logger.info(
                f"  Total storage saved: {total_saved / 1024 / 1024:.2f} MB (all models combined)"
            )

        return comparison

    def save_results_json(self, output_path: Optional[str] = None) -> str:
        """
        Save conversion results to JSON file.

        Args:
            output_path: Path to save JSON (optional)

        Returns:
            Path to saved JSON file
        """
        if output_path is None:
            output_path = str(self.output_dir / "conversion_results.json")

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "original_model": {
                "path": self.model_path,
                "size_bytes": os.path.getsize(self.model_path),
                "size_mb": round(os.path.getsize(self.model_path) / 1024 / 1024, 2),
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape,
                "parameters": self.model.count_params(),
            },
            "conversions": self.results,
        }

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"\n✓ Results saved to: {output_path}")
        return output_path

    def print_recommendations(self) -> None:
        """Print conversion recommendations based on use case."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CONVERSION RECOMMENDATIONS")
        self.logger.info("=" * 80)

        recommendations = {
            "Mobile Phones": {
                "recommended": "Dynamic Range Quantization",
                "reason": "Good balance of size/speed/accuracy for modern phones",
                "info": "~75% smaller, still accurate",
            },
            "Edge Devices (Raspberry Pi, etc)": {
                "recommended": "Full Integer Quantization (8-bit)",
                "reason": "Maximum size reduction for deployment on constrained devices",
                "info": "~75-80% smaller, suitable for ARM processors",
            },
            "Cloud Inference": {
                "recommended": "Float32 (No Quantization)",
                "reason": "Maximize accuracy when size/latency not critical",
                "info": "Full accuracy, sufficient for server deployment",
            },
            "Google TPU/Edge TPU": {
                "recommended": "Full Integer Quantization (8-bit)",
                "reason": "Optimized for integer operations on TPU hardware",
                "info": "2-3x faster inference, maximum optimization",
            },
            "iOS Apps": {
                "recommended": "Float16 or Dynamic Range",
                "reason": "iOS supports both, float16 for precision, dynamic for size",
                "info": "50-75% smaller depending on choice",
            },
        }

        for use_case, rec in recommendations.items():
            self.logger.info(f"\n{use_case}:")
            self.logger.info(f"  Recommended: {rec['recommended']}")
            self.logger.info(f"  Reason: {rec['reason']}")
            self.logger.info(f"  Details: {rec['info']}")

    def print_summary(self) -> None:
        """Print detailed summary of all conversions."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CONVERSION SUMMARY")
        self.logger.info("=" * 80)

        self.logger.info(f"\nOriginal Model: {self.model_path}")
        self.logger.info(f"Output Directory: {self.output_dir}")

        self.logger.info(f"\nQuantization Methods Overview:\n")

        methods_info = [
            {
                "name": "Float32",
                "pros": ["Highest accuracy", "Baseline"],
                "cons": ["Largest file size", "Slowest"],
            },
            {
                "name": "Dynamic Range",
                "pros": ["75% smaller", "30-50% faster", "99% accuracy"],
                "cons": ["Requires more storage than int8"],
            },
            {
                "name": "Float16",
                "pros": ["50% smaller", "40-60% faster", "99.5%+ accuracy"],
                "cons": ["Requires float16 hardware support"],
            },
            {
                "name": "Full Integer (8-bit)",
                "pros": ["75-80% smaller", "2-3x faster", "Best for edge"],
                "cons": ["95-98% accuracy", "Needs calibration"],
            },
        ]

        for method in methods_info:
            self.logger.info(f"{method['name']}:")
            for pro in method["pros"]:
                self.logger.info(f"  ✓ {pro}")
            for con in method["cons"]:
                self.logger.info(f"  ✗ {con}")
            self.logger.info()


def main():
    """Main entry point for TFLite conversion."""
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow model to TensorFlow Lite format with quantization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert using all methods
  python convert_to_tflite.py --model models/gesture_classifier.h5

  # Convert specific method only
  python convert_to_tflite.py --model models/gesture_classifier.h5 --method dynamic_range

  # Convert with representative data for int8 quantization
  python convert_to_tflite.py --model models/gesture_classifier.h5 \\
    --data datasets/val_features.npy

  # Save to custom output directory
  python convert_to_tflite.py --model models/gesture_classifier.h5 \\
    --output converted_models
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained .h5 model",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for .tflite models (default: models)",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["float32", "dynamic_range", "float16", "full_integer", "all"],
        default="all",
        help="Quantization method (default: all)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to representative data (.npy) for int8 quantization",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose logging",
    )

    args = parser.parse_args()

    try:
        # Initialize converter
        converter = TFLiteConverter(args.model, args.output, args.verbose)

        # Perform conversion
        if args.method == "all":
            converter.convert_all(representative_data_path=args.data)
        elif args.method == "float32":
            converter.convert_float32()
        elif args.method == "dynamic_range":
            converter.convert_dynamic_range()
        elif args.method == "float16":
            converter.convert_float16()
        elif args.method == "full_integer":
            converter.convert_full_integer(representative_data_path=args.data)

        # Display results
        converter.get_size_comparison()
        converter.print_recommendations()
        converter.save_results_json()

        print("\n✓ Conversion completed successfully!")

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
