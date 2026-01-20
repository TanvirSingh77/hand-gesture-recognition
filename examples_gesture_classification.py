"""
Examples: Gesture Classification Model Usage

Demonstrates various ways to use the trained gesture classification model:
1. Loading a trained model
2. Making predictions on feature vectors
3. Real-time gesture recognition with webcam
4. Batch prediction with confidence filtering
5. Performance analysis

Usage:
    python examples_gesture_classification.py --mode predict
    python examples_gesture_classification.py --mode realtime
    python examples_gesture_classification.py --mode batch
"""

import argparse
import numpy as np
import cv2
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.gesture_model import GestureClassificationModel
from src.hand_landmarks import HandLandmarkDetector
from src.feature_extractor import HandGestureFeatureExtractor


def example_load_and_predict():
    """Example 1: Load model and make predictions."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Load Model and Make Predictions")
    print("="*70 + "\n")
    
    model_path = "models/gesture_classifier.h5"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Train a model first: python train_gesture_model.py")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = GestureClassificationModel.load_model(model_path)
    print(f"✓ Model loaded successfully")
    
    # Display model info
    print("\nModel Information:")
    info = model.get_model_info()
    print(f"  Architecture: {info['architecture']}")
    print(f"  Input features: {info['input_features']}")
    print(f"  Gesture classes: {info['num_gestures']}")
    print(f"  Total parameters: {info['total_parameters']:,}")
    
    # Generate sample features
    print("\nGenerating sample feature vectors...")
    num_samples = 5
    sample_features = np.random.randn(num_samples, model.input_features).astype(np.float32)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(sample_features)
    
    # Display predictions
    print("\nPrediction Results:")
    for i, pred in enumerate(predictions):
        gesture_class = np.argmax(pred)
        confidence = pred[gesture_class]
        print(f"  Sample {i+1}: Gesture {gesture_class} (confidence: {confidence:.4f})")


def example_batch_prediction_with_confidence():
    """Example 2: Batch prediction with confidence filtering."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Prediction with Confidence Filtering")
    print("="*70 + "\n")
    
    model_path = "models/gesture_classifier.h5"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Load model
    model = GestureClassificationModel.load_model(model_path)
    
    # Generate sample features
    print("Generating 10 sample feature vectors...")
    sample_features = np.random.randn(10, model.input_features).astype(np.float32)
    
    # Predict with confidence and top-k
    print("\nMaking predictions with confidence and top-3...\n")
    results = model.predict_batch_with_confidence(
        sample_features,
        confidence_threshold=0.3,
        return_top_k=3
    )
    
    # Display results
    print("Batch Prediction Results:")
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Predicted gesture: {result['class_id']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Above threshold (0.3): {result['above_threshold']}")
        print(f"  Top-3 predictions:")
        for j, top_k_pred in enumerate(result['top_k'], 1):
            print(f"    {j}. Gesture {top_k_pred['class_id']}: {top_k_pred['confidence']:.4f}")


def example_realtime_gesture_recognition():
    """Example 3: Real-time gesture recognition with webcam."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Real-Time Gesture Recognition")
    print("="*70 + "\n")
    
    model_path = "models/gesture_classifier.h5"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print("Initializing components...")
    
    # Load model
    model = GestureClassificationModel.load_model(model_path)
    
    # Initialize detectors
    detector = HandLandmarkDetector()
    extractor = HandGestureFeatureExtractor()
    
    print("✓ Model loaded")
    print("✓ Hand detector initialized")
    print("✓ Feature extractor initialized")
    
    print("\nOpening webcam (press 'q' to quit)...\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    gesture_counts = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hand landmarks
        success, landmarks = detector.detect(frame)
        
        if success and landmarks is not None:
            # Extract features
            features = extractor.extract(landmarks)
            
            if features is not None:
                # Make prediction
                features_batch = np.array([features], dtype=np.float32)
                prediction = model.predict(features_batch)[0]
                gesture_class = np.argmax(prediction)
                confidence = float(prediction[gesture_class])
                
                # Update gesture counts
                gesture_counts[gesture_class] = gesture_counts.get(gesture_class, 0) + 1
                
                # Draw on frame
                cv2.putText(
                    frame,
                    f"Gesture: {gesture_class}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"Confidence: {confidence:.2%}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                
                # Draw landmarks
                if landmarks is not None:
                    for landmark in landmarks:
                        x = int(landmark[0] * frame.shape[1])
                        y = int(landmark[1] * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        # Display FPS
        fps = frame_count / (frame_count / 30 if frame_count > 0 else 1)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        cv2.imshow("Gesture Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nRecognition Statistics:")
    if gesture_counts:
        for gesture_class, count in sorted(gesture_counts.items()):
            percentage = count / frame_count * 100
            print(f"  Gesture {gesture_class}: {count} frames ({percentage:.1f}%)")
    print(f"Total frames processed: {frame_count}")


def example_model_comparison():
    """Example 4: Compare different model architectures."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Model Architecture Comparison")
    print("="*70 + "\n")
    
    import time
    
    # Generate test data
    test_features = np.random.randn(100, 46).astype(np.float32)
    
    architectures = ["lightweight", "balanced", "powerful"]
    results = {}
    
    print("Building and testing different architectures...\n")
    
    for arch in architectures:
        print(f"Testing {arch} architecture...")
        
        model = GestureClassificationModel(
            num_gestures=5,
            input_features=46,
            architecture=arch
        )
        model.build(verbose=False)
        model.compile()
        
        # Measure inference time
        start_time = time.time()
        for _ in range(10):
            model.predict(test_features, batch_size=32)
        inference_time = (time.time() - start_time) / 10
        
        info = model.get_model_info()
        
        results[arch] = {
            "parameters": info["total_parameters"],
            "inference_time_ms": inference_time * 1000,
            "throughput_fps": 100 / inference_time  # For 100 samples
        }
    
    # Display comparison
    print("\nArchitecture Comparison:")
    print(f"{'Architecture':<15} {'Parameters':<15} {'Inference (ms)':<20} {'Throughput (fps)':<20}")
    print("-" * 70)
    for arch in architectures:
        r = results[arch]
        print(
            f"{arch:<15} {r['parameters']:<15,} "
            f"{r['inference_time_ms']:<20.4f} "
            f"{r['throughput_fps']:<20.2f}"
        )
    print()


def example_feature_space_analysis():
    """Example 5: Analyze feature space and model decision boundaries."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Feature Space and Decision Boundaries")
    print("="*70 + "\n")
    
    model_path = "models/gesture_classifier.h5"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print("Loading model...")
    model = GestureClassificationModel.load_model(model_path)
    
    # Generate random points in feature space
    print("Generating 1000 random points in feature space...")
    n_points = 1000
    features = np.random.randn(n_points, model.input_features).astype(np.float32)
    
    # Get predictions
    print("Computing predictions...")
    predictions = model.predict(features, batch_size=100)
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Analyze distribution
    print("\nPrediction Distribution:")
    unique_classes, counts = np.unique(predicted_classes, return_counts=True)
    for gesture_class, count in zip(unique_classes, counts):
        percentage = count / n_points * 100
        print(f"  Gesture {gesture_class}: {count} samples ({percentage:.1f}%)")
    
    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {np.mean(confidences):.4f}")
    print(f"  Std Dev: {np.std(confidences):.4f}")
    print(f"  Min: {np.min(confidences):.4f}")
    print(f"  Max: {np.max(confidences):.4f}")
    print(f"  Median: {np.median(confidences):.4f}")
    
    # Distribution by confidence bins
    print(f"\nConfidence Distribution:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        count = mask.sum()
        percentage = count / n_points * 100
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {count} samples ({percentage:.1f}%)")


def main():
    """Run examples."""
    parser = argparse.ArgumentParser(
        description="Gesture Classification Model Usage Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples_gesture_classification.py --mode predict
  python examples_gesture_classification.py --mode batch
  python examples_gesture_classification.py --mode realtime
  python examples_gesture_classification.py --mode comparison
  python examples_gesture_classification.py --mode analysis
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["predict", "batch", "realtime", "comparison", "analysis"],
        default="predict",
        help="Example mode to run (default: predict)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "predict":
        example_load_and_predict()
    elif args.mode == "batch":
        example_batch_prediction_with_confidence()
    elif args.mode == "realtime":
        example_realtime_gesture_recognition()
    elif args.mode == "comparison":
        example_model_comparison()
    elif args.mode == "analysis":
        example_feature_space_analysis()


if __name__ == "__main__":
    main()
