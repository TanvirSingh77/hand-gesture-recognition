"""
Gesture Classification Model Training Script

Complete end-to-end training pipeline demonstrating:
- Loading preprocessed data
- Building and compiling the neural network
- Training with callbacks and validation
- Evaluating model performance
- Saving trained model
- Making predictions on new data

This script serves as both a training utility and reference implementation.

Usage:
    python train_gesture_model.py --architecture lightweight --epochs 100
    python train_gesture_model.py --help
"""

import argparse
import numpy as np
import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.gesture_model import GestureClassificationModel


def load_preprocessed_data(data_dir: str = "datasets") -> dict:
    """
    Load preprocessed training and validation data.
    
    Args:
        data_dir (str): Directory containing preprocessed data
    
    Returns:
        dict: Dictionary with train/val features and labels
    
    Raises:
        FileNotFoundError: If required data files not found
    """
    logger.info(f"Loading preprocessed data from {data_dir}")
    
    required_files = [
        "train_features.npy",
        "train_labels.npy",
        "val_features.npy",
        "val_labels.npy"
    ]
    
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Required data file not found: {filepath}\n"
                f"Run preprocessing pipeline first: python examples_preprocessing_pipeline.py"
            )
    
    # Load data
    train_features = np.load(os.path.join(data_dir, "train_features.npy"))
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    val_features = np.load(os.path.join(data_dir, "val_features.npy"))
    val_labels = np.load(os.path.join(data_dir, "val_labels.npy"))
    
    logger.info(f"Loaded training data: {train_features.shape}, {train_labels.shape}")
    logger.info(f"Loaded validation data: {val_features.shape}, {val_labels.shape}")
    
    return {
        "train_features": train_features,
        "train_labels": train_labels,
        "val_features": val_features,
        "val_labels": val_labels
    }


def create_and_train_model(
    num_gestures: int,
    input_features: int = 46,
    architecture: str = "lightweight",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_save_path: str = "models/gesture_classifier.h5",
    data_dir: str = "datasets"
) -> GestureClassificationModel:
    """
    Create, build, compile, and train gesture classification model.
    
    Args:
        num_gestures (int): Number of gesture classes
        input_features (int): Number of input features
        architecture (str): Model architecture ('lightweight', 'balanced', 'powerful')
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Initial learning rate
        model_save_path (str): Path to save trained model
        data_dir (str): Directory with preprocessed data
    
    Returns:
        GestureClassificationModel: Trained and saved model
    """
    
    # Load data
    data = load_preprocessed_data(data_dir)
    
    # Create model
    logger.info(f"Creating {architecture} model for {num_gestures} gesture classes")
    model = GestureClassificationModel(
        num_gestures=num_gestures,
        input_features=input_features,
        model_name="gesture_classifier",
        architecture=architecture
    )
    
    # Build model
    logger.info("Building model architecture")
    model.build(verbose=True)
    
    # Compile model
    logger.info(f"Compiling model with learning rate {learning_rate}")
    model.compile(learning_rate=learning_rate)
    
    # Train model
    logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
    result = model.train(
        data["train_features"],
        data["train_labels"],
        data["val_features"],
        data["val_labels"],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        class_weight_strategy="balanced",
        model_save_path=model_save_path,
        verbose=1
    )
    
    # Evaluate on validation set
    logger.info("Evaluating model on validation set")
    metrics = model.evaluate(
        data["val_features"],
        data["val_labels"],
        verbose=0
    )
    
    logger.info(f"Validation metrics: {metrics}")
    
    # Display training summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Architecture: {architecture}")
    print(f"Epochs trained: {result['metadata']['epochs_trained']}")
    print(f"Final training loss: {result['metadata']['final_train_loss']:.6f}")
    print(f"Final validation loss: {result['metadata']['final_val_loss']:.6f}")
    print(f"Final training accuracy: {result['metadata']['final_train_accuracy']:.4f}")
    print(f"Final validation accuracy: {result['metadata']['final_val_accuracy']:.4f}")
    print(f"Model saved to: {model_save_path}")
    print("="*70 + "\n")
    
    return model


def evaluate_model(
    model: GestureClassificationModel,
    features: np.ndarray,
    labels: np.ndarray,
    gesture_names: list = None
) -> dict:
    """
    Evaluate model and print detailed metrics.
    
    Args:
        model (GestureClassificationModel): Trained model
        features (np.ndarray): Test/validation features
        labels (np.ndarray): Test/validation labels
        gesture_names (list): Optional gesture class names
    
    Returns:
        dict: Detailed evaluation results
    """
    logger.info("Evaluating model on provided data")
    
    # Get metrics
    metrics = model.evaluate(features, labels, verbose=0)
    
    # Get predictions
    predictions = model.predict(features)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    
    # Compute per-class accuracy
    num_gestures = model.num_gestures
    per_class_accuracy = {}
    
    for gesture_id in range(num_gestures):
        mask = true_classes == gesture_id
        if mask.sum() > 0:
            accuracy = (predicted_classes[mask] == gesture_id).mean()
            gesture_name = gesture_names[gesture_id] if gesture_names else f"Gesture {gesture_id}"
            per_class_accuracy[gesture_name] = accuracy
    
    results = {
        "overall_metrics": metrics,
        "per_class_accuracy": per_class_accuracy,
        "total_samples": len(features),
        "correct_predictions": (predicted_classes == true_classes).sum(),
        "overall_accuracy": (predicted_classes == true_classes).mean()
    }
    
    # Print results
    print("\n" + "="*70)
    print("DETAILED EVALUATION RESULTS")
    print("="*70)
    print(f"Total samples: {results['total_samples']}")
    print(f"Correct predictions: {results['correct_predictions']}/{results['total_samples']}")
    print(f"Overall accuracy: {results['overall_accuracy']:.4f}")
    print("\nPer-class accuracy:")
    for gesture_name, accuracy in results['per_class_accuracy'].items():
        print(f"  {gesture_name}: {accuracy:.4f}")
    print("="*70 + "\n")
    
    return results


def demo_predictions(
    model: GestureClassificationModel,
    features: np.ndarray,
    labels: np.ndarray = None,
    gesture_names: list = None,
    num_samples: int = 5
) -> None:
    """
    Demonstrate model predictions on sample data.
    
    Args:
        model (GestureClassificationModel): Trained model
        features (np.ndarray): Input features
        labels (np.ndarray): Optional true labels
        gesture_names (list): Optional gesture class names
        num_samples (int): Number of samples to demonstrate
    """
    logger.info(f"Demonstrating predictions on {num_samples} samples")
    
    # Select random samples
    indices = np.random.choice(len(features), min(num_samples, len(features)), replace=False)
    
    # Get predictions
    predictions = model.predict_batch_with_confidence(
        features[indices],
        confidence_threshold=0.5,
        return_top_k=3
    )
    
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    for i, (idx, pred) in enumerate(zip(indices, predictions)):
        print(f"\nSample {i+1} (index {idx}):")
        
        if labels is not None:
            true_class = np.argmax(labels[idx])
            true_name = gesture_names[true_class] if gesture_names else f"Gesture {true_class}"
            print(f"  True class: {true_name}")
        
        predicted_class = pred["class_id"]
        predicted_name = gesture_names[predicted_class] if gesture_names else f"Gesture {predicted_class}"
        print(f"  Predicted class: {predicted_name}")
        print(f"  Confidence: {pred['confidence']:.4f}")
        print(f"  Above threshold: {pred['above_threshold']}")
        print(f"  Top-3 predictions:")
        for j, top_pred in enumerate(pred["top_k"], 1):
            class_name = gesture_names[top_pred["class_id"]] if gesture_names else f"Gesture {top_pred['class_id']}"
            print(f"    {j}. {class_name}: {top_pred['confidence']:.4f}")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train gesture classification neural network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_gesture_model.py --architecture lightweight --epochs 100
  python train_gesture_model.py --architecture powerful --batch_size 16
  python train_gesture_model.py --learning_rate 0.0005 --epochs 200
        """
    )
    
    parser.add_argument(
        "--architecture",
        choices=["lightweight", "balanced", "powerful"],
        default="lightweight",
        help="Model architecture preset (default: lightweight)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--model_path",
        default="models/gesture_classifier.h5",
        help="Path to save trained model (default: models/gesture_classifier.h5)"
    )
    parser.add_argument(
        "--data_dir",
        default="datasets",
        help="Directory with preprocessed data (default: datasets)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Show prediction demonstrations after training"
    )
    
    args = parser.parse_args()
    
    try:
        # Check if preprocessed data exists
        if not os.path.exists(args.data_dir):
            logger.error(f"Data directory not found: {args.data_dir}")
            logger.info("Please run preprocessing first: python examples_preprocessing_pipeline.py")
            return
        
        # Load data to get dimensions
        data = load_preprocessed_data(args.data_dir)
        num_gestures = data["train_labels"].shape[1]
        input_features = data["train_features"].shape[1]
        
        logger.info(f"Dataset info: {num_gestures} gestures, {input_features} features")
        logger.info(f"Training samples: {len(data['train_features'])}")
        logger.info(f"Validation samples: {len(data['val_features'])}")
        
        # Create and train model
        model = create_and_train_model(
            num_gestures=num_gestures,
            input_features=input_features,
            architecture=args.architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_save_path=args.model_path,
            data_dir=args.data_dir
        )
        
        # Evaluate on validation set
        evaluate_model(
            model,
            data["val_features"],
            data["val_labels"]
        )
        
        # Demo predictions if requested
        if args.demo:
            demo_predictions(
                model,
                data["val_features"],
                data["val_labels"],
                num_samples=10
            )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
