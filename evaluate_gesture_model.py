"""
Model Evaluation Script for Gesture Classification

Comprehensive evaluation of trained gesture classification models with:
- Accuracy metrics (overall, per-class)
- Confusion matrix calculation and visualization
- Detailed classification reports
- Results logging to file and console
- Performance analysis

Usage:
    python evaluate_gesture_model.py --model models/gesture_classifier.h5 \
                                     --data datasets/val_features.npy \
                                     --labels datasets/val_labels.npy \
                                     --output eval_results.txt
    
    python evaluate_gesture_model.py --help

Output:
    - Console: Detailed evaluation metrics and confusion matrix
    - File: eval_results.txt with complete analysis
    - Optional: Confusion matrix visualization (requires matplotlib)
"""

import argparse
import numpy as np
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Try to import optional visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.gesture_model import GestureClassificationModel


# Configure logging
class FormattedLogger:
    """Custom logger with formatted output."""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        """
        Initialize logger.
        
        Args:
            log_file (str): Path to log file (optional)
            verbose (bool): Print to console
        """
        self.log_file = log_file
        self.verbose = verbose
        self.logs = []
        
        # Create log file if needed
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            with open(log_file, 'w') as f:
                f.write(f"Model Evaluation Results\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
    
    def log(self, message: str, section: bool = False):
        """
        Log a message.
        
        Args:
            message (str): Message to log
            section (bool): If True, format as section header
        """
        self.logs.append(message)
        
        if self.verbose:
            if section:
                print(f"\n{'=' * 80}")
                print(message)
                print('=' * 80)
            else:
                print(message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                if section:
                    f.write(f"\n{'=' * 80}\n")
                    f.write(message + "\n")
                    f.write('=' * 80 + "\n")
                else:
                    f.write(message + "\n")
    
    def save_summary(self, data: Dict):
        """Save summary as JSON."""
        if self.log_file:
            json_path = self.log_file.replace('.txt', '_summary.json')
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)


def load_data(features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load validation data.
    
    Args:
        features_path (str): Path to features (npy file)
        labels_path (str): Path to labels (npy file)
    
    Returns:
        Tuple: (features, labels) as numpy arrays
    
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If data shapes don't match
    """
    print(f"Loading data from {features_path}...")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Data mismatch: {features.shape[0]} features, "
            f"{labels.shape[0]} labels"
        )
    
    print(f"✓ Loaded {len(features)} samples")
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}\n")
    
    return features, labels


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true (np.ndarray): True class indices (one-hot or indices)
        y_pred (np.ndarray): Predicted class indices
        num_classes (int): Number of classes
    
    Returns:
        np.ndarray: Confusion matrix (num_classes x num_classes)
    """
    # Convert one-hot to indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Fill confusion matrix
    for true_class, pred_class in zip(y_true, y_pred):
        cm[int(true_class), int(pred_class)] += 1
    
    return cm


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        y_true (np.ndarray): True labels (one-hot or indices)
        y_pred (np.ndarray): Predicted class indices
        num_classes (int): Number of classes
    
    Returns:
        dict: Dictionary with metrics
    """
    # Convert one-hot to indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.astype(int)
    
    # Overall accuracy
    accuracy = np.mean(y_true_indices == y_pred)
    
    # Per-class metrics
    per_class_metrics = {}
    for class_id in range(num_classes):
        mask = y_true_indices == class_id
        if mask.sum() > 0:
            class_accuracy = np.mean(y_pred[mask] == class_id)
            class_recall = class_accuracy
            
            # Precision: TP / (TP + FP)
            pred_mask = y_pred == class_id
            if pred_mask.sum() > 0:
                tp = np.sum((y_true_indices == class_id) & (y_pred == class_id))
                precision = tp / pred_mask.sum()
            else:
                precision = 0.0
            
            # F1 score
            if precision + class_recall > 0:
                f1 = 2 * (precision * class_recall) / (precision + class_recall)
            else:
                f1 = 0.0
            
            per_class_metrics[int(class_id)] = {
                "accuracy": float(class_accuracy),
                "precision": float(precision),
                "recall": float(class_recall),
                "f1_score": float(f1),
                "samples": int(mask.sum())
            }
    
    return {
        "overall_accuracy": float(accuracy),
        "per_class": per_class_metrics,
        "num_classes": num_classes,
        "total_samples": len(y_true_indices)
    }


def format_confusion_matrix_str(cm: np.ndarray, class_names: Optional[List[str]] = None) -> str:
    """
    Format confusion matrix as string.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): Optional class names
    
    Returns:
        str: Formatted confusion matrix string
    """
    lines = []
    lines.append("\nConfusion Matrix:")
    
    num_classes = cm.shape[0]
    
    # Header
    header = "Actual \\ Predicted"
    for i in range(num_classes):
        if class_names:
            header += f" {class_names[i][:8]:>10}"
        else:
            header += f" Cls{i:>6}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for i in range(num_classes):
        if class_names:
            row = f"{class_names[i][:15]:>15}"
        else:
            row = f"Class {i:>10}"
        
        for j in range(num_classes):
            row += f" {cm[i, j]:>10}"
        lines.append(row)
    
    return "\n".join(lines)


def visualize_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    class_names: Optional[List[str]] = None
):
    """
    Create confusion matrix visualization.
    
    Args:
        cm (np.ndarray): Confusion matrix
        output_path (str): Path to save figure
        class_names (list): Optional class names
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ Matplotlib not available - skipping visualization")
        return
    
    try:
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix for display
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=cm,  # Show actual counts
            fmt='d',
            cmap='Blues',
            xticklabels=class_names or [f'Class {i}' for i in range(cm.shape[0])],
            yticklabels=class_names or [f'Class {i}' for i in range(cm.shape[0])],
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {output_path}")
        plt.close()
    
    except Exception as e:
        print(f"⚠ Could not save visualization: {e}")


def format_metrics_report(metrics: Dict, logger: FormattedLogger) -> str:
    """
    Format metrics as readable report.
    
    Args:
        metrics (dict): Metrics from compute_metrics
        logger (FormattedLogger): Logger instance
    
    Returns:
        str: Formatted report
    """
    lines = []
    
    # Overall accuracy
    lines.append(f"\nOVERALL ACCURACY: {metrics['overall_accuracy']:.4f} "
                f"({metrics['overall_accuracy']*100:.2f}%)")
    lines.append(f"Total Samples: {metrics['total_samples']}")
    
    # Per-class metrics
    lines.append("\n\nPER-CLASS METRICS:")
    lines.append("-" * 90)
    lines.append(f"{'Class':<15} {'Accuracy':<15} {'Precision':<15} "
                f"{'Recall':<15} {'F1 Score':<15} {'Samples':<15}")
    lines.append("-" * 90)
    
    for class_id in sorted(metrics['per_class'].keys()):
        metrics_data = metrics['per_class'][class_id]
        lines.append(
            f"Class {class_id:<9} "
            f"{metrics_data['accuracy']:<15.4f} "
            f"{metrics_data['precision']:<15.4f} "
            f"{metrics_data['recall']:<15.4f} "
            f"{metrics_data['f1_score']:<15.4f} "
            f"{metrics_data['samples']:<15d}"
        )
    
    lines.append("-" * 90)
    
    report = "\n".join(lines)
    
    # Log to file
    for line in lines:
        logger.log(line)
    
    return report


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained gesture classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_gesture_model.py \\
    --model models/gesture_classifier.h5 \\
    --data datasets/val_features.npy \\
    --labels datasets/val_labels.npy
    
  python evaluate_gesture_model.py \\
    --model models/gesture_classifier.h5 \\
    --data datasets/val_features.npy \\
    --labels datasets/val_labels.npy \\
    --output eval_results.txt \\
    --plot confusion_matrix.png
        """
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.h5 file)"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to validation features (.npy file)"
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to validation labels (.npy file)"
    )
    parser.add_argument(
        "--output",
        default="eval_results.txt",
        help="Path to save evaluation results (default: eval_results.txt)"
    )
    parser.add_argument(
        "--plot",
        help="Path to save confusion matrix plot (optional)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for predictions (default: 32)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print to console (default: True)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize logger
        logger = FormattedLogger(log_file=args.output, verbose=args.verbose)
        logger.log("Starting Model Evaluation", section=True)
        
        # Load model
        logger.log(f"\nLoading model from {args.model}...")
        try:
            model = GestureClassificationModel.load_model(args.model)
            logger.log("✓ Model loaded successfully")
            
            # Get model info
            info = model.get_model_info()
            logger.log(f"\nModel Information:")
            logger.log(f"  Architecture: {info['architecture']}")
            logger.log(f"  Input Features: {info['input_features']}")
            logger.log(f"  Gesture Classes: {info['num_gestures']}")
            logger.log(f"  Total Parameters: {info['total_parameters']:,}")
        except Exception as e:
            logger.log(f"✗ Error loading model: {e}")
            return 1
        
        # Load validation data
        try:
            val_features, val_labels = load_data(args.data, args.labels)
        except Exception as e:
            logger.log(f"✗ Error loading data: {e}")
            return 1
        
        # Make predictions
        logger.log(f"\nMaking predictions on {len(val_features)} samples...")
        predictions = model.predict(val_features, batch_size=args.batch_size)
        predicted_classes = np.argmax(predictions, axis=1)
        logger.log(f"✓ Predictions complete")
        
        # Compute metrics
        logger.log(f"\nComputing evaluation metrics...")
        metrics = compute_metrics(val_labels, predicted_classes, model.num_gestures)
        logger.log(f"✓ Metrics computed")
        
        # Print metrics report
        logger.log("", section=True)
        logger.log("EVALUATION METRICS", section=True)
        format_metrics_report(metrics, logger)
        
        # Compute confusion matrix
        logger.log(f"\nComputing confusion matrix...")
        cm = compute_confusion_matrix(val_labels, predicted_classes, model.num_gestures)
        logger.log(f"✓ Confusion matrix computed")
        
        # Format and print confusion matrix
        cm_str = format_confusion_matrix_str(cm)
        logger.log("", section=True)
        logger.log("CONFUSION MATRIX", section=True)
        logger.log(cm_str)
        
        # Visualize if matplotlib available
        if args.plot and MATPLOTLIB_AVAILABLE:
            logger.log(f"\nCreating confusion matrix visualization...")
            visualize_confusion_matrix(cm, args.plot)
        
        # Summary statistics
        logger.log("\n", section=True)
        logger.log("SUMMARY STATISTICS", section=True)
        
        correct_predictions = np.sum(np.diag(cm))
        total_predictions = np.sum(cm)
        accuracy = correct_predictions / total_predictions
        
        logger.log(f"\nCorrect Predictions: {correct_predictions}/{total_predictions}")
        logger.log(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class summary
        logger.log(f"\nPer-Class Summary:")
        best_class = max(metrics['per_class'].items(), 
                        key=lambda x: x[1]['accuracy'])
        worst_class = min(metrics['per_class'].items(), 
                         key=lambda x: x[1]['accuracy'])
        
        logger.log(f"  Best Performance: Class {best_class[0]} "
                  f"({best_class[1]['accuracy']:.4f} accuracy)")
        logger.log(f"  Worst Performance: Class {worst_class[0]} "
                  f"({worst_class[1]['accuracy']:.4f} accuracy)")
        
        # Class distribution
        logger.log(f"\nClass Distribution in Validation Set:")
        for class_id in sorted(metrics['per_class'].keys()):
            samples = metrics['per_class'][class_id]['samples']
            percentage = samples / metrics['total_samples'] * 100
            logger.log(f"  Class {class_id}: {samples} samples ({percentage:.1f}%)")
        
        # Save summary to JSON
        summary_data = {
            "model": args.model,
            "evaluation_date": datetime.now().isoformat(),
            "overall_accuracy": metrics['overall_accuracy'],
            "total_samples": metrics['total_samples'],
            "per_class_metrics": metrics['per_class'],
            "confusion_matrix": cm.tolist()
        }
        logger.save_summary(summary_data)
        
        # Final status
        logger.log("\n", section=True)
        logger.log("EVALUATION COMPLETE", section=True)
        logger.log(f"Results saved to: {args.output}")
        if args.plot:
            logger.log(f"Confusion matrix plot saved to: {args.plot}")
        logger.log(f"Summary JSON saved to: {args.output.replace('.txt', '_summary.json')}")
        
        return 0
    
    except Exception as e:
        logger.log(f"✗ Evaluation failed: {str(e)}", section=True)
        import traceback
        logger.log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
