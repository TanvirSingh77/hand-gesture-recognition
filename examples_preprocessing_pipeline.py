"""
Preprocessing Pipeline Examples

Demonstrates various usage patterns for the preprocessing pipeline:
1. Basic preprocessing with default settings
2. Custom configuration with different splits
3. Loading and applying saved preprocessing
4. Using preprocessed data for model training
5. Advanced options and troubleshooting
"""

import numpy as np
from pathlib import Path

from src.preprocessing import PreprocessingPipeline, PreprocessingConfig


def example_1_basic_preprocessing():
    """
    Example 1: Basic preprocessing with default settings.
    
    Creates a preprocessing pipeline with default configuration,
    processes the data, and saves it for later use.
    """
    print("="*70)
    print("Example 1: Basic Preprocessing")
    print("="*70)
    
    # Create pipeline with defaults
    pipeline = PreprocessingPipeline(
        data_dir="data/collected_gestures",
        verbose=True
    )
    
    # Run preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()
    
    # Save for later
    output_dir = pipeline.save_datasets("processed_data")
    
    print(f"\nResults:")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")
    print(f"  Features per sample: {X_train.shape[1]}")
    print(f"  Saved to: {output_dir}\n")


def example_2_custom_configuration():
    """
    Example 2: Custom configuration with different splits and normalization.
    
    Shows how to customize the preprocessing pipeline for specific needs:
    - Different train/val/test splits
    - Alternative normalization method
    - Custom random seed for reproducibility
    """
    print("="*70)
    print("Example 2: Custom Configuration")
    print("="*70)
    
    # Create custom configuration
    config = PreprocessingConfig(
        data_dir="data/collected_gestures",
        output_dir="processed_data_custom",
        random_seed=123,           # Different seed for different split
        train_split=0.8,           # 80% training
        val_split=0.1,             # 10% validation
        test_split=0.1,            # 10% test
        normalize_method="minmax",  # Use MinMax instead of StandardScaler
        min_samples_per_gesture=10, # Require at least 10 samples per gesture
        verbose=True
    )
    
    # Create and run pipeline
    pipeline = PreprocessingPipeline(config)
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()
    
    # Check feature ranges after MinMax normalization
    print(f"\nFeature Statistics (MinMax normalized [0, 1]):")
    print(f"  Min value: {X_train.min():.4f}")
    print(f"  Max value: {X_train.max():.4f}")
    print(f"  Mean value: {X_train.mean():.4f}")
    
    # Save
    pipeline.save_datasets()


def example_3_reproducible_preprocessing():
    """
    Example 3: Demonstrate reproducibility with fixed random seed.
    
    Shows that using the same random seed produces identical datasets
    across multiple runs, which is critical for:
    - Comparing model performance
    - Reproducing research results
    - Debugging
    """
    print("="*70)
    print("Example 3: Reproducible Preprocessing")
    print("="*70)
    
    # Run preprocessing twice with same seed
    print("Run 1:")
    pipeline1 = PreprocessingPipeline(
        data_dir="data/collected_gestures",
        random_seed=42,
        verbose=False
    )
    X1_train, X1_val, X1_test, y1_train, y1_val, y1_test, _ = pipeline1.process()
    
    print("Run 2:")
    pipeline2 = PreprocessingPipeline(
        data_dir="data/collected_gestures",
        random_seed=42,
        verbose=False
    )
    X2_train, X2_val, X2_test, y2_train, y2_val, y2_test, _ = pipeline2.process()
    
    # Verify reproducibility
    print(f"\nReproducibility Check:")
    print(f"  X_train identical: {np.array_equal(X1_train, X2_train)}")
    print(f"  y_train identical: {np.array_equal(y1_train, y2_train)}")
    print(f"  X_val identical: {np.array_equal(X1_val, X2_val)}")
    print(f"  y_val identical: {np.array_equal(y1_val, y2_val)}")
    
    # Different seed produces different split
    print("\nRun 3 (different seed):")
    pipeline3 = PreprocessingPipeline(
        data_dir="data/collected_gestures",
        random_seed=123,
        verbose=False
    )
    X3_train, _, _, y3_train, _, _, _ = pipeline3.process()
    
    print(f"  X_train different: {not np.array_equal(X1_train, X3_train)}")
    print(f"  y_train different: {not np.array_equal(y1_train, y3_train)}\n")


def example_4_load_preprocessed_data():
    """
    Example 4: Load previously preprocessed data.
    
    Shows how to load saved preprocessed datasets and use them
    without re-running the full pipeline (faster for iterative development).
    """
    print("="*70)
    print("Example 4: Loading Preprocessed Data")
    print("="*70)
    
    # First, create and save preprocessed data
    print("Creating initial preprocessed datasets...")
    pipeline = PreprocessingPipeline(
        data_dir="data/collected_gestures",
        verbose=False
    )
    pipeline.process()
    pipeline.save_datasets("processed_data")
    
    # Later, load the saved data
    print("\nLoading preprocessed datasets...")
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = (
        PreprocessingPipeline.load_datasets("processed_data")
    )
    
    print(f"\nLoaded data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape if len(X_val) > 0 else 'N/A'}")
    print(f"  X_test: {X_test.shape if len(X_test) > 0 else 'N/A'}")
    print(f"  y_train: {y_train.shape}")
    print(f"  Gesture names: {metadata.gesture_names}\n")
    
    # Load scaler for applying to new data
    print("Loading fitted scaler...")
    scaler = PreprocessingPipeline.load_scaler("processed_data")
    print(f"  Scaler type: {type(scaler).__name__}")
    print(f"  Can transform new data: {hasattr(scaler, 'transform')}\n")


def example_5_model_training_pipeline():
    """
    Example 5: Complete pipeline from preprocessing to model training.
    
    Shows how to integrate preprocessing with model training,
    demonstrating end-to-end workflow.
    """
    print("="*70)
    print("Example 5: Model Training Pipeline")
    print("="*70)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        print("This example requires scikit-learn. Install with: pip install scikit-learn")
        return
    
    # Step 1: Preprocess data
    print("Step 1: Preprocessing data...")
    pipeline = PreprocessingPipeline(
        data_dir="data/collected_gestures",
        random_seed=42,
        verbose=False
    )
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()
    
    print(f"  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")
    
    # Step 2: Train model
    print("\nStep 2: Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("  Model trained!")
    
    # Step 3: Evaluate on validation set
    print("\nStep 3: Evaluating on validation set...")
    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"  Validation accuracy: {val_accuracy:.4f}")
    
    # Step 4: Final test
    if len(X_test) > 0:
        print("\nStep 4: Final test set evaluation...")
        y_test_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"  Test accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        report = classification_report(
            y_test,
            y_test_pred,
            target_names=metadata.gesture_names
        )
        print(report)
    
    # Step 5: Feature importance
    print("\nStep 5: Top 10 Most Important Features:")
    feature_names = metadata.feature_names
    importances = clf.feature_importances_
    
    top_indices = np.argsort(importances)[-10:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {feature_names[idx]:35s}: {importances[idx]:.4f}\n")


def example_6_data_statistics():
    """
    Example 6: Analyze preprocessed data statistics.
    
    Shows how to examine and understand the preprocessed dataset
    through statistical analysis.
    """
    print("="*70)
    print("Example 6: Data Statistics")
    print("="*70)
    
    # Preprocess data
    pipeline = PreprocessingPipeline(
        data_dir="data/collected_gestures",
        verbose=False
    )
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()
    
    print(f"Dataset Overview:")
    print(f"  Total samples: {len(y_train) + len(y_val) + len(y_test)}")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"  Number of gestures: {len(metadata.gesture_names)}")
    
    print(f"\nClass Distribution (Training Set):")
    unique, counts = np.unique(y_train, return_counts=True)
    for gesture_idx, count in zip(unique, counts):
        gesture_name = metadata.gesture_names[gesture_idx]
        percentage = count / len(y_train) * 100
        print(f"  {gesture_name:20s}: {count:4d} samples ({percentage:5.1f}%)")
    
    print(f"\nFeature Statistics (Training Set):")
    print(f"  Mean: {X_train.mean(axis=0)[:5]}... (first 5)")
    print(f"  Std:  {X_train.std(axis=0)[:5]}... (first 5)")
    print(f"  Min:  {X_train.min(axis=0)[:5]}... (first 5)")
    print(f"  Max:  {X_train.max(axis=0)[:5]}... (first 5)")
    
    print(f"\nNormalization Method:")
    print(f"  Method: {metadata.normalize_method}")
    if metadata.scaler_mean is not None:
        print(f"  Mean (StandardScaler): {metadata.scaler_mean[:5]}... (first 5)")
        print(f"  Scale: {metadata.scaler_scale[:5]}... (first 5)")
    
    print(f"\nProcessing Info:")
    print(f"  Random seed: {metadata.random_seed}")
    print(f"  Aggregation method: {metadata.aggregation_method}")
    print(f"  Processed timestamp: {metadata.processed_timestamp}\n")


def example_7_applying_preprocessing_to_new_data():
    """
    Example 7: Apply fitted preprocessing to new data.
    
    Shows how to apply preprocessing (normalization) that was fitted
    on training data to new inference data.
    """
    print("="*70)
    print("Example 7: Applying Preprocessing to New Data")
    print("="*70)
    
    # Step 1: Preprocess and save
    print("Step 1: Initial preprocessing...")
    pipeline = PreprocessingPipeline(
        data_dir="data/collected_gestures",
        verbose=False
    )
    X_train, _, _, _, _, _, metadata = pipeline.process()
    pipeline.save_datasets("processed_data")
    
    print(f"  Training data range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    
    # Step 2: Load scaler
    print("\nStep 2: Loading fitted scaler...")
    scaler = PreprocessingPipeline.load_scaler("processed_data")
    
    # Step 3: Simulate new raw features
    print("\nStep 3: Simulating new raw features...")
    # In real scenario, these would come from feature_extractor.extract()
    new_features = np.random.rand(10, 46) * 100  # Random features in arbitrary range
    print(f"  New features range: [{new_features.min():.4f}, {new_features.max():.4f}]")
    
    # Step 4: Apply preprocessing
    print("\nStep 4: Applying fitted preprocessing...")
    new_features_normalized = scaler.transform(new_features)
    print(f"  Normalized range: [{new_features_normalized.min():.4f}, {new_features_normalized.max():.4f}]")
    
    print(f"\nComparison:")
    print(f"  Original features range: [{new_features.min():.4f}, {new_features.max():.4f}]")
    print(f"  Normalized features range: [{new_features_normalized.min():.4f}, {new_features_normalized.max():.4f}]")
    print(f"  Match training normalization: Yes\n")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE EXAMPLES")
    print("="*70 + "\n")
    
    examples = [
        ("Basic Preprocessing", example_1_basic_preprocessing),
        ("Custom Configuration", example_2_custom_configuration),
        ("Reproducibility", example_3_reproducible_preprocessing),
        ("Load Preprocessed Data", example_4_load_preprocessed_data),
        ("Model Training", example_5_model_training_pipeline),
        ("Data Statistics", example_6_data_statistics),
        ("Apply to New Data", example_7_applying_preprocessing_to_new_data),
    ]
    
    print("Available Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nTo run a specific example, call it directly:")
    print("  example_1_basic_preprocessing()")
    print("  example_2_custom_configuration()")
    print("  ... and so on")
    
    print("\nOr run all examples:")
    print("  python examples_preprocessing_pipeline.py\n")


if __name__ == "__main__":
    main()
    
    # Uncomment to run examples automatically:
    # example_1_basic_preprocessing()
    # example_2_custom_configuration()
    # example_3_reproducible_preprocessing()
    # example_4_load_preprocessed_data()
    # example_5_model_training_pipeline()
    # example_6_data_statistics()
    # example_7_applying_preprocessing_to_new_data()
