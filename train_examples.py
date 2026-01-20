"""Examples of training gesture recognition models with collected data.

This file demonstrates how to:
- Load collected gesture data
- Prepare data for training
- Train different types of models
- Evaluate model performance
"""

import sys
from pathlib import Path
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_utils import GestureDataLoader


# ============================================================================
# Example 1: Basic Data Loading
# ============================================================================

def example_load_data():
    """Load collected gesture data."""
    print("\n" + "="*60)
    print("Example 1: Loading Gesture Data")
    print("="*60)
    
    loader = GestureDataLoader()
    
    # Print statistics
    loader.print_statistics()
    
    # Get all data
    all_gestures = loader.get_all_gestures()
    print(f"Loaded {len(all_gestures)} gesture classes")
    
    for gesture_name, samples in all_gestures.items():
        print(f"  {gesture_name}: {len(samples)} samples")


# ============================================================================
# Example 2: Feature Extraction
# ============================================================================

def example_feature_extraction():
    """Extract features from gesture data."""
    print("\n" + "="*60)
    print("Example 2: Feature Extraction")
    print("="*60)
    
    loader = GestureDataLoader()
    
    # Get feature vectors (using first frame only)
    X, y, gesture_names = loader.get_feature_vectors(aggregate_frames="first")
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    print(f"Feature dimension: {X.shape[1]} (21 landmarks × 2 coordinates)")
    print(f"Gesture classes: {gesture_names}")
    print(f"Number of samples: {len(X)}")
    
    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for gesture_id, gesture_name in enumerate(gesture_names):
        count = counts[gesture_id] if gesture_id < len(counts) else 0
        print(f"  {gesture_name}: {count} samples")


# ============================================================================
# Example 3: Train with Scikit-Learn (Random Forest)
# ============================================================================

def example_sklearn_random_forest():
    """Train a Random Forest classifier with scikit-learn."""
    print("\n" + "="*60)
    print("Example 3: Random Forest (Scikit-Learn)")
    print("="*60)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("scikit-learn not installed. Install with: pip install scikit-learn")
        return
    
    loader = GestureDataLoader()
    X, y, gesture_names = loader.get_feature_vectors()
    
    if len(X) == 0:
        print("No data available for training")
        return
    
    print(f"Training data size: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Detailed report
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=gesture_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


# ============================================================================
# Example 4: Train with Scikit-Learn (SVM)
# ============================================================================

def example_sklearn_svm():
    """Train a Support Vector Machine with scikit-learn."""
    print("\n" + "="*60)
    print("Example 4: Support Vector Machine (Scikit-Learn)")
    print("="*60)
    
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("scikit-learn not installed. Install with: pip install scikit-learn")
        return
    
    loader = GestureDataLoader()
    X, y, gesture_names = loader.get_feature_vectors()
    
    if len(X) == 0:
        print("No data available for training")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize data (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training data: {X_train_scaled.shape}")
    
    # Train model
    print("Training SVM...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")


# ============================================================================
# Example 5: Train with TensorFlow/Keras
# ============================================================================

def example_tensorflow_neural_network():
    """Train a neural network with TensorFlow."""
    print("\n" + "="*60)
    print("Example 5: Neural Network (TensorFlow/Keras)")
    print("="*60)
    
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("TensorFlow not installed. Install with: pip install tensorflow")
        return
    
    loader = GestureDataLoader()
    X, y, gesture_names = loader.get_feature_vectors()
    
    if len(X) == 0:
        print("No data available for training")
        return
    
    # Convert labels to one-hot
    num_classes = len(gesture_names)
    y_categorical = keras.utils.to_categorical(y, num_classes)
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")
    
    # Build model
    print("\nBuilding model...")
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")


# ============================================================================
# Example 6: Cross-Validation
# ============================================================================

def example_cross_validation():
    """Perform cross-validation on gesture data."""
    print("\n" + "="*60)
    print("Example 6: Cross-Validation")
    print("="*60)
    
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("scikit-learn not installed. Install with: pip install scikit-learn")
        return
    
    loader = GestureDataLoader()
    X, y, gesture_names = loader.get_feature_vectors()
    
    if len(X) == 0:
        print("No data available for training")
        return
    
    print(f"Data shape: {X.shape}")
    
    # Perform 5-fold cross-validation
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=5)
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")


# ============================================================================
# Example 7: Aggregate Frames Comparison
# ============================================================================

def example_aggregate_methods():
    """Compare different frame aggregation methods."""
    print("\n" + "="*60)
    print("Example 7: Frame Aggregation Methods")
    print("="*60)
    
    loader = GestureDataLoader()
    
    methods = ["first", "mean", "flatten"]
    
    for method in methods:
        X, y, gesture_names = loader.get_feature_vectors(
            aggregate_frames=method
        )
        
        print(f"\nMethod: {method}")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Number of samples: {len(X)}")
        print(f"  Feature dimension: {X.shape[1]}")


# ============================================================================
# Example 8: Data Export
# ============================================================================

def example_export_data():
    """Export data to different formats."""
    print("\n" + "="*60)
    print("Example 8: Data Export")
    print("="*60)
    
    loader = GestureDataLoader()
    
    # Export to CSV
    print("\nExporting to CSV...")
    loader.export_to_csv("data/gestures_landmarks.csv")
    print("✓ Saved to: data/gestures_landmarks.csv")
    
    # Export to NumPy
    print("\nExporting to NumPy...")
    X, y = loader.export_to_numpy("data/gestures_landmarks.npz")
    if X is not None:
        print(f"✓ Saved to: data/gestures_landmarks.npz")
        print(f"  Features shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GESTURE RECOGNITION - TRAINING EXAMPLES")
    print("="*70)
    
    examples = [
        ("1", "Load and explore data", example_load_data),
        ("2", "Feature extraction", example_feature_extraction),
        ("3", "Random Forest (Scikit-Learn)", example_sklearn_random_forest),
        ("4", "SVM (Scikit-Learn)", example_sklearn_svm),
        ("5", "Neural Network (TensorFlow)", example_tensorflow_neural_network),
        ("6", "Cross-validation", example_cross_validation),
        ("7", "Aggregation methods", example_aggregate_methods),
        ("8", "Data export", example_export_data),
    ]
    
    print("\nAvailable examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    print("  0. Run all")
    print("  q. Quit")
    
    while True:
        choice = input("\nSelect example (0-8, q): ").strip()
        
        if choice == 'q':
            print("Exiting...")
            break
        
        elif choice == '0':
            for num, name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"Error in example {num}: {e}")
        
        else:
            for num, name, func in examples:
                if num == choice:
                    try:
                        func()
                    except Exception as e:
                        print(f"Error: {e}")
                    break
            else:
                print("Invalid choice")


if __name__ == "__main__":
    main()
