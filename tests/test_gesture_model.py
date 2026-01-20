"""
Unit Tests for Gesture Classification Model

Comprehensive test suite covering:
- Model building with different architectures
- Model compilation and training
- Prediction and evaluation
- Model persistence (save/load)
- Edge cases and error handling
- Performance characteristics
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tempfile
import os
import json
from pathlib import Path

from src.gesture_model import GestureClassificationModel


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    
    num_samples = 100
    input_features = 46
    num_gestures = 5
    
    # Generate random features
    train_features = np.random.randn(num_samples, input_features).astype(np.float32)
    train_labels = to_categorical(
        np.random.randint(0, num_gestures, num_samples),
        num_classes=num_gestures
    )
    
    val_features = np.random.randn(20, input_features).astype(np.float32)
    val_labels = to_categorical(
        np.random.randint(0, num_gestures, 20),
        num_classes=num_gestures
    )
    
    return {
        "train_features": train_features,
        "train_labels": train_labels,
        "val_features": val_features,
        "val_labels": val_labels,
        "input_features": input_features,
        "num_gestures": num_gestures
    }


@pytest.fixture
def model():
    """Create a fresh model instance."""
    return GestureClassificationModel(
        num_gestures=5,
        input_features=46,
        model_name="test_model",
        architecture="lightweight"
    )


@pytest.fixture
def built_model(model):
    """Create a built and compiled model."""
    model.build(verbose=False)
    model.compile()
    return model


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestModelInitialization:
    """Test model initialization and configuration."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        model = GestureClassificationModel(num_gestures=10)
        
        assert model.num_gestures == 10
        assert model.input_features == 46
        assert model.model_name == "gesture_classifier"
        assert model.architecture == "lightweight"
        assert model.model is None
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = GestureClassificationModel(
            num_gestures=8,
            input_features=30,
            model_name="custom_model",
            architecture="powerful"
        )
        
        assert model.num_gestures == 8
        assert model.input_features == 30
        assert model.model_name == "custom_model"
        assert model.architecture == "powerful"
    
    def test_init_invalid_num_gestures(self):
        """Test initialization with invalid number of gestures."""
        with pytest.raises(ValueError):
            GestureClassificationModel(num_gestures=1)
        
        with pytest.raises(ValueError):
            GestureClassificationModel(num_gestures=0)
    
    def test_init_invalid_input_features(self):
        """Test initialization with invalid number of features."""
        with pytest.raises(ValueError):
            GestureClassificationModel(num_gestures=5, input_features=0)
        
        with pytest.raises(ValueError):
            GestureClassificationModel(num_gestures=5, input_features=-5)


# ============================================================================
# MODEL BUILDING TESTS
# ============================================================================

class TestModelBuilding:
    """Test model architecture and building."""
    
    def test_build_lightweight(self, model):
        """Test building lightweight architecture."""
        model.build(verbose=False)
        
        assert model.model is not None
        assert model.model.input_shape == (None, 46)
        assert model.model.output_shape == (None, 5)
        assert model.model.count_params() < 30000
    
    def test_build_balanced(self):
        """Test building balanced architecture."""
        model = GestureClassificationModel(
            num_gestures=5,
            architecture="balanced"
        )
        model.build(verbose=False)
        
        assert model.model is not None
        assert model.model.count_params() < 70000
        assert model.model.count_params() > 20000
    
    def test_build_powerful(self):
        """Test building powerful architecture."""
        model = GestureClassificationModel(
            num_gestures=5,
            architecture="powerful"
        )
        model.build(verbose=False)
        
        assert model.model is not None
        assert model.model.count_params() > 50000
    
    def test_build_invalid_architecture(self, model):
        """Test building with invalid architecture."""
        model.architecture = "invalid_arch"
        
        with pytest.raises(ValueError):
            model.build(verbose=False)
    
    def test_build_model_structure(self, model):
        """Test that built model has expected layer structure."""
        model.build(verbose=False)
        
        # Check for expected layer types
        layer_types = [type(layer).__name__ for layer in model.model.layers]
        
        assert "InputLayer" in layer_types or "InputLayer" in str(model.model.layers[0])
        assert "Dense" in layer_types
        assert "Dropout" in layer_types
        assert "BatchNormalization" in layer_types


# ============================================================================
# MODEL COMPILATION TESTS
# ============================================================================

class TestModelCompilation:
    """Test model compilation with different optimizers."""
    
    def test_compile_adam(self, built_model):
        """Test compilation with Adam optimizer."""
        built_model.compile(learning_rate=0.001, optimizer_type="adam")
        
        assert built_model.model.optimizer is not None
        assert isinstance(built_model.model.optimizer, tf.keras.optimizers.Adam)
    
    def test_compile_sgd(self, built_model):
        """Test compilation with SGD optimizer."""
        built_model.compile(learning_rate=0.01, optimizer_type="sgd")
        
        assert isinstance(built_model.model.optimizer, tf.keras.optimizers.SGD)
    
    def test_compile_rmsprop(self, built_model):
        """Test compilation with RMSprop optimizer."""
        built_model.compile(learning_rate=0.001, optimizer_type="rmsprop")
        
        assert isinstance(built_model.model.optimizer, tf.keras.optimizers.RMSprop)
    
    def test_compile_invalid_optimizer(self, built_model):
        """Test compilation with invalid optimizer."""
        with pytest.raises(ValueError):
            built_model.compile(optimizer_type="invalid_optimizer")
    
    def test_compile_before_build(self, model):
        """Test that compilation fails if model not built."""
        with pytest.raises(ValueError):
            model.compile()


# ============================================================================
# TRAINING TESTS
# ============================================================================

class TestModelTraining:
    """Test model training functionality."""
    
    def test_train_basic(self, built_model, sample_data):
        """Test basic training."""
        history = built_model.train(
            sample_data["train_features"],
            sample_data["train_labels"],
            sample_data["val_features"],
            sample_data["val_labels"],
            epochs=3,
            batch_size=16,
            verbose=0
        )
        
        assert history is not None
        assert "history" in history
        assert "metadata" in history
        assert len(history["history"].history["loss"]) > 0
    
    def test_train_stores_metadata(self, built_model, sample_data):
        """Test that training stores metadata correctly."""
        result = built_model.train(
            sample_data["train_features"],
            sample_data["train_labels"],
            sample_data["val_features"],
            sample_data["val_labels"],
            epochs=2,
            batch_size=16,
            learning_rate=0.001,
            verbose=0
        )
        
        metadata = result["metadata"]
        assert "epochs_trained" in metadata
        assert "final_train_loss" in metadata
        assert "final_val_accuracy" in metadata
        assert metadata["learning_rate"] == 0.001
    
    def test_train_with_class_weights(self, built_model, sample_data):
        """Test training with class weights."""
        result = built_model.train(
            sample_data["train_features"],
            sample_data["train_labels"],
            sample_data["val_features"],
            sample_data["val_labels"],
            epochs=2,
            batch_size=16,
            class_weight_strategy="balanced",
            verbose=0
        )
        
        assert result is not None
        assert result["metadata"]["class_weight_strategy"] == "balanced"
    
    def test_train_wrong_feature_dimension(self, built_model, sample_data):
        """Test training with wrong feature dimensions."""
        wrong_features = np.random.randn(100, 30)
        
        with pytest.raises(ValueError):
            built_model.train(
                wrong_features,
                sample_data["train_labels"],
                sample_data["val_features"],
                sample_data["val_labels"],
                epochs=1,
                verbose=0
            )
    
    def test_train_mismatched_samples(self, built_model):
        """Test training with mismatched number of samples."""
        features = np.random.randn(100, 46)
        labels = to_categorical(np.random.randint(0, 5, 50), num_classes=5)
        
        with pytest.raises(ValueError):
            built_model.train(
                features, labels,
                features, labels,
                epochs=1,
                verbose=0
            )


# ============================================================================
# EVALUATION TESTS
# ============================================================================

class TestModelEvaluation:
    """Test model evaluation functionality."""
    
    def test_evaluate_basic(self, built_model, sample_data):
        """Test basic evaluation."""
        # Train first
        built_model.train(
            sample_data["train_features"],
            sample_data["train_labels"],
            sample_data["val_features"],
            sample_data["val_labels"],
            epochs=1,
            verbose=0
        )
        
        # Evaluate
        metrics = built_model.evaluate(
            sample_data["val_features"],
            sample_data["val_labels"],
            verbose=0
        )
        
        assert "loss" in metrics or metrics[0] is not None
        assert "accuracy" in metrics or len(metrics) > 1
    
    def test_evaluate_before_build(self, model):
        """Test that evaluation fails if model not built."""
        features = np.random.randn(10, 46)
        labels = to_categorical(np.random.randint(0, 5, 10), num_classes=5)
        
        with pytest.raises(ValueError):
            model.evaluate(features, labels)


# ============================================================================
# PREDICTION TESTS
# ============================================================================

class TestModelPrediction:
    """Test model prediction functionality."""
    
    def test_predict_basic(self, built_model, sample_data):
        """Test basic prediction."""
        predictions = built_model.predict(sample_data["val_features"])
        
        assert predictions.shape == (20, 5)
        assert np.allclose(predictions.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(predictions >= 0) and np.all(predictions <= 1)
    
    def test_predict_single_sample(self, built_model, sample_data):
        """Test prediction on single sample."""
        single_feature = sample_data["val_features"][[0], :]
        predictions = built_model.predict(single_feature)
        
        assert predictions.shape == (1, 5)
    
    def test_predict_wrong_dimension(self, built_model):
        """Test prediction with wrong feature dimensions."""
        features = np.random.randn(10, 30)  # Wrong dimension
        
        with pytest.raises(ValueError):
            built_model.predict(features)
    
    def test_predict_with_confidence(self, built_model, sample_data):
        """Test prediction with confidence scores."""
        results = built_model.predict_batch_with_confidence(
            sample_data["val_features"],
            confidence_threshold=0.3,
            return_top_k=3
        )
        
        assert len(results) == 20
        assert all("class_id" in r for r in results)
        assert all("confidence" in r for r in results)
        assert all("above_threshold" in r for r in results)
        assert all("top_k" in r for r in results)
        assert all(len(r["top_k"]) <= 3 for r in results)
    
    def test_predict_before_build(self, model):
        """Test that prediction fails if model not built."""
        features = np.random.randn(10, 46)
        
        with pytest.raises(ValueError):
            model.predict(features)


# ============================================================================
# PERSISTENCE TESTS
# ============================================================================

class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_save_model(self, built_model):
        """Test saving model to HDF5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.h5")
            built_model.save_model(filepath)
            
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
    
    def test_save_model_creates_metadata(self, built_model, sample_data):
        """Test that saving model creates metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.h5")
            
            # Train first to create metadata
            built_model.train(
                sample_data["train_features"],
                sample_data["train_labels"],
                sample_data["val_features"],
                sample_data["val_labels"],
                epochs=1,
                verbose=0
            )
            
            built_model.save_model(filepath, include_metadata=True)
            
            metadata_path = filepath.replace(".h5", "_metadata.json")
            assert os.path.exists(metadata_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            assert metadata["model_name"] == "test_model"
            assert metadata["num_gestures"] == 5
            assert metadata["input_features"] == 46
    
    def test_save_before_build(self, model):
        """Test that saving fails if model not built."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.h5")
            
            with pytest.raises(ValueError):
                model.save_model(filepath)
    
    def test_load_model(self, built_model, sample_data):
        """Test loading model from HDF5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.h5")
            
            # Save model
            built_model.save_model(filepath)
            
            # Load model
            loaded_model = GestureClassificationModel.load_model(filepath)
            
            assert loaded_model.model is not None
            assert loaded_model.input_features == 46
            assert loaded_model.num_gestures == 5
    
    def test_load_nonexistent_model(self):
        """Test loading nonexistent model."""
        with pytest.raises(FileNotFoundError):
            GestureClassificationModel.load_model("nonexistent_path.h5")
    
    def test_load_and_predict_consistency(self, built_model, sample_data):
        """Test that predictions are consistent after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.h5")
            
            # Get predictions before save
            predictions_before = built_model.predict(sample_data["val_features"][:5])
            
            # Save and load
            built_model.save_model(filepath)
            loaded_model = GestureClassificationModel.load_model(filepath)
            
            # Get predictions after load
            predictions_after = loaded_model.predict(sample_data["val_features"][:5])
            
            # Check consistency
            assert np.allclose(predictions_before, predictions_after, atol=1e-5)


# ============================================================================
# MODEL INFO TESTS
# ============================================================================

class TestModelInfo:
    """Test model information retrieval."""
    
    def test_get_info_before_build(self, model):
        """Test getting info before model is built."""
        info = model.get_model_info()
        
        assert info["status"] == "not_built"
        assert info["model_name"] == "test_model"
        assert info["num_gestures"] == 5
    
    def test_get_info_after_build(self, built_model):
        """Test getting info after model is built."""
        info = built_model.get_model_info()
        
        assert info["status"] == "built"
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0


# ============================================================================
# CLASS WEIGHT TESTS
# ============================================================================

class TestClassWeights:
    """Test class weight computation."""
    
    def test_compute_class_weights_balanced(self):
        """Test class weight computation for balanced dataset."""
        labels = to_categorical([0, 0, 1, 1, 2, 2], num_classes=3)
        weights = GestureClassificationModel._compute_class_weights(labels)
        
        # All classes should have equal weight in balanced dataset
        assert weights[0] == weights[1]
        assert weights[1] == weights[2]
    
    def test_compute_class_weights_imbalanced(self):
        """Test class weight computation for imbalanced dataset."""
        labels = to_categorical([0, 0, 0, 0, 1, 1], num_classes=2)
        weights = GestureClassificationModel._compute_class_weights(labels)
        
        # Class 1 (underrepresented) should have higher weight
        assert weights[1] > weights[0]


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_gesture_class(self):
        """Test that model rejects single gesture class."""
        with pytest.raises(ValueError):
            GestureClassificationModel(num_gestures=1)
    
    def test_two_gesture_classes(self):
        """Test model with binary classification."""
        model = GestureClassificationModel(num_gestures=2)
        model.build(verbose=False)
        
        assert model.model.output_shape[-1] == 2
    
    def test_many_gesture_classes(self):
        """Test model with many gesture classes."""
        model = GestureClassificationModel(num_gestures=20)
        model.build(verbose=False)
        
        assert model.model.output_shape[-1] == 20
    
    def test_small_feature_vector(self):
        """Test model with small feature vector."""
        model = GestureClassificationModel(
            num_gestures=5,
            input_features=5
        )
        model.build(verbose=False)
        
        assert model.model.input_shape[-1] == 5
    
    def test_large_feature_vector(self):
        """Test model with large feature vector."""
        model = GestureClassificationModel(
            num_gestures=5,
            input_features=200
        )
        model.build(verbose=False)
        
        assert model.model.input_shape[-1] == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
