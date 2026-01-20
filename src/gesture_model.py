"""
Lightweight Neural Network for Gesture Classification

This module provides a production-ready neural network for classifying hand gestures
from engineered feature vectors. Optimized for real-time inference with:

- Lightweight architecture for fast inference
- Comprehensive training pipeline with validation
- Best-practice callbacks (early stopping, model checkpointing, learning rate scheduling)
- Model persistence in HDF5 format
- Detailed accuracy reporting and metrics

Architecture:
    Input: 46-dimensional feature vector (from FeatureExtractor)
    Hidden: Dense layers with dropout regularization
    Output: Softmax layer for gesture class probability distribution

Example:
    from src.gesture_model import GestureClassificationModel
    import numpy as np
    
    # Create model
    model = GestureClassificationModel(
        num_gestures=5,
        input_features=46,
        model_name="gesture_classifier"
    )
    
    # Build model
    model.build()
    
    # Load data
    train_features = np.load('datasets/train_features.npy')
    train_labels = np.load('datasets/train_labels.npy')
    val_features = np.load('datasets/val_features.npy')
    val_labels = np.load('datasets/val_labels.npy')
    
    # Train
    history = model.train(
        train_features, train_labels,
        val_features, val_labels,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    metrics = model.evaluate(val_features, val_labels)
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    
    # Save model
    model.save_model("models/gesture_classifier.h5")
    
    # Predict
    features = np.array([[...46 features...]])  # Single sample
    prediction = model.predict(features)
    gesture_class = np.argmax(prediction[0])
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from typing import Tuple, Dict, Optional, List
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


class GestureClassificationModel:
    """
    Lightweight neural network for gesture classification.
    
    Designed for real-time inference with optimizations including:
    - Shallow architecture for low latency
    - Dropout regularization to prevent overfitting
    - Batch normalization for stable training
    - Early stopping to find optimal convergence point
    - Learning rate scheduling for better convergence
    
    Attributes:
        input_features (int): Dimension of input feature vector (default 46)
        num_gestures (int): Number of gesture classes to classify
        model_name (str): Name identifier for the model
        model (keras.Model): The compiled Keras model (None until built)
        class_weights (dict): Class weights for imbalanced datasets
    """
    
    def __init__(
        self,
        num_gestures: int,
        input_features: int = 46,
        model_name: str = "gesture_classifier",
        architecture: str = "lightweight"
    ):
        """
        Initialize the gesture classification model.
        
        Args:
            num_gestures (int): Number of gesture classes to predict
            input_features (int): Dimension of input feature vector (default 46)
            model_name (str): Name identifier for this model
            architecture (str): Model architecture preset ('lightweight', 'balanced', 'powerful')
                - 'lightweight': Fast inference, ~2-5ms per sample
                - 'balanced': Trade-off between speed and accuracy
                - 'powerful': Higher accuracy, ~10-20ms per sample
        
        Raises:
            ValueError: If num_gestures < 2 or input_features < 1
        """
        if num_gestures < 2:
            raise ValueError(f"num_gestures must be >= 2, got {num_gestures}")
        if input_features < 1:
            raise ValueError(f"input_features must be >= 1, got {input_features}")
        
        self.num_gestures = num_gestures
        self.input_features = input_features
        self.model_name = model_name
        self.architecture = architecture
        self.model = None
        self.class_weights = None
        self.history = None
        self.training_metadata = {}
        
        logger.info(
            f"Initialized {model_name} for {num_gestures} gesture classes "
            f"with {input_features} input features using {architecture} architecture"
        )
    
    def build(self, verbose: bool = True) -> keras.Model:
        """
        Build the neural network model.
        
        Creates an optimized architecture based on the selected preset:
        
        - Lightweight: 2 hidden layers (64, 32 units), ~18k parameters
        - Balanced: 3 hidden layers (128, 64, 32 units), ~50k parameters
        - Powerful: 3 hidden layers (256, 128, 64 units), ~100k parameters
        
        All architectures include:
        - Batch normalization for stable training
        - Dropout regularization (0.3-0.5) to prevent overfitting
        - ReLU activation for non-linearity
        - Softmax output layer for probability distribution
        
        Args:
            verbose (bool): If True, print model architecture summary
        
        Returns:
            keras.Model: The built (but not compiled) model
        """
        # Define architecture presets
        architectures = {
            "lightweight": {
                "hidden_layers": [64, 32],
                "dropout_rate": 0.3,
                "description": "Fast inference, minimal parameters"
            },
            "balanced": {
                "hidden_layers": [128, 64, 32],
                "dropout_rate": 0.4,
                "description": "Balanced speed and accuracy"
            },
            "powerful": {
                "hidden_layers": [256, 128, 64],
                "dropout_rate": 0.5,
                "description": "High accuracy, higher latency"
            }
        }
        
        if self.architecture not in architectures:
            raise ValueError(
                f"Unknown architecture: {self.architecture}. "
                f"Choose from {list(architectures.keys())}"
            )
        
        config = architectures[self.architecture]
        hidden_layers = config["hidden_layers"]
        dropout_rate = config["dropout_rate"]
        
        logger.info(f"Building {self.architecture} model: {config['description']}")
        
        # Build model using Functional API for flexibility
        inputs = layers.Input(shape=(self.input_features,), name="input_features")
        x = inputs
        
        # Hidden layers with batch normalization and dropout
        for i, units in enumerate(hidden_layers):
            x = layers.Dense(
                units,
                activation=None,
                name=f"dense_{i+1}"
            )(x)
            x = layers.BatchNormalization(name=f"batch_norm_{i+1}")(x)
            x = layers.Activation("relu", name=f"relu_{i+1}")(x)
            x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)
        
        # Output layer: softmax for multi-class classification
        outputs = layers.Dense(
            self.num_gestures,
            activation="softmax",
            name="output_probabilities"
        )(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name=self.model_name)
        
        # Count parameters for inference optimization
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        logger.info(f"Model built successfully")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Model: {self.model_name} ({self.architecture})")
            print(f"{'='*70}")
            self.model.summary()
            print(f"{'='*70}\n")
        
        return self.model
    
    def compile(
        self,
        learning_rate: float = 0.001,
        optimizer_type: str = "adam",
        loss_function: str = "categorical_crossentropy",
        metrics: Optional[List[str]] = None
    ) -> None:
        """
        Compile the model with optimizer, loss function, and metrics.
        
        Uses best practices:
        - Adam optimizer with adjustable learning rate (adaptive learning)
        - Categorical crossentropy for multi-class classification
        - Comprehensive metrics for monitoring
        
        Args:
            learning_rate (float): Learning rate for optimizer (default 0.001)
            optimizer_type (str): Optimizer type ('adam', 'sgd', 'rmsprop')
            loss_function (str): Loss function for training
            metrics (list): List of metrics to track (default: ['accuracy', 'top_k_categorical_accuracy'])
        
        Raises:
            ValueError: If model is not built yet
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation. Call build() first.")
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = [
                "accuracy",
                keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")
            ]
        
        # Create optimizer
        if optimizer_type.lower() == "adam":
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type.lower() == "sgd":
            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics
        )
        
        logger.info(
            f"Model compiled with {optimizer_type} optimizer "
            f"(lr={learning_rate}) and {loss_function} loss"
        )
    
    def _create_callbacks(
        self,
        model_save_path: str,
        early_stopping_patience: int = 15,
        reduce_lr_patience: int = 5
    ) -> List[callbacks.Callback]:
        """
        Create training callbacks following best practices.
        
        Callbacks included:
        1. ModelCheckpoint: Save best model based on validation loss
        2. EarlyStopping: Stop training if validation loss stops improving
        3. ReduceLROnPlateau: Reduce learning rate if validation loss plateaus
        4. TensorBoard: Optional visualization (requires tensorboard)
        
        Args:
            model_save_path (str): Path where best model will be saved
            early_stopping_patience (int): Epochs to wait before stopping
            reduce_lr_patience (int): Epochs to wait before reducing LR
        
        Returns:
            list: List of configured callback instances
        """
        callback_list = []
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        
        # 1. Model Checkpoint: Save best model
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
            save_freq="epoch"
        )
        callback_list.append(checkpoint_callback)
        logger.info(f"ModelCheckpoint: Saving best model to {model_save_path}")
        
        # 2. Early Stopping: Prevent overfitting
        early_stopping_callback = callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            verbose=1,
            restore_best_weights=True
        )
        callback_list.append(early_stopping_callback)
        logger.info(f"EarlyStopping: Stop if val_loss doesn't improve for {early_stopping_patience} epochs")
        
        # 3. Reduce Learning Rate on Plateau
        reduce_lr_callback = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
            mode="min"
        )
        callback_list.append(reduce_lr_callback)
        logger.info(f"ReduceLROnPlateau: Reduce LR by 50% if val_loss plateaus for {reduce_lr_patience} epochs")
        
        return callback_list
    
    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        class_weight_strategy: Optional[str] = "balanced",
        model_save_path: str = "models/gesture_model.h5",
        verbose: int = 1
    ) -> Dict:
        """
        Train the gesture classification model.
        
        Includes comprehensive logging and monitoring:
        - Training progress with loss and accuracy
        - Validation metrics at each epoch
        - Early stopping to prevent overfitting
        - Learning rate scheduling for better convergence
        - Best model persistence
        
        Args:
            train_features (np.ndarray): Training feature vectors, shape (N, input_features)
            train_labels (np.ndarray): Training labels (one-hot encoded), shape (N, num_gestures)
            val_features (np.ndarray): Validation feature vectors
            val_labels (np.ndarray): Validation labels (one-hot encoded)
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Initial learning rate for optimizer
            class_weight_strategy (str or None): How to handle class imbalance
                - 'balanced': Automatically compute class weights
                - 'auto': TensorFlow computes from labels
                - None: No class weighting
            model_save_path (str): Path to save the best model
            verbose (int): Verbosity level (0=silent, 1=progress bar, 2=line per epoch)
        
        Returns:
            dict: Training history and metadata
                - history: keras.callbacks.History object
                - metadata: Training configuration and results
        
        Raises:
            ValueError: If features or labels have wrong shape
        """
        # Validate input shapes
        if train_features.shape[0] != train_labels.shape[0]:
            raise ValueError(
                f"train_features and train_labels have different number of samples: "
                f"{train_features.shape[0]} vs {train_labels.shape[0]}"
            )
        if train_features.shape[1] != self.input_features:
            raise ValueError(
                f"train_features has wrong feature dimension: "
                f"expected {self.input_features}, got {train_features.shape[1]}"
            )
        
        # Ensure model is compiled
        if self.model is None:
            raise ValueError("Model must be built and compiled before training. Call build() and compile() first.")
        
        logger.info(f"Starting training with {train_features.shape[0]} samples")
        logger.info(f"Validation set has {val_features.shape[0]} samples")
        
        # Compute class weights if requested
        class_weights = None
        if class_weight_strategy == "balanced":
            class_weights = self._compute_class_weights(train_labels)
            logger.info(f"Using balanced class weights: {class_weights}")
        elif class_weight_strategy == "auto":
            # TensorFlow's automatic class weight computation
            unique_labels = np.argmax(train_labels, axis=1)
            class_counts = np.bincount(unique_labels)
            class_weights = {i: len(unique_labels) / (len(class_counts) * count) 
                           for i, count in enumerate(class_counts)}
            logger.info(f"Auto-computed class weights: {class_weights}")
        
        # Recompile with updated learning rate
        self.compile(learning_rate=learning_rate)
        
        # Create callbacks
        callbacks_list = self._create_callbacks(model_save_path)
        
        # Train model
        logger.info(f"Training starts: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
        history = self.model.fit(
            train_features,
            train_labels,
            validation_data=(val_features, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=verbose
        )
        
        # Store training metadata
        self.history = history
        self.training_metadata = {
            "epochs_trained": len(history.history["loss"]),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "class_weight_strategy": class_weight_strategy,
            "model_save_path": model_save_path
        }
        
        logger.info(f"Training completed: {self.training_metadata['epochs_trained']} epochs")
        logger.info(
            f"Final validation accuracy: {self.training_metadata['final_val_accuracy']:.4f}"
        )
        
        return {
            "history": history,
            "metadata": self.training_metadata
        }
    
    def evaluate(
        self,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict:
        """
        Evaluate model on test or validation data.
        
        Args:
            test_features (np.ndarray): Test feature vectors
            test_labels (np.ndarray): Test labels (one-hot encoded)
            batch_size (int): Batch size for evaluation
            verbose (int): Verbosity level
        
        Returns:
            dict: Dictionary with evaluation metrics
                - loss: Model loss on test data
                - accuracy: Classification accuracy
                - top_2_accuracy: Top-2 accuracy (if available)
        """
        if self.model is None:
            raise ValueError("Model must be built before evaluation.")
        
        logger.info(f"Evaluating on {test_features.shape[0]} samples")
        
        eval_results = self.model.evaluate(
            test_features,
            test_labels,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Get metric names from compiled model
        metric_names = [self.model.metrics_names[i] if i < len(self.model.metrics_names) 
                       else f"metric_{i}" for i in range(len(eval_results))]
        
        metrics_dict = dict(zip(metric_names, eval_results))
        
        logger.info(f"Evaluation results: {metrics_dict}")
        
        return metrics_dict
    
    def predict(
        self,
        features: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict gesture class probabilities for input features.
        
        Args:
            features (np.ndarray): Input feature vectors, shape (N, input_features)
            batch_size (int): Batch size for prediction
        
        Returns:
            np.ndarray: Predicted class probabilities, shape (N, num_gestures)
                Each row is a probability distribution over gesture classes
        
        Example:
            # Single sample
            features = np.array([[...46 features...]])
            predictions = model.predict(features)
            gesture_class = np.argmax(predictions[0])
            confidence = predictions[0][gesture_class]
        """
        if self.model is None:
            raise ValueError("Model must be built before prediction.")
        
        if features.shape[1] != self.input_features:
            raise ValueError(
                f"features has wrong dimension: "
                f"expected {self.input_features}, got {features.shape[1]}"
            )
        
        predictions = self.model.predict(features, batch_size=batch_size, verbose=0)
        return predictions
    
    def predict_batch_with_confidence(
        self,
        features: np.ndarray,
        confidence_threshold: float = 0.5,
        return_top_k: int = 1
    ) -> List[Dict]:
        """
        Predict gestures with confidence scores and filtering.
        
        Args:
            features (np.ndarray): Input feature vectors
            confidence_threshold (float): Minimum confidence to report prediction
            return_top_k (int): Return top-k predictions per sample
        
        Returns:
            list: List of prediction dictionaries with keys:
                - 'class_id': Predicted gesture class index
                - 'class_name': Name of gesture (if available)
                - 'confidence': Prediction confidence
                - 'top_k': Top-k predictions with confidence scores
        """
        predictions = self.predict(features)
        results = []
        
        for pred in predictions:
            top_indices = np.argsort(pred)[::-1][:return_top_k]
            top_k = [
                {"class_id": int(idx), "confidence": float(pred[idx])}
                for idx in top_indices
            ]
            
            best_class = int(top_indices[0])
            best_confidence = float(pred[best_class])
            
            result = {
                "class_id": best_class,
                "confidence": best_confidence,
                "above_threshold": best_confidence >= confidence_threshold,
                "top_k": top_k
            }
            results.append(result)
        
        return results
    
    def save_model(self, filepath: str, include_metadata: bool = True) -> None:
        """
        Save trained model to HDF5 file.
        
        Args:
            filepath (str): Path to save model (should end with .h5)
            include_metadata (bool): If True, save metadata alongside model
        
        Raises:
            ValueError: If model is not built
        """
        if self.model is None:
            raise ValueError("Model must be built before saving.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        # Save model
        self.model.save(filepath, save_format='h5')
        logger.info(f"Model saved to {filepath}")
        
        # Save metadata as JSON
        if include_metadata:
            metadata_path = filepath.replace(".h5", "_metadata.json")
            metadata = {
                "model_name": self.model_name,
                "architecture": self.architecture,
                "input_features": self.input_features,
                "num_gestures": self.num_gestures,
                "training_metadata": self.training_metadata
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
    
    @staticmethod
    def load_model(filepath: str) -> 'GestureClassificationModel':
        """
        Load trained model from HDF5 file.
        
        Args:
            filepath (str): Path to model file (.h5)
        
        Returns:
            GestureClassificationModel: Loaded model instance
        
        Example:
            model = GestureClassificationModel.load_model("models/gesture_model.h5")
            predictions = model.predict(features)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        logger.info(f"Loading model from {filepath}")
        
        # Load metadata if available
        metadata_path = filepath.replace(".h5", "_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
        
        # Load model
        keras_model = keras.models.load_model(filepath)
        
        # Create instance
        if metadata:
            instance = GestureClassificationModel(
                num_gestures=metadata.get("num_gestures", 5),
                input_features=metadata.get("input_features", 46),
                model_name=metadata.get("model_name", "gesture_classifier"),
                architecture=metadata.get("architecture", "lightweight")
            )
            instance.training_metadata = metadata.get("training_metadata", {})
        else:
            # Infer from loaded model
            num_gestures = keras_model.output_shape[-1]
            input_features = keras_model.input_shape[-1]
            instance = GestureClassificationModel(
                num_gestures=num_gestures,
                input_features=input_features
            )
        
        instance.model = keras_model
        logger.info(f"Model loaded successfully")
        
        return instance
    
    @staticmethod
    def _compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights to handle imbalanced datasets.
        
        Args:
            labels (np.ndarray): One-hot encoded labels, shape (N, num_classes)
        
        Returns:
            dict: Mapping from class index to weight
        """
        # Convert one-hot to class indices
        class_indices = np.argmax(labels, axis=1)
        
        # Count samples per class
        unique, counts = np.unique(class_indices, return_counts=True)
        total_samples = len(class_indices)
        
        # Compute weights: w_i = total_samples / (num_classes * count_i)
        weights = {}
        for class_idx, count in zip(unique, counts):
            weights[int(class_idx)] = total_samples / (len(unique) * count)
        
        return weights
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information.
        
        Returns:
            dict: Model configuration and statistics
        """
        if self.model is None:
            return {
                "status": "not_built",
                "model_name": self.model_name,
                "architecture": self.architecture,
                "input_features": self.input_features,
                "num_gestures": self.num_gestures
            }
        
        return {
            "status": "built",
            "model_name": self.model_name,
            "architecture": self.architecture,
            "input_features": self.input_features,
            "num_gestures": self.num_gestures,
            "total_parameters": self.model.count_params(),
            "trainable_parameters": sum([tf.keras.backend.count_params(w) 
                                        for w in self.model.trainable_weights]),
            "training_metadata": self.training_metadata
        }
