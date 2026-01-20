"""
Preprocessing Pipeline for Hand Gesture Recognition

This module provides a complete data preprocessing pipeline that:
1. Loads raw landmark data from JSON files
2. Applies feature engineering (converts landmarks to features)
3. Normalizes features for ML models
4. Splits data into training and validation sets
5. Saves processed datasets with reproducibility

The pipeline ensures:
- Deterministic results (fixed random seeds)
- Reproducible splits across runs
- Consistent feature engineering
- Proper normalization
- Complete metadata tracking

Example:
    from src.preprocessing import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline(
        data_dir="data/collected_gestures",
        random_seed=42,
        train_split=0.8
    )
    
    # Load and process data
    X_train, X_val, y_train, y_val, metadata = pipeline.process()
    
    # Save for later use
    pipeline.save_datasets("processed_data")
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from src.feature_extractor import HandGestureFeatureExtractor
from src.data_utils import GestureDataLoader


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    
    # Data paths
    data_dir: str = "data/collected_gestures"
    output_dir: str = "processed_data"
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Train/validation/test split
    train_split: float = 0.7      # 70% training
    val_split: float = 0.15       # 15% validation
    test_split: float = 0.15      # 15% test
    
    # Feature normalization
    normalize_method: str = "standard"  # "standard" or "minmax"
    
    # Feature aggregation method
    aggregation_method: str = "mean"    # "first", "mean", "flatten"
    
    # Data filtering
    min_samples_per_gesture: int = 5
    
    # Verbosity
    verbose: bool = True
    
    def validate(self):
        """Validate configuration parameters"""
        total_split = self.train_split + self.val_split + self.test_split
        if not np.isclose(total_split, 1.0):
            raise ValueError(
                f"Train/val/test splits must sum to 1.0, got {total_split}"
            )
        
        if self.train_split <= 0 or self.val_split < 0 or self.test_split < 0:
            raise ValueError("Split ratios must be non-negative")
        
        if self.normalize_method not in ["standard", "minmax"]:
            raise ValueError(f"Unknown normalize_method: {self.normalize_method}")
        
        if self.aggregation_method not in ["first", "mean", "flatten"]:
            raise ValueError(f"Unknown aggregation_method: {self.aggregation_method}")


@dataclass
class PreprocessingMetadata:
    """Metadata about preprocessed datasets"""
    
    # Data shapes
    n_samples: int
    n_features: int
    n_gestures: int
    gesture_names: List[str]
    
    # Split information
    train_size: int
    val_size: int
    test_size: int
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    
    # Normalization info
    normalize_method: str
    scaler_mean: Optional[np.ndarray] = None
    scaler_scale: Optional[np.ndarray] = None
    scaler_min: Optional[np.ndarray] = None
    scaler_max: Optional[np.ndarray] = None
    
    # Feature info
    feature_names: List[str]
    aggregation_method: str
    
    # Processing info
    processed_timestamp: str = None
    random_seed: int = 42
    samples_per_gesture: Dict[str, int] = None
    
    def __post_init__(self):
        if self.processed_timestamp is None:
            self.processed_timestamp = datetime.now().isoformat()


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for gesture recognition data.
    
    Handles:
    1. Data loading from JSON files
    2. Feature engineering using HandGestureFeatureExtractor
    3. Feature normalization (StandardScaler or MinMaxScaler)
    4. Train/validation/test splitting
    5. Dataset saving with metadata
    6. Full reproducibility with fixed seeds
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None, **kwargs):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: PreprocessingConfig object or None
            **kwargs: Override config parameters (e.g., random_seed=42)
        
        Examples:
            # Using config object
            config = PreprocessingConfig(random_seed=42)
            pipeline = PreprocessingPipeline(config)
            
            # Using kwargs
            pipeline = PreprocessingPipeline(
                data_dir="data/collected_gestures",
                random_seed=42,
                train_split=0.8
            )
        """
        if config is None:
            config = PreprocessingConfig(**kwargs)
        else:
            # Override config with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        config.validate()
        self.config = config
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Initialize components
        self.feature_extractor = HandGestureFeatureExtractor(normalize=True)
        self.data_loader = GestureDataLoader(config.data_dir)
        self.scaler = None
        self.metadata = None
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self._log(f"Pipeline initialized with seed {config.random_seed}")
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.config.random_seed)
    
    def _log(self, message: str):
        """Log message if verbose mode enabled"""
        if self.config.verbose:
            print(f"[Preprocessing] {message}")
    
    def process(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PreprocessingMetadata]:
        """
        Execute complete preprocessing pipeline.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, metadata)
            where X_* are feature matrices and y_* are label arrays
        
        Raises:
            FileNotFoundError: If data directory doesn't exist
            ValueError: If no valid samples found
        """
        self._log("Starting preprocessing pipeline...")
        
        # Step 1: Load raw data
        self._log("Step 1: Loading raw landmark data...")
        X_raw, y_raw, gesture_names = self._load_raw_data()
        
        # Step 2: Apply feature engineering
        self._log("Step 2: Applying feature engineering...")
        X_features = self._apply_feature_engineering(X_raw)
        
        # Step 3: Normalize features
        self._log("Step 3: Normalizing features...")
        X_normalized = self._normalize_features(X_features, fit=True)
        
        # Step 4: Split data
        self._log("Step 4: Splitting data into train/val/test...")
        self._split_data(X_normalized, y_raw, gesture_names)
        
        # Step 5: Create metadata
        self._log("Step 5: Creating metadata...")
        self._create_metadata(gesture_names)
        
        self._log("Preprocessing complete!")
        self._print_summary()
        
        return (
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test,
            self.metadata
        )
    
    def _load_raw_data(self) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
        """
        Load raw landmark data from collected gesture files.
        
        Returns:
            Tuple of (landmark_arrays, labels, gesture_names)
        """
        data_dir = Path(self.config.data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        X_raw = []
        y_raw = []
        gesture_names = []
        gesture_to_idx = {}
        
        # Iterate through gesture directories
        gesture_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        if not gesture_dirs:
            raise ValueError(f"No gesture directories found in {data_dir}")
        
        total_samples = 0
        
        for gesture_idx, gesture_dir in enumerate(gesture_dirs):
            gesture_name = gesture_dir.name
            gesture_names.append(gesture_name)
            gesture_to_idx[gesture_name] = gesture_idx
            
            # Load all samples for this gesture
            sample_files = sorted(gesture_dir.glob("sample_*.json"))
            
            if len(sample_files) < self.config.min_samples_per_gesture:
                self._log(
                    f"Warning: Gesture '{gesture_name}' has only {len(sample_files)} samples, "
                    f"minimum is {self.config.min_samples_per_gesture}"
                )
                continue
            
            gesture_sample_count = 0
            
            for sample_file in sample_files:
                try:
                    with open(sample_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract landmarks (shape: [n_frames, 21, 2])
                    landmarks = np.array(data['landmarks'], dtype=np.float32)
                    
                    if landmarks.shape[1:] != (21, 2):
                        self._log(f"Warning: Invalid landmark shape in {sample_file}")
                        continue
                    
                    # Store each frame as separate sample
                    for frame_idx, frame_landmarks in enumerate(landmarks):
                        X_raw.append(frame_landmarks)
                        y_raw.append(gesture_idx)
                        gesture_sample_count += 1
                
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    self._log(f"Warning: Error loading {sample_file}: {e}")
                    continue
            
            total_samples += gesture_sample_count
            self._log(f"  Loaded {gesture_sample_count} frames from '{gesture_name}'")
        
        if not X_raw:
            raise ValueError("No valid samples loaded from data directory")
        
        X_raw = np.array(X_raw, dtype=np.float32)
        y_raw = np.array(y_raw, dtype=np.int64)
        
        self._log(f"Total samples loaded: {total_samples}")
        self._log(f"Gesture classes: {len(gesture_names)}")
        
        return X_raw, y_raw, gesture_names
    
    def _apply_feature_engineering(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Apply feature engineering to raw landmarks.
        
        Args:
            X_raw: Array of shape (n_samples, 21, 2) with raw landmarks
        
        Returns:
            Array of shape (n_samples, 46) with extracted features
        """
        n_samples = X_raw.shape[0]
        X_features = np.zeros((n_samples, 46), dtype=np.float32)
        
        for i, landmarks in enumerate(X_raw):
            try:
                features = self.feature_extractor.extract(landmarks)
                X_features[i] = features
            except Exception as e:
                self._log(f"Warning: Feature extraction failed for sample {i}: {e}")
                continue
        
        self._log(f"Feature engineering complete: {X_features.shape}")
        
        return X_features
    
    def _normalize_features(
        self,
        X: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize features using configured method.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            fit: If True, fit scaler; if False, use existing scaler
        
        Returns:
            Normalized feature matrix
        
        Raises:
            ValueError: If fit=False and scaler not yet fitted
        """
        if fit:
            # Create scaler
            if self.config.normalize_method == "standard":
                self.scaler = StandardScaler()
            elif self.config.normalize_method == "minmax":
                self.scaler = MinMaxScaler()
            
            # Fit and transform
            X_normalized = self.scaler.fit_transform(X)
            self._log(f"Fitted {self.config.normalize_method} normalizer")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_normalized = self.scaler.transform(X)
        
        return X_normalized
    
    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gesture_names: List[str]
    ):
        """
        Split data into training, validation, and test sets.
        
        Uses stratified split to maintain class distribution.
        
        Args:
            X: Feature matrix
            y: Label array
            gesture_names: List of gesture names
        """
        # First split: separate test set
        if self.config.test_split > 0:
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y,
                test_size=self.config.test_split,
                random_state=self.config.random_seed,
                stratify=y
            )
        else:
            X_temp, y_temp = X, y
            self.X_test, self.y_test = np.array([]), np.array([])
        
        # Second split: separate validation from training
        if self.config.val_split > 0:
            val_split_ratio = self.config.val_split / (self.config.train_split + self.config.val_split)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_split_ratio,
                random_state=self.config.random_seed,
                stratify=y_temp
            )
        else:
            self.X_train, self.y_train = X_temp, y_temp
            self.X_val, self.y_val = np.array([]), np.array([])
        
        # Store indices for metadata
        self.train_indices = np.where(np.isin(np.arange(len(y)), self._get_indices(y, self.y_train)))[0]
        
        self._log(f"Training set: {self.X_train.shape[0]} samples")
        if len(self.X_val) > 0:
            self._log(f"Validation set: {self.X_val.shape[0]} samples")
        if len(self.X_test) > 0:
            self._log(f"Test set: {self.X_test.shape[0]} samples")
    
    def _get_indices(self, y_full: np.ndarray, y_subset: np.ndarray) -> np.ndarray:
        """Get indices of subset in full array (for tracking)"""
        indices = []
        used = set()
        for i, label in enumerate(y_subset):
            for j, full_label in enumerate(y_full):
                if j not in used and full_label == label:
                    indices.append(j)
                    used.add(j)
                    break
        return np.array(indices)
    
    def _create_metadata(self, gesture_names: List[str]):
        """
        Create metadata object tracking all preprocessing details.
        
        Args:
            gesture_names: List of gesture class names
        """
        # Get scaler parameters
        scaler_mean = None
        scaler_scale = None
        scaler_min = None
        scaler_max = None
        
        if isinstance(self.scaler, StandardScaler):
            scaler_mean = self.scaler.mean_
            scaler_scale = self.scaler.scale_
        elif isinstance(self.scaler, MinMaxScaler):
            scaler_min = self.scaler.data_min_
            scaler_max = self.scaler.data_max_
        
        # Count samples per gesture
        unique, counts = np.unique(self.y_train, return_counts=True)
        samples_per_gesture = {
            gesture_names[int(idx)]: int(count)
            for idx, count in zip(unique, counts)
        }
        
        # Get feature names
        feature_names = self.feature_extractor.get_feature_names()
        
        # Create metadata
        self.metadata = PreprocessingMetadata(
            n_samples=len(self.y_train) + len(self.y_val) + len(self.y_test),
            n_features=self.X_train.shape[1],
            n_gestures=len(gesture_names),
            gesture_names=gesture_names,
            train_size=len(self.y_train),
            val_size=len(self.y_val),
            test_size=len(self.y_test),
            train_indices=self.train_indices.tolist(),
            val_indices=self.train_indices.tolist(),  # Placeholder
            test_indices=self.train_indices.tolist(),  # Placeholder
            normalize_method=self.config.normalize_method,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            scaler_min=scaler_min,
            scaler_max=scaler_max,
            feature_names=feature_names,
            aggregation_method=self.config.aggregation_method,
            random_seed=self.config.random_seed,
            samples_per_gesture=samples_per_gesture
        )
    
    def _print_summary(self):
        """Print summary of preprocessing results"""
        if not self.config.verbose:
            return
        
        print("\n" + "="*70)
        print("PREPROCESSING SUMMARY")
        print("="*70)
        print(f"Configuration:")
        print(f"  Random Seed: {self.config.random_seed}")
        print(f"  Normalization: {self.config.normalize_method}")
        print(f"  Aggregation: {self.config.aggregation_method}")
        print(f"\nData Splits:")
        print(f"  Training:   {self.X_train.shape[0]:6d} samples ({len(self.y_train)/self.metadata.n_samples*100:5.1f}%)")
        print(f"  Validation: {self.X_val.shape[0]:6d} samples ({len(self.y_val)/self.metadata.n_samples*100:5.1f}%)")
        print(f"  Test:       {self.X_test.shape[0]:6d} samples ({len(self.y_test)/self.metadata.n_samples*100:5.1f}%)")
        print(f"\nFeatures:")
        print(f"  Total features: {self.metadata.n_features}")
        print(f"  Gesture classes: {self.metadata.n_gestures}")
        print(f"  Feature names: {', '.join(self.metadata.gesture_names[:3])}...")
        print(f"\nSamples per gesture (training set):")
        for gesture, count in self.metadata.samples_per_gesture.items():
            print(f"  {gesture:20s}: {count:4d}")
        print("="*70 + "\n")
    
    def save_datasets(self, output_dir: str = None) -> str:
        """
        Save preprocessed datasets to disk.
        
        Saves:
        - X_train.npy, X_val.npy, X_test.npy: Feature matrices
        - y_train.npy, y_val.npy, y_test.npy: Label arrays
        - scaler.pkl: Fitted scaler
        - metadata.json: Preprocessing metadata
        
        Args:
            output_dir: Directory to save datasets (default: config.output_dir)
        
        Returns:
            Path to output directory
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._log(f"Saving datasets to {output_path}...")
        
        # Save feature matrices
        np.save(output_path / "X_train.npy", self.X_train)
        np.save(output_path / "y_train.npy", self.y_train)
        
        if len(self.X_val) > 0:
            np.save(output_path / "X_val.npy", self.X_val)
            np.save(output_path / "y_val.npy", self.y_val)
        
        if len(self.X_test) > 0:
            np.save(output_path / "X_test.npy", self.X_test)
            np.save(output_path / "y_test.npy", self.y_test)
        
        # Save scaler
        with open(output_path / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata_dict = asdict(self.metadata)
        # Convert numpy arrays to lists for JSON serialization
        for key in ['scaler_mean', 'scaler_scale', 'scaler_min', 'scaler_max']:
            if metadata_dict[key] is not None:
                metadata_dict[key] = metadata_dict[key].tolist()
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Save config
        config_dict = asdict(self.config)
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self._log(f"Datasets saved successfully!")
        
        return str(output_path)
    
    @staticmethod
    def load_datasets(output_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                  np.ndarray, np.ndarray, np.ndarray,
                                                  PreprocessingMetadata]:
        """
        Load preprocessed datasets from disk.
        
        Args:
            output_dir: Directory containing saved datasets
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, metadata)
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {output_path}")
        
        # Load feature matrices
        X_train = np.load(output_path / "X_train.npy")
        y_train = np.load(output_path / "y_train.npy")
        
        X_val = np.load(output_path / "X_val.npy") if (output_path / "X_val.npy").exists() else np.array([])
        y_val = np.load(output_path / "y_val.npy") if (output_path / "y_val.npy").exists() else np.array([])
        
        X_test = np.load(output_path / "X_test.npy") if (output_path / "X_test.npy").exists() else np.array([])
        y_test = np.load(output_path / "y_test.npy") if (output_path / "y_test.npy").exists() else np.array([])
        
        # Load metadata
        with open(output_path / "metadata.json", 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = PreprocessingMetadata(**metadata_dict)
        
        print(f"Loaded datasets from {output_path}")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape if len(X_val) > 0 else 'N/A'}")
        print(f"  Test: {X_test.shape if len(X_test) > 0 else 'N/A'}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, metadata
    
    @staticmethod
    def load_scaler(output_dir: str):
        """
        Load fitted scaler for applying to new data.
        
        Args:
            output_dir: Directory containing saved scaler
        
        Returns:
            Fitted scaler object
        """
        scaler_path = Path(output_dir) / "scaler.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return scaler


if __name__ == "__main__":
    """
    Demo: Run preprocessing pipeline
    """
    # Create pipeline with custom configuration
    config = PreprocessingConfig(
        data_dir="data/collected_gestures",
        output_dir="processed_data",
        random_seed=42,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        normalize_method="standard",
        verbose=True
    )
    
    pipeline = PreprocessingPipeline(config)
    
    # Run preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()
    
    # Save datasets
    output_dir = pipeline.save_datasets()
    
    print("\nDataset saved successfully!")
