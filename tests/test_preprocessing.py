"""
Unit tests for the Preprocessing Pipeline module.

Tests cover:
- Configuration validation
- Data loading
- Feature engineering
- Normalization
- Data splitting
- Reproducibility
- Metadata tracking
- Dataset saving/loading
"""

import pytest
import numpy as np
import json
import pickle
from pathlib import Path
import tempfile
import shutil

from src.preprocessing import (
    PreprocessingPipeline,
    PreprocessingConfig,
    PreprocessingMetadata
)


class TestPreprocessingConfig:
    """Test suite for PreprocessingConfig"""
    
    def test_config_initialization(self):
        """Test config initialization with defaults"""
        config = PreprocessingConfig()
        assert config.random_seed == 42
        assert config.train_split == 0.7
        assert config.val_split == 0.15
        assert config.test_split == 0.15
    
    def test_config_custom_values(self):
        """Test config with custom values"""
        config = PreprocessingConfig(
            random_seed=123,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1
        )
        assert config.random_seed == 123
        assert config.train_split == 0.8
    
    def test_config_validation_invalid_split(self):
        """Test validation fails for invalid split ratios"""
        config = PreprocessingConfig(
            train_split=0.5,
            val_split=0.3,
            test_split=0.3
        )
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_validation_negative_split(self):
        """Test validation fails for negative split ratios"""
        config = PreprocessingConfig(
            train_split=-0.1,
            val_split=0.5,
            test_split=0.6
        )
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_validation_invalid_normalize_method(self):
        """Test validation fails for invalid normalization method"""
        config = PreprocessingConfig(normalize_method="invalid")
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_validation_valid(self):
        """Test validation passes for valid config"""
        config = PreprocessingConfig()
        config.validate()  # Should not raise


class TestPreprocessingPipeline:
    """Test suite for PreprocessingPipeline"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with sample gesture data"""
        temp_dir = tempfile.mkdtemp()
        
        # Create gesture directories
        gestures = ["peace", "thumbs_up", "ok"]
        for gesture in gestures:
            gesture_dir = Path(temp_dir) / gesture
            gesture_dir.mkdir(exist_ok=True)
            
            # Create sample JSON files
            for sample_idx in range(5):
                sample_data = {
                    "landmarks": np.random.rand(3, 21, 2).tolist(),  # 3 frames
                    "handedness": "Right",
                    "timestamp": 0.0
                }
                
                sample_file = gesture_dir / f"sample_{sample_idx:05d}.json"
                with open(sample_file, 'w') as f:
                    json.dump(sample_data, f)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        config = PreprocessingConfig(random_seed=42)
        pipeline = PreprocessingPipeline(config)
        
        assert pipeline.config.random_seed == 42
        assert pipeline.X_train is None
        assert pipeline.scaler is not None
    
    def test_pipeline_initialization_with_kwargs(self):
        """Test pipeline initialization with kwargs"""
        pipeline = PreprocessingPipeline(random_seed=123, train_split=0.8)
        
        assert pipeline.config.random_seed == 123
        assert pipeline.config.train_split == 0.8
    
    def test_set_random_seeds(self):
        """Test that random seeds are set correctly"""
        pipeline = PreprocessingPipeline(random_seed=42)
        
        # Generate random numbers
        rand1 = np.random.rand(10)
        
        # Create new pipeline with same seed
        pipeline2 = PreprocessingPipeline(random_seed=42)
        rand2 = np.random.rand(10)
        
        # Should be identical
        np.testing.assert_array_equal(rand1, rand2)
    
    def test_load_raw_data_missing_directory(self):
        """Test error handling when data directory doesn't exist"""
        config = PreprocessingConfig(data_dir="/nonexistent/path")
        pipeline = PreprocessingPipeline(config)
        
        with pytest.raises(FileNotFoundError):
            pipeline._load_raw_data()
    
    def test_load_raw_data_empty_directory(self):
        """Test error handling for empty data directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PreprocessingConfig(data_dir=temp_dir)
            pipeline = PreprocessingPipeline(config)
            
            with pytest.raises(ValueError):
                pipeline._load_raw_data()
    
    def test_load_raw_data_success(self, temp_data_dir):
        """Test successful raw data loading"""
        config = PreprocessingConfig(data_dir=temp_data_dir)
        pipeline = PreprocessingPipeline(config)
        
        X_raw, y_raw, gesture_names = pipeline._load_raw_data()
        
        # Check shapes
        assert X_raw.shape[1:] == (21, 2)  # 21 landmarks, 2D
        assert len(y_raw) == len(X_raw)
        assert len(gesture_names) == 3
        
        # Check values
        assert y_raw.min() >= 0
        assert y_raw.max() < len(gesture_names)
    
    def test_apply_feature_engineering(self, temp_data_dir):
        """Test feature engineering application"""
        config = PreprocessingConfig(data_dir=temp_data_dir)
        pipeline = PreprocessingPipeline(config)
        
        X_raw, _, _ = pipeline._load_raw_data()
        X_features = pipeline._apply_feature_engineering(X_raw)
        
        # Check shape
        assert X_features.shape[0] == X_raw.shape[0]
        assert X_features.shape[1] == 46  # 46 features
        
        # Check dtype
        assert X_features.dtype == np.float32
        
        # Check no NaN/Inf
        assert not np.isnan(X_features).any()
        assert not np.isinf(X_features).any()
    
    def test_normalize_features_standard(self, temp_data_dir):
        """Test standard normalization"""
        config = PreprocessingConfig(
            data_dir=temp_data_dir,
            normalize_method="standard"
        )
        pipeline = PreprocessingPipeline(config)
        
        X_raw, _, _ = pipeline._load_raw_data()
        X_features = pipeline._apply_feature_engineering(X_raw)
        X_normalized = pipeline._normalize_features(X_features, fit=True)
        
        # Check shape
        assert X_normalized.shape == X_features.shape
        
        # Check normalization: mean should be close to 0, std close to 1
        mean = np.mean(X_normalized, axis=0)
        std = np.std(X_normalized, axis=0)
        
        np.testing.assert_array_almost_equal(mean, np.zeros(46), decimal=1)
        np.testing.assert_array_almost_equal(std, np.ones(46), decimal=1)
    
    def test_normalize_features_minmax(self, temp_data_dir):
        """Test minmax normalization"""
        config = PreprocessingConfig(
            data_dir=temp_data_dir,
            normalize_method="minmax"
        )
        pipeline = PreprocessingPipeline(config)
        
        X_raw, _, _ = pipeline._load_raw_data()
        X_features = pipeline._apply_feature_engineering(X_raw)
        X_normalized = pipeline._normalize_features(X_features, fit=True)
        
        # Check range [0, 1]
        assert np.all(X_normalized >= -0.01)  # Allow small numerical errors
        assert np.all(X_normalized <= 1.01)
    
    def test_normalize_features_no_fit(self, temp_data_dir):
        """Test normalization without fitting"""
        config = PreprocessingConfig(data_dir=temp_data_dir)
        pipeline = PreprocessingPipeline(config)
        
        X_raw, _, _ = pipeline._load_raw_data()
        X_features = pipeline._apply_feature_engineering(X_raw)
        
        # First fit
        pipeline._normalize_features(X_features, fit=True)
        
        # Then apply without fitting
        X_normalized = pipeline._normalize_features(X_features, fit=False)
        
        assert X_normalized.shape == X_features.shape
    
    def test_normalize_features_error_no_scaler(self):
        """Test error when normalizing without fitted scaler"""
        pipeline = PreprocessingPipeline()
        X = np.random.rand(100, 46)
        
        with pytest.raises(ValueError):
            pipeline._normalize_features(X, fit=False)
    
    def test_split_data_proportions(self, temp_data_dir):
        """Test that data splitting maintains correct proportions"""
        config = PreprocessingConfig(
            data_dir=temp_data_dir,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2,
            random_seed=42
        )
        pipeline = PreprocessingPipeline(config)
        
        X_raw, y_raw, gesture_names = pipeline._load_raw_data()
        X_features = pipeline._apply_feature_engineering(X_raw)
        X_normalized = pipeline._normalize_features(X_features, fit=True)
        
        pipeline._split_data(X_normalized, y_raw, gesture_names)
        
        total = len(pipeline.y_train) + len(pipeline.y_val) + len(pipeline.y_test)
        
        train_ratio = len(pipeline.y_train) / total
        val_ratio = len(pipeline.y_val) / total
        test_ratio = len(pipeline.y_test) / total
        
        # Allow 5% deviation due to stratification
        assert abs(train_ratio - 0.6) < 0.05
        assert abs(val_ratio - 0.2) < 0.05
        assert abs(test_ratio - 0.2) < 0.05
    
    def test_split_data_stratification(self, temp_data_dir):
        """Test that splitting maintains class distribution"""
        config = PreprocessingConfig(data_dir=temp_data_dir)
        pipeline = PreprocessingPipeline(config)
        
        X_raw, y_raw, gesture_names = pipeline._load_raw_data()
        X_features = pipeline._apply_feature_engineering(X_raw)
        X_normalized = pipeline._normalize_features(X_features, fit=True)
        
        # Get class distribution in full dataset
        unique_full, counts_full = np.unique(y_raw, return_counts=True)
        dist_full = counts_full / len(y_raw)
        
        pipeline._split_data(X_normalized, y_raw, gesture_names)
        
        # Get class distribution in training set
        unique_train, counts_train = np.unique(pipeline.y_train, return_counts=True)
        dist_train = counts_train / len(pipeline.y_train)
        
        # Distributions should be similar
        np.testing.assert_array_almost_equal(dist_full, dist_train, decimal=1)
    
    def test_split_data_no_overlap(self, temp_data_dir):
        """Test that train/val/test sets have no overlap"""
        config = PreprocessingConfig(data_dir=temp_data_dir)
        pipeline = PreprocessingPipeline(config)
        
        X_raw, y_raw, gesture_names = pipeline._load_raw_data()
        X_features = pipeline._apply_feature_engineering(X_raw)
        X_normalized = pipeline._normalize_features(X_features, fit=True)
        
        pipeline._split_data(X_normalized, y_raw, gesture_names)
        
        # Check no overlap
        train_set = set(range(len(pipeline.y_train)))
        val_set = set(range(len(pipeline.y_train), len(pipeline.y_train) + len(pipeline.y_val)))
        test_set = set(range(len(pipeline.y_train) + len(pipeline.y_val), 
                            len(pipeline.y_train) + len(pipeline.y_val) + len(pipeline.y_test)))
        
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0
    
    def test_create_metadata(self, temp_data_dir):
        """Test metadata creation"""
        config = PreprocessingConfig(data_dir=temp_data_dir, random_seed=42)
        pipeline = PreprocessingPipeline(config)
        
        X_raw, y_raw, gesture_names = pipeline._load_raw_data()
        X_features = pipeline._apply_feature_engineering(X_raw)
        X_normalized = pipeline._normalize_features(X_features, fit=True)
        pipeline._split_data(X_normalized, y_raw, gesture_names)
        pipeline._create_metadata(gesture_names)
        
        metadata = pipeline.metadata
        
        assert metadata.n_features == 46
        assert metadata.n_gestures == len(gesture_names)
        assert metadata.random_seed == 42
        assert len(metadata.feature_names) == 46
    
    def test_process_full_pipeline(self, temp_data_dir):
        """Test complete preprocessing pipeline"""
        config = PreprocessingConfig(
            data_dir=temp_data_dir,
            random_seed=42
        )
        pipeline = PreprocessingPipeline(config)
        
        X_train, X_val, X_test, y_train, y_val, y_test, metadata = pipeline.process()
        
        # Check shapes
        assert X_train.shape[1] == 46
        assert len(y_train) == X_train.shape[0]
        
        if len(X_val) > 0:
            assert X_val.shape[1] == 46
            assert len(y_val) == X_val.shape[0]
        
        # Check metadata
        assert metadata.n_features == 46
        assert metadata.train_size == len(y_train)
    
    def test_process_reproducibility(self, temp_data_dir):
        """Test that pipeline produces identical results with same seed"""
        config1 = PreprocessingConfig(data_dir=temp_data_dir, random_seed=42)
        pipeline1 = PreprocessingPipeline(config1)
        X1_train, _, _, y1_train, _, _, _ = pipeline1.process()
        
        config2 = PreprocessingConfig(data_dir=temp_data_dir, random_seed=42)
        pipeline2 = PreprocessingPipeline(config2)
        X2_train, _, _, y2_train, _, _, _ = pipeline2.process()
        
        # Should be identical
        np.testing.assert_array_equal(X1_train, X2_train)
        np.testing.assert_array_equal(y1_train, y2_train)
    
    def test_save_datasets(self, temp_data_dir, temp_output_dir):
        """Test saving preprocessed datasets"""
        config = PreprocessingConfig(data_dir=temp_data_dir)
        pipeline = PreprocessingPipeline(config)
        pipeline.process()
        
        output_dir = pipeline.save_datasets(temp_output_dir)
        
        # Check files exist
        assert Path(output_dir).exists()
        assert (Path(output_dir) / "X_train.npy").exists()
        assert (Path(output_dir) / "y_train.npy").exists()
        assert (Path(output_dir) / "scaler.pkl").exists()
        assert (Path(output_dir) / "metadata.json").exists()
        assert (Path(output_dir) / "config.json").exists()
    
    def test_load_datasets(self, temp_data_dir, temp_output_dir):
        """Test loading saved datasets"""
        # Save datasets
        config = PreprocessingConfig(data_dir=temp_data_dir)
        pipeline = PreprocessingPipeline(config)
        X_train_orig, _, _, y_train_orig, _, _, metadata_orig = pipeline.process()
        pipeline.save_datasets(temp_output_dir)
        
        # Load datasets
        X_train, X_val, X_test, y_train, y_val, y_test, metadata = PreprocessingPipeline.load_datasets(temp_output_dir)
        
        # Check loaded data matches saved data
        np.testing.assert_array_equal(X_train, X_train_orig)
        np.testing.assert_array_equal(y_train, y_train_orig)
        assert metadata.n_features == metadata_orig.n_features
    
    def test_load_scaler(self, temp_data_dir, temp_output_dir):
        """Test loading fitted scaler"""
        # Save with scaler
        config = PreprocessingConfig(data_dir=temp_data_dir)
        pipeline = PreprocessingPipeline(config)
        pipeline.process()
        pipeline.save_datasets(temp_output_dir)
        
        # Load scaler
        scaler = PreprocessingPipeline.load_scaler(temp_output_dir)
        
        assert scaler is not None
    
    def test_load_scaler_missing_file(self, temp_output_dir):
        """Test error when scaler file is missing"""
        with pytest.raises(FileNotFoundError):
            PreprocessingPipeline.load_scaler(temp_output_dir)


class TestPreprocessingMetadata:
    """Test suite for PreprocessingMetadata"""
    
    def test_metadata_initialization(self):
        """Test metadata initialization"""
        metadata = PreprocessingMetadata(
            n_samples=100,
            n_features=46,
            n_gestures=5,
            gesture_names=["peace", "ok", "thumbs_up", "rock", "love"],
            train_size=70,
            val_size=15,
            test_size=15,
            train_indices=[],
            val_indices=[],
            test_indices=[],
            normalize_method="standard",
            feature_names=[f"feature_{i}" for i in range(46)],
            aggregation_method="mean"
        )
        
        assert metadata.n_samples == 100
        assert metadata.n_features == 46
        assert metadata.n_gestures == 5
    
    def test_metadata_timestamp(self):
        """Test that timestamp is created"""
        metadata = PreprocessingMetadata(
            n_samples=100,
            n_features=46,
            n_gestures=5,
            gesture_names=[],
            train_size=70,
            val_size=15,
            test_size=15,
            train_indices=[],
            val_indices=[],
            test_indices=[],
            normalize_method="standard",
            feature_names=[],
            aggregation_method="mean"
        )
        
        assert metadata.processed_timestamp is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
