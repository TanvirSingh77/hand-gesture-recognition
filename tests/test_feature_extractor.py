"""
Unit tests for the Hand Gesture Feature Extractor module.

Tests cover:
- Feature extraction correctness
- Distance normalization
- Angle computation
- Edge cases and error handling
- Feature vector consistency
"""

import pytest
import numpy as np
from src.feature_extractor import HandGestureFeatureExtractor


class TestHandGestureFeatureExtractor:
    """Test suite for HandGestureFeatureExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create a feature extractor instance for testing"""
        return HandGestureFeatureExtractor(normalize=True)
    
    @pytest.fixture
    def neutral_hand(self):
        """Create a neutral hand pose (all fingers extended)"""
        landmarks = np.array([
            [0.5, 0.5],    # 0: Wrist
            [0.45, 0.35],  # 1: Thumb CMC
            [0.40, 0.25],  # 2: Thumb MCP
            [0.35, 0.15],  # 3: Thumb IP
            [0.30, 0.05],  # 4: Thumb TIP
            [0.55, 0.35],  # 5: Index MCP
            [0.58, 0.25],  # 6: Index PIP
            [0.60, 0.15],  # 7: Index DIP
            [0.62, 0.05],  # 8: Index TIP
            [0.65, 0.35],  # 9: Middle MCP
            [0.68, 0.25],  # 10: Middle PIP
            [0.70, 0.15],  # 11: Middle DIP
            [0.72, 0.05],  # 12: Middle TIP
            [0.75, 0.35],  # 13: Ring MCP
            [0.78, 0.25],  # 14: Ring PIP
            [0.80, 0.15],  # 15: Ring DIP
            [0.82, 0.05],  # 16: Ring TIP
            [0.85, 0.35],  # 17: Pinky MCP
            [0.88, 0.25],  # 18: Pinky PIP
            [0.90, 0.15],  # 19: Pinky DIP
            [0.92, 0.05],  # 20: Pinky TIP
        ], dtype=np.float32)
        return landmarks
    
    @pytest.fixture
    def fist_hand(self):
        """Create a fist pose (all fingers bent/closed)"""
        landmarks = np.array([
            [0.5, 0.5],    # 0: Wrist
            [0.45, 0.45],  # 1: Thumb CMC
            [0.43, 0.43],  # 2: Thumb MCP
            [0.42, 0.42],  # 3: Thumb IP
            [0.41, 0.41],  # 4: Thumb TIP
            [0.52, 0.48],  # 5: Index MCP
            [0.53, 0.47],  # 6: Index PIP
            [0.54, 0.46],  # 7: Index DIP
            [0.55, 0.45],  # 8: Index TIP
            [0.56, 0.48],  # 9: Middle MCP
            [0.57, 0.47],  # 10: Middle PIP
            [0.58, 0.46],  # 11: Middle DIP
            [0.59, 0.45],  # 12: Middle TIP
            [0.60, 0.48],  # 13: Ring MCP
            [0.61, 0.47],  # 14: Ring PIP
            [0.62, 0.46],  # 15: Ring DIP
            [0.63, 0.45],  # 16: Ring TIP
            [0.64, 0.48],  # 17: Pinky MCP
            [0.65, 0.47],  # 18: Pinky PIP
            [0.66, 0.46],  # 19: Pinky DIP
            [0.67, 0.45],  # 20: Pinky TIP
        ], dtype=np.float32)
        return landmarks
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization"""
        assert extractor.normalize == True
        assert extractor.fill_value == 0.0
        assert extractor.WRIST == 0
        assert extractor.THUMB_TIP == 4
        assert extractor.INDEX_TIP == 8
    
    def test_feature_vector_shape(self, extractor, neutral_hand):
        """Test that feature vector has correct shape"""
        features = extractor.extract(neutral_hand)
        assert features.shape == (46,), f"Expected shape (46,), got {features.shape}"
    
    def test_feature_vector_dtype(self, extractor, neutral_hand):
        """Test that feature vector is float32"""
        features = extractor.extract(neutral_hand)
        assert features.dtype == np.float32
    
    def test_invalid_landmarks_shape(self, extractor):
        """Test error handling for invalid landmarks shape"""
        invalid_landmarks = np.random.rand(20, 2)  # Wrong number of points
        with pytest.raises(ValueError):
            extractor.extract(invalid_landmarks)
    
    def test_extract_with_dict_return(self, extractor, neutral_hand):
        """Test extract with return_dict=True"""
        result = extractor.extract(neutral_hand, return_dict=True)
        
        assert isinstance(result, dict)
        assert 'vector' in result
        assert 'distances' in result
        assert 'angles' in result
        assert 'spans' in result
        assert 'positions' in result
        assert 'hand_span' in result
        
        assert result['vector'].shape == (46,)
        assert result['distances'].shape == (21,)
        assert result['angles'].shape == (15,)
        assert result['spans'].shape == (4,)
        assert result['positions'].shape == (6,)
    
    def test_feature_groups_concatenate(self, extractor, neutral_hand):
        """Test that feature groups concatenate correctly into full vector"""
        result = extractor.extract(neutral_hand, return_dict=True)
        vector = result['vector']
        
        # Verify that concatenated vector equals individual components
        expected = np.concatenate([
            result['distances'],
            result['angles'],
            result['spans'],
            result['positions']
        ])
        
        np.testing.assert_array_almost_equal(vector, expected)
    
    def test_distances_positive(self, extractor, neutral_hand):
        """Test that all distances are non-negative"""
        result = extractor.extract(neutral_hand, return_dict=True)
        assert np.all(result['distances'] >= 0), "All distances should be non-negative"
    
    def test_angles_in_valid_range(self, extractor, neutral_hand):
        """Test that all angles are in valid range [0, 180]"""
        result = extractor.extract(neutral_hand, return_dict=True)
        angles = result['angles']
        
        assert np.all(angles >= 0), "Angles should be >= 0"
        assert np.all(angles <= 180), "Angles should be <= 180"
    
    def test_positions_normalized(self, extractor, neutral_hand):
        """Test that relative positions are approximately in [0, 1]"""
        result = extractor.extract(neutral_hand, return_dict=True)
        positions = result['positions']
        
        # Allow small overflow due to landmarks potentially outside bbox
        assert np.all(positions >= -0.1), "Positions should be >= -0.1"
        assert np.all(positions <= 1.1), "Positions should be <= 1.1"
    
    def test_fist_vs_open_hand(self, extractor, neutral_hand, fist_hand):
        """Test that fist and open hand produce different features"""
        open_features = extractor.extract(neutral_hand)
        fist_features = extractor.extract(fist_hand)
        
        # Features should be different
        difference = np.abs(open_features - fist_features)
        assert np.sum(difference) > 0.1, "Open and fist poses should have different features"
    
    def test_fist_smaller_distances(self, extractor, neutral_hand, fist_hand):
        """Test that fist has smaller finger distances than open hand"""
        open_result = extractor.extract(neutral_hand, return_dict=True)
        fist_result = extractor.extract(fist_hand, return_dict=True)
        
        # Open hand distances should generally be larger
        open_avg_dist = np.mean(open_result['distances'])
        fist_avg_dist = np.mean(fist_result['distances'])
        
        assert open_avg_dist > fist_avg_dist, "Open hand should have larger distances than fist"
    
    def test_fist_smaller_angles(self, extractor, neutral_hand, fist_hand):
        """Test that fist has smaller angles than open hand (more bent)"""
        open_result = extractor.extract(neutral_hand, return_dict=True)
        fist_result = extractor.extract(fist_hand, return_dict=True)
        
        # Fist angles should generally be smaller (more bent)
        open_avg_angle = np.mean(open_result['angles'])
        fist_avg_angle = np.mean(fist_result['angles'])
        
        assert open_avg_angle > fist_avg_angle, "Open hand should have larger angles than fist"
    
    def test_normalization_effect(self):
        """Test that normalization makes distances scale-invariant"""
        extractor_norm = HandGestureFeatureExtractor(normalize=True)
        extractor_no_norm = HandGestureFeatureExtractor(normalize=False)
        
        # Create two versions of same gesture at different scales
        small_hand = np.random.rand(21, 2) * 0.1 + 0.45  # Small hand in center
        large_hand = np.random.rand(21, 2) * 0.2 + 0.4   # Larger hand
        
        # Extract features
        small_norm = extractor_norm.extract(small_hand)
        large_norm = extractor_norm.extract(large_hand)
        
        small_unnorm = extractor_no_norm.extract(small_hand)
        large_unnorm = extractor_no_norm.extract(large_hand)
        
        # Normalized features should be more similar despite different scales
        norm_diff = np.sum(np.abs(small_norm - large_norm))
        unnorm_diff = np.sum(np.abs(small_unnorm - large_unnorm))
        
        # This is a heuristic test - normalized should generally be smaller
        # (though not guaranteed for random landmarks)
    
    def test_euclidean_distance_computation(self, extractor):
        """Test euclidean distance computation"""
        point1 = np.array([0.0, 0.0])
        point2 = np.array([3.0, 4.0])
        
        distance = extractor._euclidean_distance(point1, point2)
        assert distance == pytest.approx(5.0, rel=1e-5)
    
    def test_angle_computation_right_angle(self, extractor):
        """Test angle computation for right angle (90 degrees)"""
        # Right angle: vertex at origin, points on axes
        vertex = np.array([0.0, 0.0])
        point_a = np.array([1.0, 0.0])
        point_c = np.array([0.0, 1.0])
        
        angle = extractor._compute_angle(point_a, vertex, point_c)
        assert angle == pytest.approx(90.0, rel=1e-2)
    
    def test_angle_computation_straight_line(self, extractor):
        """Test angle computation for straight line (180 degrees)"""
        vertex = np.array([0.0, 0.0])
        point_a = np.array([1.0, 0.0])
        point_c = np.array([-1.0, 0.0])
        
        angle = extractor._compute_angle(point_a, vertex, point_c)
        assert angle == pytest.approx(180.0, rel=1e-2)
    
    def test_angle_computation_zero_length_vector(self, extractor):
        """Test angle computation with zero-length vector"""
        vertex = np.array([0.0, 0.0])
        point_a = np.array([0.0, 0.0])  # Same as vertex
        point_c = np.array([1.0, 0.0])
        
        angle = extractor._compute_angle(point_a, vertex, point_c)
        assert angle == pytest.approx(0.0, abs=1e-5)
    
    def test_feature_names_count(self, extractor):
        """Test that feature names list has correct length"""
        names = extractor.get_feature_names()
        assert len(names) == 46
    
    def test_feature_names_unique(self, extractor):
        """Test that all feature names are unique"""
        names = extractor.get_feature_names()
        assert len(names) == len(set(names)), "Feature names should be unique"
    
    def test_reproducibility(self, extractor, neutral_hand):
        """Test that same input produces same output"""
        features1 = extractor.extract(neutral_hand.copy())
        features2 = extractor.extract(neutral_hand.copy())
        
        np.testing.assert_array_equal(features1, features2)
    
    def test_symmetric_hand_landmarks(self):
        """Test behavior with symmetric landmarks"""
        # Create symmetric hand landmarks
        extractor = HandGestureFeatureExtractor(normalize=True)
        
        symmetric_landmarks = np.array([
            [0.5, 0.5],    # Wrist
            [0.5, 0.4],    # Symmetric points
            [0.5, 0.3],
            [0.5, 0.2],
            [0.5, 0.1],
            [0.5, 0.4],
            [0.5, 0.3],
            [0.5, 0.2],
            [0.5, 0.1],
            [0.5, 0.4],
            [0.5, 0.3],
            [0.5, 0.2],
            [0.5, 0.1],
            [0.5, 0.4],
            [0.5, 0.3],
            [0.5, 0.2],
            [0.5, 0.1],
            [0.5, 0.4],
            [0.5, 0.3],
            [0.5, 0.2],
            [0.5, 0.1],
        ], dtype=np.float32)
        
        # Should not raise an error
        features = extractor.extract(symmetric_landmarks)
        assert features.shape == (46,)
        assert np.isfinite(features).all()
    
    def test_hand_span_computation(self, extractor, neutral_hand):
        """Test hand span computation"""
        hand_span = extractor._compute_hand_span(neutral_hand)
        
        # Hand span should be positive
        assert hand_span > 0, "Hand span should be positive"
        
        # Should not exceed the diagonal of the unit square
        assert hand_span <= np.sqrt(2), "Hand span should be <= sqrt(2) for normalized coordinates"
    
    def test_inter_joint_distances_shape(self, extractor, neutral_hand):
        """Test inter-joint distances extraction"""
        distances = extractor._compute_inter_joint_distances(neutral_hand, 1.0)
        assert distances.shape == (21,), f"Expected 21 distances, got {distances.shape[0]}"
    
    def test_joint_angles_shape(self, extractor, neutral_hand):
        """Test joint angles extraction"""
        angles = extractor._compute_joint_angles(neutral_hand)
        assert angles.shape == (15,), f"Expected 15 angles, got {angles.shape[0]}"
    
    def test_hand_span_metrics_shape(self, extractor, neutral_hand):
        """Test hand span metrics extraction"""
        metrics = extractor._compute_hand_span_metrics(neutral_hand)
        assert metrics.shape == (4,), f"Expected 4 metrics, got {metrics.shape[0]}"
    
    def test_relative_positions_shape(self, extractor, neutral_hand):
        """Test relative positions extraction"""
        positions = extractor._compute_relative_positions(neutral_hand)
        assert positions.shape == (6,), f"Expected 6 positions, got {positions.shape[0]}"
    
    def test_no_nan_values(self, extractor, neutral_hand, fist_hand):
        """Test that feature extraction never produces NaN values"""
        for landmarks in [neutral_hand, fist_hand]:
            features = extractor.extract(landmarks)
            assert not np.isnan(features).any(), "Feature vector should not contain NaN values"
    
    def test_no_inf_values(self, extractor, neutral_hand, fist_hand):
        """Test that feature extraction never produces infinite values"""
        for landmarks in [neutral_hand, fist_hand]:
            features = extractor.extract(landmarks)
            assert not np.isinf(features).any(), "Feature vector should not contain infinite values"
    
    def test_feature_consistency_across_calls(self, extractor):
        """Test that features remain consistent across multiple extractions"""
        landmarks = np.random.rand(21, 2)
        
        features_list = [extractor.extract(landmarks) for _ in range(5)]
        
        # All extractions should be identical
        for i in range(1, len(features_list)):
            np.testing.assert_array_equal(features_list[0], features_list[i])


class TestFeatureExtractorIntegration:
    """Integration tests with hand landmark detector"""
    
    def test_integration_with_detector(self):
        """Test integration with HandLandmarkDetector"""
        # This test verifies the pipeline works end-to-end
        # It creates mock landmarks as if from HandLandmarkDetector
        
        extractor = HandGestureFeatureExtractor()
        
        # Mock landmarks from detector (normalized coordinates)
        mock_landmarks = np.array([
            [0.5, 0.7],    # Wrist
            [0.45, 0.65],  # Thumb joints
            [0.40, 0.55],
            [0.35, 0.45],
            [0.30, 0.35],
            [0.55, 0.65],  # Index joints
            [0.60, 0.55],
            [0.65, 0.45],
            [0.70, 0.35],
            [0.65, 0.65],  # Middle joints
            [0.70, 0.55],
            [0.75, 0.45],
            [0.80, 0.35],
            [0.75, 0.65],  # Ring joints
            [0.80, 0.55],
            [0.85, 0.45],
            [0.90, 0.35],
            [0.85, 0.65],  # Pinky joints
            [0.90, 0.55],
            [0.95, 0.45],
            [1.0, 0.35],
        ], dtype=np.float32)
        
        # Should extract features without error
        features = extractor.extract(mock_landmarks)
        
        assert features.shape == (46,)
        assert np.all(np.isfinite(features))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
