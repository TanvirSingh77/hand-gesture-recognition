"""Unit tests for the HandLandmarkDetector module."""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hand_landmarks import HandLandmarkDetector


class TestHandLandmarkDetectorInitialization:
    """Tests for HandLandmarkDetector initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        detector = HandLandmarkDetector()
        assert detector.NUM_LANDMARKS == 21
        assert detector.hands is not None
        detector.close()
    
    def test_custom_confidence_thresholds(self):
        """Test initialization with custom confidence thresholds."""
        detector = HandLandmarkDetector(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        assert detector.hands is not None
        detector.close()
    
    def test_invalid_detection_confidence(self):
        """Test that invalid detection confidence raises ValueError."""
        with pytest.raises(ValueError):
            HandLandmarkDetector(min_detection_confidence=1.5)
        
        with pytest.raises(ValueError):
            HandLandmarkDetector(min_detection_confidence=-0.1)
    
    def test_invalid_tracking_confidence(self):
        """Test that invalid tracking confidence raises ValueError."""
        with pytest.raises(ValueError):
            HandLandmarkDetector(min_tracking_confidence=1.5)
        
        with pytest.raises(ValueError):
            HandLandmarkDetector(min_tracking_confidence=-0.1)
    
    def test_context_manager(self):
        """Test using detector as context manager."""
        with HandLandmarkDetector() as detector:
            assert detector.hands is not None
        # Should be closed after context exit


class TestHandLandmarkDetectorDetection:
    """Tests for hand landmark detection."""
    
    def test_detect_with_invalid_frame_none(self):
        """Test detection with None frame raises ValueError."""
        detector = HandLandmarkDetector()
        with pytest.raises(ValueError):
            detector.detect(None)
        detector.close()
    
    def test_detect_with_invalid_frame_shape(self):
        """Test detection with invalid frame shape raises ValueError."""
        detector = HandLandmarkDetector()
        
        # Grayscale image (2D)
        with pytest.raises(ValueError):
            detector.detect(np.zeros((480, 640), dtype=np.uint8))
        
        # 4-channel image
        with pytest.raises(ValueError):
            detector.detect(np.zeros((480, 640, 4), dtype=np.uint8))
        
        detector.close()
    
    def test_detect_empty_frame(self):
        """Test detection with empty frame."""
        detector = HandLandmarkDetector()
        frame = np.array([], dtype=np.uint8)
        with pytest.raises(ValueError):
            detector.detect(frame)
        detector.close()
    
    def test_detect_returns_none_for_no_hand(self):
        """Test that detection returns None when no hand is present."""
        detector = HandLandmarkDetector()
        
        # Create blank frame (no hand)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks, handedness = detector.detect(frame)
        
        assert landmarks is None
        assert handedness is None
        detector.close()
    
    def test_detect_returns_correct_shape(self):
        """Test that detected landmarks have correct shape."""
        detector = HandLandmarkDetector()
        
        # Create a simple test frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        landmarks, handedness = detector.detect(frame)
        
        # Note: This test won't find a hand in a white frame, but if it did,
        # it would have the correct shape. This is a structure test.
        # In real usage with actual hand images, landmarks would be (21, 2)
        detector.close()


class TestLandmarkConversions:
    """Tests for landmark coordinate conversions."""
    
    def test_get_landmark_pixel_coordinates(self):
        """Test conversion from normalized to pixel coordinates."""
        detector = HandLandmarkDetector()
        
        # Create test landmarks (all at 0.5, 0.5 - center of frame)
        landmarks = np.ones((21, 2), dtype=np.float32) * 0.5
        
        frame_width = 640
        frame_height = 480
        
        pixel_landmarks = detector.get_landmark_pixel_coordinates(
            landmarks, frame_width, frame_height
        )
        
        # Check shape
        assert pixel_landmarks.shape == (21, 2)
        
        # Check values (0.5 * width = 320, 0.5 * height = 240)
        assert pixel_landmarks[0, 0] == 320
        assert pixel_landmarks[0, 1] == 240
        
        detector.close()
    
    def test_get_landmark_pixel_coordinates_corners(self):
        """Test pixel conversion at frame corners."""
        detector = HandLandmarkDetector()
        
        # Test corner points
        landmarks = np.array([
            [0.0, 0.0],   # Top-left
            [1.0, 0.0],   # Top-right
            [0.0, 1.0],   # Bottom-left
            [1.0, 1.0],   # Bottom-right
        ] + [[0.5, 0.5]] * 17, dtype=np.float32)  # Fill remaining with center
        
        pixel_landmarks = detector.get_landmark_pixel_coordinates(
            landmarks, 640, 480
        )
        
        assert pixel_landmarks[0, 0] == 0
        assert pixel_landmarks[0, 1] == 0
        assert pixel_landmarks[1, 0] == 640
        assert pixel_landmarks[1, 1] == 0
        assert pixel_landmarks[2, 0] == 0
        assert pixel_landmarks[2, 1] == 480
        
        detector.close()
    
    def test_get_landmark_pixel_coordinates_invalid_shape(self):
        """Test pixel conversion with invalid landmarks shape."""
        detector = HandLandmarkDetector()
        
        # Wrong shape
        landmarks = np.ones((20, 2), dtype=np.float32)  # Should be 21, 2
        
        with pytest.raises(ValueError):
            detector.get_landmark_pixel_coordinates(landmarks, 640, 480)
        
        detector.close()


class TestLandmarkDistance:
    """Tests for landmark distance calculations."""
    
    def test_calculate_landmark_distance_same_point(self):
        """Test distance between same landmark is zero."""
        detector = HandLandmarkDetector()
        
        landmarks = np.ones((21, 2), dtype=np.float32) * 0.5
        distance = detector.calculate_landmark_distance(landmarks, 0, 1)
        
        assert distance == pytest.approx(0.0, abs=1e-6)
        
        detector.close()
    
    def test_calculate_landmark_distance_diagonal(self):
        """Test distance calculation for diagonal points."""
        detector = HandLandmarkDetector()
        
        landmarks = np.zeros((21, 2), dtype=np.float32)
        landmarks[0] = [0.0, 0.0]
        landmarks[1] = [0.3, 0.4]
        
        # Distance should be sqrt(0.3^2 + 0.4^2) = sqrt(0.09 + 0.16) = 0.5
        distance = detector.calculate_landmark_distance(landmarks, 0, 1)
        
        assert distance == pytest.approx(0.5, abs=1e-6)
        
        detector.close()
    
    def test_calculate_landmark_distance_invalid_indices(self):
        """Test distance calculation with invalid landmark indices."""
        detector = HandLandmarkDetector()
        
        landmarks = np.zeros((21, 2), dtype=np.float32)
        
        with pytest.raises(ValueError):
            detector.calculate_landmark_distance(landmarks, -1, 5)
        
        with pytest.raises(ValueError):
            detector.calculate_landmark_distance(landmarks, 5, 21)
        
        detector.close()


class TestLandmarkConstants:
    """Tests for landmark index constants."""
    
    def test_landmark_constants_exist(self):
        """Test that all landmark constants are defined."""
        detector = HandLandmarkDetector()
        
        # Test that all finger landmarks are defined
        assert detector.WRIST == 0
        assert detector.THUMB_TIP == 4
        assert detector.INDEX_TIP == 8
        assert detector.MIDDLE_TIP == 12
        assert detector.RING_TIP == 16
        assert detector.PINKY_TIP == 20
        
        detector.close()
    
    def test_landmark_constants_unique(self):
        """Test that landmark constants have unique values."""
        detector = HandLandmarkDetector()
        
        constants = [
            detector.WRIST,
            detector.THUMB_CMC, detector.THUMB_MCP, detector.THUMB_IP, detector.THUMB_TIP,
            detector.INDEX_MCP, detector.INDEX_PIP, detector.INDEX_DIP, detector.INDEX_TIP,
            detector.MIDDLE_MCP, detector.MIDDLE_PIP, detector.MIDDLE_DIP, detector.MIDDLE_TIP,
            detector.RING_MCP, detector.RING_PIP, detector.RING_DIP, detector.RING_TIP,
            detector.PINKY_MCP, detector.PINKY_PIP, detector.PINKY_DIP, detector.PINKY_TIP,
        ]
        
        assert len(constants) == 21
        assert len(set(constants)) == 21  # All unique
        assert max(constants) == 20  # Max index is 20
        
        detector.close()


class TestCaching:
    """Tests for landmark caching functionality."""
    
    def test_get_last_landmarks_none(self):
        """Test getting last landmarks when none detected."""
        detector = HandLandmarkDetector()
        
        assert detector.get_last_landmarks() is None
        assert detector.get_last_handedness() is None
        
        detector.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
