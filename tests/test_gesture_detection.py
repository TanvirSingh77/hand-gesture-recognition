"""Unit tests for gesture detection module."""

import unittest
from src.gesture_detection import GestureDetector


class TestGestureDetector(unittest.TestCase):
    """Test cases for GestureDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = GestureDetector()
    
    def tearDown(self):
        """Clean up after tests."""
        self.detector.close()
    
    def test_detector_initialization(self):
        """Test that detector initializes correctly."""
        self.assertIsNotNone(self.detector.hands)
        self.assertIsNotNone(self.detector.mp_drawing)
    
    def test_detect_method_exists(self):
        """Test that detect method exists."""
        self.assertTrue(hasattr(self.detector, 'detect'))
    
    def test_draw_landmarks_method_exists(self):
        """Test that draw_landmarks method exists."""
        self.assertTrue(hasattr(self.detector, 'draw_landmarks'))


if __name__ == "__main__":
    unittest.main()
