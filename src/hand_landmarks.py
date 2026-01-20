"""Hand landmark detection module using MediaPipe.

This module provides an optimized, reusable interface for detecting hand landmarks
in real-time from video frames. It detects a single hand and extracts normalized
landmark coordinates for efficient processing.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List


class HandLandmarkDetector:
    """Detects hand landmarks from video frames with real-time optimization.
    
    This class wraps MediaPipe's hand detection to provide a clean interface for:
    - Single hand detection (optimized for real-time performance)
    - Extraction of 21 normalized landmarks
    - Efficient frame preprocessing
    - Configurable detection thresholds
    """
    
    # Number of hand landmarks detected by MediaPipe
    NUM_LANDMARKS = 21
    
    # Landmark indices for quick reference
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False
    ):
        """Initialize the hand landmark detector.
        
        Args:
            min_detection_confidence: Minimum confidence threshold for hand detection
                (0.0 to 1.0). Lower values are more sensitive but may include false positives.
            min_tracking_confidence: Minimum confidence threshold for hand tracking
                (0.0 to 1.0). Used after initial detection for temporal consistency.
            static_image_mode: If True, treats each frame independently. If False,
                uses tracking for better performance. Default is False for real-time video.
        
        Raises:
            ValueError: If confidence thresholds are not in valid range [0.0, 1.0]
        """
        if not (0.0 <= min_detection_confidence <= 1.0):
            raise ValueError("min_detection_confidence must be between 0.0 and 1.0")
        if not (0.0 <= min_tracking_confidence <= 1.0):
            raise ValueError("min_tracking_confidence must be between 0.0 and 1.0")
        
        # Initialize MediaPipe hand detector
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,  # Optimize for single hand detection
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Cache for storing current frame results
        self._last_landmarks = None
        self._last_handedness = None
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Detect hand landmarks in a video frame.
        
        Args:
            frame: Input video frame in BGR format (from OpenCV).
                   Shape should be (height, width, 3).
        
        Returns:
            Tuple containing:
            - landmarks: numpy array of shape (21, 2) containing normalized (x, y)
                        coordinates for each landmark, or None if no hand detected
            - handedness: String 'Left' or 'Right' indicating hand orientation,
                         or None if no hand detected
        
        Raises:
            ValueError: If frame is empty or has invalid shape
        
        Example:
            >>> detector = HandLandmarkDetector()
            >>> frame = cv2.imread('hand.jpg')
            >>> landmarks, hand_side = detector.detect(frame)
            >>> if landmarks is not None:
            ...     print(f"Detected {hand_side} hand with {len(landmarks)} landmarks")
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame cannot be empty")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must have shape (height, width, 3), got {frame.shape}")
        
        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run MediaPipe hand detection
        results = self.hands.process(rgb_frame)
        
        # Extract landmarks if hand is detected
        if results.multi_hand_landmarks and results.multi_handedness:
            # Extract first (and only) hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            
            # Convert landmarks to normalized (x, y) coordinates
            landmarks = self._extract_normalized_landmarks(hand_landmarks)
            
            # Cache results for potential reuse
            self._last_landmarks = landmarks
            self._last_handedness = handedness
            
            return landmarks, handedness
        
        return None, None
    
    def _extract_normalized_landmarks(
        self,
        hand_landmarks: 'mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList'
    ) -> np.ndarray:
        """Extract normalized (x, y) coordinates from MediaPipe landmarks.
        
        MediaPipe returns landmarks with x, y, z coordinates where:
        - x, y: Normalized coordinates (0.0 to 1.0) in frame space
        - z: Relative depth (used for 3D, excluded here for 2D detection)
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
        
        Returns:
            numpy array of shape (21, 2) with normalized (x, y) coordinates
        """
        landmarks_2d = []
        
        for landmark in hand_landmarks.landmark:
            # Extract normalized x, y coordinates (z is ignored for 2D)
            landmarks_2d.append([landmark.x, landmark.y])
        
        return np.array(landmarks_2d, dtype=np.float32)
    
    def get_landmark_pixel_coordinates(
        self,
        landmarks: np.ndarray,
        frame_width: int,
        frame_height: int
    ) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates.
        
        Args:
            landmarks: Normalized landmarks array of shape (21, 2)
            frame_width: Width of the video frame in pixels
            frame_height: Height of the video frame in pixels
        
        Returns:
            numpy array of shape (21, 2) with pixel coordinates
        
        Raises:
            ValueError: If landmarks array has incorrect shape
        
        Example:
            >>> landmarks_norm, _ = detector.detect(frame)
            >>> h, w = frame.shape[:2]
            >>> landmarks_px = detector.get_landmark_pixel_coordinates(landmarks_norm, w, h)
        """
        if landmarks.shape != (self.NUM_LANDMARKS, 2):
            raise ValueError(
                f"Landmarks array must have shape (21, 2), got {landmarks.shape}"
            )
        
        pixel_landmarks = landmarks.copy()
        pixel_landmarks[:, 0] *= frame_width   # Scale x coordinates
        pixel_landmarks[:, 1] *= frame_height  # Scale y coordinates
        
        return pixel_landmarks.astype(np.int32)
    
    def get_hand_bounding_box(
        self,
        landmarks: np.ndarray,
        frame_width: int,
        frame_height: int,
        padding: float = 0.1
    ) -> Tuple[int, int, int, int]:
        """Calculate bounding box around detected hand landmarks.
        
        Args:
            landmarks: Normalized landmarks array of shape (21, 2)
            frame_width: Width of the video frame in pixels
            frame_height: Height of the video frame in pixels
            padding: Padding factor as fraction of box size (default: 0.1 = 10%)
        
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates
        
        Example:
            >>> landmarks_norm, _ = detector.detect(frame)
            >>> h, w = frame.shape[:2]
            >>> x_min, y_min, x_max, y_max = detector.get_hand_bounding_box(
            ...     landmarks_norm, w, h, padding=0.1
            ... )
        """
        # Convert to pixel coordinates for easier calculation
        pixel_landmarks = self.get_landmark_pixel_coordinates(landmarks, frame_width, frame_height)
        
        # Find min/max coordinates
        x_coords = pixel_landmarks[:, 0]
        y_coords = pixel_landmarks[:, 1]
        
        x_min = int(np.min(x_coords))
        y_min = int(np.min(y_coords))
        x_max = int(np.max(x_coords))
        y_max = int(np.max(y_coords))
        
        # Apply padding
        width = x_max - x_min
        height = y_max - y_max
        
        x_min = max(0, int(x_min - width * padding))
        y_min = max(0, int(y_min - height * padding))
        x_max = min(frame_width, int(x_max + width * padding))
        y_max = min(frame_height, int(y_max + height * padding))
        
        return x_min, y_min, x_max, y_max
    
    def calculate_landmark_distance(
        self,
        landmarks: np.ndarray,
        landmark_idx_1: int,
        landmark_idx_2: int
    ) -> float:
        """Calculate Euclidean distance between two landmarks.
        
        Args:
            landmarks: Normalized landmarks array of shape (21, 2)
            landmark_idx_1: Index of first landmark (0-20)
            landmark_idx_2: Index of second landmark (0-20)
        
        Returns:
            Euclidean distance in normalized space (0.0 to ~1.4)
        
        Raises:
            ValueError: If landmark indices are out of range
        
        Example:
            >>> landmarks_norm, _ = detector.detect(frame)
            >>> distance = detector.calculate_landmark_distance(
            ...     landmarks_norm, HandLandmarkDetector.INDEX_TIP, HandLandmarkDetector.THUMB_TIP
            ... )
        """
        if not (0 <= landmark_idx_1 < self.NUM_LANDMARKS):
            raise ValueError(f"Landmark index 1 must be 0-20, got {landmark_idx_1}")
        if not (0 <= landmark_idx_2 < self.NUM_LANDMARKS):
            raise ValueError(f"Landmark index 2 must be 0-20, got {landmark_idx_2}")
        
        point1 = landmarks[landmark_idx_1]
        point2 = landmarks[landmark_idx_2]
        
        return float(np.linalg.norm(point1 - point2))
    
    def get_last_landmarks(self) -> Optional[np.ndarray]:
        """Get the last detected landmarks without processing a new frame.
        
        Returns:
            numpy array of shape (21, 2) with normalized coordinates, or None if
            no landmarks have been detected yet
        """
        return self._last_landmarks
    
    def get_last_handedness(self) -> Optional[str]:
        """Get the handedness of the last detected landmarks.
        
        Returns:
            String 'Left' or 'Right', or None if no landmarks have been detected yet
        """
        return self._last_handedness
    
    def close(self):
        """Release resources held by the detector.
        
        Should be called when the detector is no longer needed to free
        MediaPipe resources.
        """
        if self.hands is not None:
            self.hands.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic resource cleanup."""
        self.close()
