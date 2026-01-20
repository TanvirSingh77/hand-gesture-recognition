"""
Hand Landmark Detection - Best Practices and Design Patterns

This document outlines clean code principles and design patterns used in
the HandLandmarkDetector module.
"""


# ============================================================================
# DESIGN PATTERNS
# ============================================================================

# 1. CONTEXT MANAGER PATTERN
# =========================
# Purpose: Ensure resources are properly cleaned up

"""
Pattern Implementation:
    with HandLandmarkDetector() as detector:
        landmarks, hand = detector.detect(frame)
    # Resources automatically cleaned up

Advantages:
    • Automatic resource cleanup (no memory leaks)
    • Exception-safe (cleanup happens even if error occurs)
    • Pythonic and readable
    • No need to remember to call close()
"""


# 2. SINGLE RESPONSIBILITY PRINCIPLE
# ===================================
# Purpose: Each method has one clear responsibility

"""
Examples in HandLandmarkDetector:

    detect() 
        → Detects landmarks in a frame
        → Nothing else
    
    get_landmark_pixel_coordinates()
        → Converts coordinates
        → Doesn't detect, doesn't calculate distances
    
    calculate_landmark_distance()
        → Calculates distance
        → Doesn't modify landmarks, doesn't draw

Advantages:
    • Easy to test (each method can be tested independently)
    • Easy to understand (clear purpose)
    • Easy to modify (changes don't affect other methods)
    • Reusable (methods can be used in different contexts)
"""


# 3. CONSTANT ENUMERATION
# =======================
# Purpose: Replace magic numbers with named constants

"""
Bad Practice:
    distance = detector.calculate_landmark_distance(landmarks, 4, 8)
    # What are 4 and 8? Nobody knows without looking at documentation

Good Practice:
    distance = detector.calculate_landmark_distance(
        landmarks,
        detector.THUMB_TIP,
        detector.INDEX_TIP
    )
    # Clear and self-documenting
"""


# ============================================================================
# CLEAN CODE PRINCIPLES
# ============================================================================

# 1. TYPE HINTS
# =============
# Usage: Every function has input and return type hints

"""
Benefits:
    • IDE autocomplete works better
    • Static type checking (mypy, pyright)
    • Self-documenting code
    • Easier debugging

Example:
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        # Returns either (landmarks, handedness) or (None, None)
"""


# 2. DOCSTRINGS
# ==============
# Format: Google-style docstrings with Args, Returns, Raises, Example

"""
Structure:
    - Brief description
    - Detailed description (if needed)
    - Args: Parameter descriptions
    - Returns: Return value descriptions
    - Raises: Possible exceptions
    - Example: Code example showing usage

Example:
    def calculate_landmark_distance(
        self,
        landmarks: np.ndarray,
        landmark_idx_1: int,
        landmark_idx_2: int
    ) -> float:
        '''Calculate Euclidean distance between two landmarks.
        
        Args:
            landmarks: Normalized landmarks array of shape (21, 2)
            landmark_idx_1: Index of first landmark (0-20)
            landmark_idx_2: Index of second landmark (0-20)
        
        Returns:
            Euclidean distance in normalized space
        
        Raises:
            ValueError: If landmark indices are out of range
        
        Example:
            >>> distance = detector.calculate_landmark_distance(
            ...     landmarks, detector.THUMB_TIP, detector.INDEX_TIP
            ... )
        '''
"""


# 3. ERROR HANDLING
# =================
# Strategy: Validate inputs early and raise meaningful exceptions

"""
Principles:
    • Fail fast (detect errors at entry point)
    • Clear error messages (user knows what went wrong)
    • Specific exceptions (ValueError, TypeError, not generic Exception)

Example:
    def __init__(self, min_detection_confidence: float = 0.5):
        if not (0.0 <= min_detection_confidence <= 1.0):
            raise ValueError(
                "min_detection_confidence must be between 0.0 and 1.0"
            )
        # ... rest of initialization

Usage:
    try:
        detector = HandLandmarkDetector(min_detection_confidence=1.5)
    except ValueError as e:
        print(f"Configuration error: {e}")
"""


# 4. SEPARATION OF CONCERNS
# ==========================
# Strategy: Each class/method handles one concern

"""
Concerns in HandLandmarkDetector:
    
    1. Detection
        - detect() uses MediaPipe to find landmarks
    
    2. Coordinate Conversion
        - get_landmark_pixel_coordinates() converts normalized→pixel
    
    3. Geometry Calculations
        - calculate_landmark_distance() calculates distances
        - get_hand_bounding_box() finds bounding box
    
    4. Resource Management
        - __init__() initializes resources
        - close() cleans up resources
        - __enter__/__exit__() implement context manager

Separation Advantages:
    • Each concern can be tested independently
    • Changes to one concern don't affect others
    • Code is modular and reusable
"""


# ============================================================================
# TESTING STRATEGY
# ============================================================================

# Test coverage is organized by functionality:

"""
1. Initialization Tests
    - Valid configurations
    - Invalid configurations (should raise errors)
    - Boundary values
    - Resource creation

2. Detection Tests
    - Valid frames
    - Invalid frames (None, wrong shape, etc.)
    - Expected outputs (shape and type)

3. Conversion Tests
    - Normalized → Pixel coordinates
    - Boundary cases (corners)
    - Invalid inputs

4. Distance Calculation Tests
    - Same landmark (distance = 0)
    - Known distances (verified mathematically)
    - Invalid indices

5. Resource Management Tests
    - Context manager usage
    - Resource cleanup
    - Reusability after close()

6. Integration Tests
    - Real detections (with actual hand images)
    - End-to-end workflows
"""


# ============================================================================
# PERFORMANCE CONSIDERATIONS
# ============================================================================

# 1. Lazy Initialization
# ======================
"""
Don't initialize MediaPipe until needed:
    self.hands = self.mp_hands.Hands(...)
    # This is expensive, only do once in __init__
"""

# 2. Caching
# ==========
"""
Store last results for reuse:
    self._last_landmarks = landmarks
    self._last_handedness = handedness
    
Usage:
    last_landmarks = detector.get_last_landmarks()
    # Useful if you need landmarks without processing a new frame
"""

# 3. Numpy Optimization
# =====================
"""
Use numpy arrays (fast, vectorized):
    landmarks = np.array(landmarks_2d, dtype=np.float32)
    # More efficient than Python lists
"""

# 4. Frame Size Optimization
# ===========================
"""
For real-time performance:
    • Keep frames at reasonable resolution (640x480 is good)
    • Avoid very large frames (1080p+ slows things down)
    • Consider resizing: frame = cv2.resize(frame, (640, 480))
"""

# 5. Detection Confidence Tuning
# ===============================
"""
Trade-off: accuracy vs speed

Lower thresholds → Faster, more detections (more false positives)
Higher thresholds → Slower, fewer detections (more accurate)

For real-time:
    min_detection_confidence=0.7  # Good balance
    min_tracking_confidence=0.5
"""


# ============================================================================
# CODE STYLE AND FORMATTING
# ============================================================================

"""
Following Python PEP-8:
    • 4-space indentation
    • CamelCase for classes
    • snake_case for functions and variables
    • UPPER_CASE for constants
    • Maximum 88 characters per line (Black style)
    • Type hints on all public methods
    • Docstrings for all public classes and methods
"""


# ============================================================================
# EXTENSIBILITY AND MAINTENANCE
# ============================================================================

# The module is designed to be extended easily:

"""
Adding a new method:

    def get_finger_angles(self, landmarks: np.ndarray) -> np.ndarray:
        '''Calculate angles of each finger.
        
        Args:
            landmarks: Normalized landmarks array (21, 2)
        
        Returns:
            Array of finger angles
        '''
        # Implementation would go here
        pass

Adding a new constant:

    # In __init__ or as class variable
    INDEX_PIP = 6  # Already defined, but for example:

Adding support for multiple hands:

    # Currently optimized for single hand (max_num_hands=1)
    # Could be extended by:
    # 1. Removing max_num_hands=1 restriction
    # 2. Returning list of landmarks instead of single landmarks
    # 3. Returning list of handedness values
    # 4. Adding methods to process multiple hands
"""


# ============================================================================
# INTEGRATION GUIDELINES
# ============================================================================

"""
When integrating with other modules:

1. With Camera Module
    from src.camera import CameraCapture
    
    camera = CameraCapture(camera_id=0)
    camera.start()
    ret, frame = camera.get_frame()
    landmarks, hand = detector.detect(frame)

2. With Gesture Classifier
    from src.gesture_classifier import GestureClassifier
    
    features = landmarks.flatten()  # (42,) array
    gesture = classifier.predict(features)

3. With Utils Module
    from src.utils import draw_text, draw_fps
    
    frame = draw_text(frame, f"Hand: {hand}", (10, 30))
    frame = draw_fps(frame, fps_value)
"""


# ============================================================================
# COMMON PITFALLS AND HOW TO AVOID THEM
# ============================================================================

"""
1. FORGETTING TO CONVERT COORDINATES
    ❌ Bad:
        landmarks, _ = detector.detect(frame)
        cv2.circle(frame, landmarks[0], 5, (0,255,0), -1)  # Landmarks are 0-1!
    
    ✅ Good:
        landmarks, _ = detector.detect(frame)
        h, w = frame.shape[:2]
        px_lm = detector.get_landmark_pixel_coordinates(landmarks, w, h)
        cv2.circle(frame, tuple(px_lm[0]), 5, (0,255,0), -1)

2. NOT CHECKING FOR NONE
    ❌ Bad:
        landmarks, hand = detector.detect(frame)
        distance = detector.calculate_landmark_distance(landmarks, 4, 8)
        # Crashes if landmarks is None
    
    ✅ Good:
        landmarks, hand = detector.detect(frame)
        if landmarks is not None:
            distance = detector.calculate_landmark_distance(landmarks, 4, 8)

3. FORGETTING TO CLOSE DETECTOR
    ❌ Bad:
        detector = HandLandmarkDetector()
        # ... use detector ...
        # Resources leak if script exits or errors occur
    
    ✅ Good:
        with HandLandmarkDetector() as detector:
            # ... use detector ...
        # Automatically cleaned up

4. USING MAGIC NUMBERS
    ❌ Bad:
        distance = detector.calculate_landmark_distance(landmarks, 4, 8)
        # What are 4 and 8?
    
    ✅ Good:
        distance = detector.calculate_landmark_distance(
            landmarks,
            detector.THUMB_TIP,
            detector.INDEX_TIP
        )

5. INEFFICIENT DETECTION LOOP
    ❌ Bad:
        while True:
            detector = HandLandmarkDetector()  # Reinitialize every frame!
            landmarks, _ = detector.detect(frame)
    
    ✅ Good:
        detector = HandLandmarkDetector()
        while True:
            landmarks, _ = detector.detect(frame)
        detector.close()
"""


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

"""
Expected Performance (on typical hardware):
    • Frame rate: 25-35 FPS (depending on resolution)
    • Latency: 30-50 ms per frame
    • Memory: ~100-200 MB
    • CPU usage: 15-25% (single core)

Factors affecting performance:
    • Frame resolution (larger = slower)
    • Detection confidence thresholds (higher = slower)
    • Number of hands (more = slower)
    • Hardware (CPU vs GPU)
    • System load
"""


# ============================================================================
# DEBUGGING TIPS
# ============================================================================

"""
To debug hand detection issues:

1. Visualize landmarks:
    if landmarks is not None:
        h, w = frame.shape[:2]
        px_lm = detector.get_landmark_pixel_coordinates(landmarks, w, h)
        for i, point in enumerate(px_lm):
            cv2.circle(frame, tuple(point), 5, (0,255,0), -1)
            cv2.putText(frame, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
        cv2.imshow('Landmarks', frame)

2. Print landmark values:
    print(landmarks[detector.THUMB_TIP])  # Should be close to [thumb x, thumb y]

3. Check distances:
    dist = detector.calculate_landmark_distance(landmarks, detector.THUMB_TIP, detector.INDEX_TIP)
    print(f"Thumb-Index distance: {dist:.3f}")

4. Verify detection:
    landmarks, hand = detector.detect(frame)
    print(f"Hand detected: {hand}")
    print(f"Landmarks shape: {landmarks.shape if landmarks is not None else 'None'}")
"""


if __name__ == "__main__":
    print(__doc__)
