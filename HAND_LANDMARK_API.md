"""
Hand Landmark Detection Module - Complete API Reference and Usage Guide

This module provides a reusable, production-ready interface for detecting hand
landmarks from video frames using MediaPipe. It's optimized for real-time
performance and follows Python best practices.
"""


# ============================================================================
# QUICK START
# ============================================================================

"""
Basic usage in 5 lines:

    from src.hand_landmarks import HandLandmarkDetector
    
    detector = HandLandmarkDetector()
    landmarks, handedness = detector.detect(frame)  # frame is a numpy array
    if landmarks is not None:
        print(f"Detected {handedness} hand with 21 landmarks")
"""


# ============================================================================
# CLASS: HandLandmarkDetector
# ============================================================================

"""
HandLandmarkDetector
    
A wrapper around MediaPipe's hand detection optimized for real-time applications.

FEATURES:
    • Single hand detection (optimized for real-time)
    • Detects 21 hand landmarks per hand
    • Returns normalized (x, y) coordinates (0.0 to 1.0)
    • Real-time performance optimization
    • Named landmark constants for easy reference
    • Utility methods for common operations
    • Full type hints for IDE support
    • Context manager support for resource cleanup
    • Comprehensive error handling

LANDMARKS DETECTED (21 total):
    Each hand has 21 landmarks corresponding to different joints:
    
    • Wrist (1): Base of hand
    • Thumb (4): CMC, MCP, IP, Tip
    • Index Finger (4): MCP, PIP, DIP, Tip
    • Middle Finger (4): MCP, PIP, DIP, Tip
    • Ring Finger (4): MCP, PIP, DIP, Tip
    • Pinky (4): MCP, PIP, DIP, Tip

COORDINATE SYSTEM:
    • x, y are normalized to frame dimensions (0.0 to 1.0)
    • (0, 0) is top-left corner
    • (1, 1) is bottom-right corner
    • Useful for frame-size-independent processing
"""


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_basic():
    """Initialize with default settings (recommended for most cases)."""
    from src.hand_landmarks import HandLandmarkDetector
    
    detector = HandLandmarkDetector()
    return detector


def init_with_custom_thresholds():
    """Initialize with custom confidence thresholds."""
    from src.hand_landmarks import HandLandmarkDetector
    
    # Higher thresholds = stricter detection (fewer false positives)
    # Lower thresholds = more lenient (more sensitivity)
    detector = HandLandmarkDetector(
        min_detection_confidence=0.8,    # Initial detection threshold
        min_tracking_confidence=0.6,     # Temporal tracking threshold
        static_image_mode=False          # Use tracking for video
    )
    return detector


def init_with_context_manager():
    """Initialize using context manager for automatic cleanup."""
    from src.hand_landmarks import HandLandmarkDetector
    
    with HandLandmarkDetector() as detector:
        # Use detector here
        pass
    # Resources are automatically released


# ============================================================================
# BASIC DETECTION
# ============================================================================

def detect_hand_landmarks(detector, frame):
    """
    Detect hand landmarks in a frame.
    
    Args:
        detector: HandLandmarkDetector instance
        frame: numpy array of shape (height, width, 3) in BGR format
    
    Returns:
        Tuple of (landmarks, handedness) or (None, None) if no hand detected
        - landmarks: numpy array shape (21, 2) with normalized (x, y) coordinates
        - handedness: 'Left' or 'Right' string
    """
    landmarks, handedness = detector.detect(frame)
    
    if landmarks is not None:
        print(f"Detected {handedness} hand")
        print(f"Landmark shape: {landmarks.shape}")  # (21, 2)
        print(f"First landmark (wrist): {landmarks[0]}")  # [0.5, 0.3]
    else:
        print("No hand detected")
    
    return landmarks, handedness


# ============================================================================
# ACCESSING LANDMARKS
# ============================================================================

def access_landmarks_by_name():
    """Access specific landmarks using named constants."""
    from src.hand_landmarks import HandLandmarkDetector
    import numpy as np
    
    detector = HandLandmarkDetector()
    landmarks = np.random.rand(21, 2)  # Mock detection
    
    # Access by name instead of magic numbers
    wrist = landmarks[detector.WRIST]          # Index 0
    thumb_tip = landmarks[detector.THUMB_TIP]  # Index 4
    index_tip = landmarks[detector.INDEX_TIP]  # Index 8
    
    print(f"Wrist position: {wrist}")
    print(f"Thumb tip: {thumb_tip}")
    print(f"Index tip: {index_tip}")
    
    detector.close()


def access_landmarks_by_index():
    """Access landmarks by direct index."""
    import numpy as np
    
    landmarks = np.random.rand(21, 2)
    
    # All 21 landmarks in order:
    # [0] Wrist
    # [1-4] Thumb (CMC, MCP, IP, Tip)
    # [5-8] Index finger (MCP, PIP, DIP, Tip)
    # [9-12] Middle finger (MCP, PIP, DIP, Tip)
    # [13-16] Ring finger (MCP, PIP, DIP, Tip)
    # [17-20] Pinky (MCP, PIP, DIP, Tip)
    
    for i, (x, y) in enumerate(landmarks):
        print(f"Landmark {i}: x={x:.3f}, y={y:.3f}")


# ============================================================================
# COORDINATE CONVERSIONS
# ============================================================================

def convert_to_pixel_coordinates():
    """Convert normalized coordinates to pixel coordinates."""
    from src.hand_landmarks import HandLandmarkDetector
    import numpy as np
    import cv2
    
    detector = HandLandmarkDetector()
    frame = cv2.imread('hand_image.jpg')
    h, w = frame.shape[:2]
    
    landmarks, _ = detector.detect(frame)
    
    if landmarks is not None:
        # Convert normalized to pixel coordinates
        pixel_landmarks = detector.get_landmark_pixel_coordinates(
            landmarks, w, h
        )
        
        # Now pixel_landmarks are in pixel space (0 to width/height)
        for i, (px, py) in enumerate(pixel_landmarks):
            cv2.circle(frame, (int(px), int(py)), 5, (0, 255, 0), -1)
        
        cv2.imshow('Landmarks', frame)
        cv2.waitKey(0)
    
    detector.close()


# ============================================================================
# DISTANCE CALCULATIONS
# ============================================================================

def detect_finger_pinch():
    """Detect pinching gesture by calculating finger distances."""
    from src.hand_landmarks import HandLandmarkDetector
    import cv2
    
    detector = HandLandmarkDetector()
    cap = cv2.VideoCapture(0)
    
    PINCH_THRESHOLD = 0.03  # Normalized distance threshold
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks, _ = detector.detect(frame)
        
        if landmarks is not None:
            # Calculate distance between thumb and index finger tips
            distance = detector.calculate_landmark_distance(
                landmarks,
                detector.THUMB_TIP,
                detector.INDEX_TIP
            )
            
            if distance < PINCH_THRESHOLD:
                print("Pinch detected!")
                cv2.putText(frame, "PINCH", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Pinch Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


def measure_hand_spread():
    """Measure overall hand spread."""
    from src.hand_landmarks import HandLandmarkDetector
    
    detector = HandLandmarkDetector()
    # Assuming landmarks detected...
    landmarks = None  # Placeholder
    
    if landmarks is not None:
        # Calculate hand span (thumb tip to pinky tip)
        hand_span = detector.calculate_landmark_distance(
            landmarks,
            detector.THUMB_TIP,
            detector.PINKY_TIP
        )
        
        # Calculate finger spread (index to pinky)
        finger_spread = detector.calculate_landmark_distance(
            landmarks,
            detector.INDEX_MCP,
            detector.PINKY_MCP
        )
        
        print(f"Hand span: {hand_span:.3f}")
        print(f"Finger spread: {finger_spread:.3f}")
    
    detector.close()


# ============================================================================
# BOUNDING BOX
# ============================================================================

def get_hand_region():
    """Extract hand region using bounding box."""
    from src.hand_landmarks import HandLandmarkDetector
    import cv2
    
    detector = HandLandmarkDetector()
    frame = cv2.imread('hand_image.jpg')
    h, w = frame.shape[:2]
    
    landmarks, _ = detector.detect(frame)
    
    if landmarks is not None:
        x_min, y_min, x_max, y_max = detector.get_hand_bounding_box(
            landmarks, w, h, padding=0.1
        )
        
        # Crop hand region
        hand_region = frame[y_min:y_max, x_min:x_max]
        cv2.imshow('Hand Region', hand_region)
        cv2.waitKey(0)
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow('Bounding Box', frame)
        cv2.waitKey(0)
    
    detector.close()


# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

def optimize_for_real_time():
    """Best practices for real-time hand detection."""
    from src.hand_landmarks import HandLandmarkDetector
    import cv2
    
    # Initialize once
    detector = HandLandmarkDetector(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        static_image_mode=False  # Critical for real-time!
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Detect
        landmarks, handedness = detector.detect(frame)
        
        # Quick processing
        if landmarks is not None:
            # Use cached values if available
            last_lm = detector.get_last_landmarks()
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


# ============================================================================
# ERROR HANDLING
# ============================================================================

def handle_errors_safely():
    """Proper error handling when working with detector."""
    from src.hand_landmarks import HandLandmarkDetector
    import cv2
    
    detector = None
    cap = None
    
    try:
        detector = HandLandmarkDetector()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        ret, frame = cap.read()
        
        if not ret:
            raise RuntimeError("Cannot read frame")
        
        landmarks, handedness = detector.detect(frame)
        
        if landmarks is None:
            print("No hand detected in current frame")
        else:
            print(f"Detected {handedness} hand with {len(landmarks)} landmarks")
    
    except ValueError as e:
        print(f"Invalid input: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
    finally:
        if cap:
            cap.release()
        if detector:
            detector.close()
        cv2.destroyAllWindows()


# ============================================================================
# INTEGRATION WITH GESTURE RECOGNITION
# ============================================================================

def integrate_with_gesture_classifier():
    """Use landmarks for downstream gesture classification."""
    from src.hand_landmarks import HandLandmarkDetector
    import numpy as np
    
    detector = HandLandmarkDetector()
    # Assume frame is available...
    frame = None  # Placeholder
    
    if frame is not None:
        landmarks, handedness = detector.detect(frame)
        
        if landmarks is not None:
            # Prepare for classification
            # Flatten landmarks for ML models
            feature_vector = landmarks.flatten()  # Shape: (42,)
            
            # Or use specific feature combinations
            features = np.array([
                detector.calculate_landmark_distance(
                    landmarks, detector.THUMB_TIP, detector.INDEX_TIP
                ),
                detector.calculate_landmark_distance(
                    landmarks, detector.INDEX_TIP, detector.MIDDLE_TIP
                ),
                # Add more features as needed...
            ])
            
            print(f"Feature vector shape: {feature_vector.shape}")
            print(f"Custom features shape: {features.shape}")
            
            # Pass to classifier...
    
    detector.close()


# ============================================================================
# MEMORY AND RESOURCE MANAGEMENT
# ============================================================================

def proper_cleanup():
    """Always clean up resources properly."""
    from src.hand_landmarks import HandLandmarkDetector
    
    # Method 1: Explicit cleanup
    detector = HandLandmarkDetector()
    try:
        # Use detector
        pass
    finally:
        detector.close()
    
    # Method 2: Context manager (recommended)
    with HandLandmarkDetector() as detector:
        # Use detector
        pass
    # Automatically closed


# ============================================================================
# COMMON PATTERNS
# ============================================================================

def pattern_single_image():
    """Process a single image."""
    from src.hand_landmarks import HandLandmarkDetector
    import cv2
    
    with HandLandmarkDetector() as detector:
        frame = cv2.imread('hand.jpg')
        landmarks, hand = detector.detect(frame)
        
        if landmarks is not None:
            print(f"{hand} hand detected")


def pattern_video_stream():
    """Process video stream."""
    from src.hand_landmarks import HandLandmarkDetector
    from src.camera import CameraCapture
    import cv2
    
    with HandLandmarkDetector() as detector:
        with CameraCapture() as cam:
            cam.start()
            
            while True:
                ret, frame = cam.get_frame()
                if not ret:
                    break
                
                landmarks, hand = detector.detect(frame)
                
                if landmarks is not None:
                    # Process landmarks
                    pass
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def pattern_batch_processing():
    """Process multiple images in batch."""
    from src.hand_landmarks import HandLandmarkDetector
    import cv2
    import glob
    
    with HandLandmarkDetector() as detector:
        for image_path in glob.glob('images/*.jpg'):
            frame = cv2.imread(image_path)
            landmarks, hand = detector.detect(frame)
            
            if landmarks is not None:
                # Process and save results
                pass


# ============================================================================
# TESTING
# ============================================================================

"""
Run tests with:
    pytest tests/test_hand_landmarks.py -v

Test coverage includes:
    • Initialization
    • Invalid inputs
    • Coordinate conversions
    • Distance calculations
    • Landmark constants
    • Resource management
"""


if __name__ == "__main__":
    print(__doc__)
