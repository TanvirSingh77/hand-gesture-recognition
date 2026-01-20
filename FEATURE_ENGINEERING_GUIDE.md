"""
FEATURE ENGINEERING MODULE DOCUMENTATION

==============================================================================
Overview
==============================================================================

The HandGestureFeatureExtractor is a comprehensive feature engineering module
that transforms raw hand landmarks into meaningful numerical features for
machine learning models. It extracts 46 features organized into 4 categories:

1. Inter-joint distances (21 features)
2. Joint angles (15 features)
3. Hand span metrics (4 features)
4. Relative positions (6 features)

All features are designed to be:
- Scale-invariant (normalized distances)
- Rotation-invariant (using relative angles)
- Reusable for both training and inference
- Interpretable (clearly documented)


==============================================================================
Feature Categories
==============================================================================

1. INTER-JOINT DISTANCES (21 features)
================================================================================

These features capture the overall geometry and spread of the hand by measuring
distances between key joints. All distances are normalized by the hand bounding
box diagonal for scale invariance.

Distances computed:

a) Wrist to Fingertips (5 features)
   - Wrist → Thumb Tip
   - Wrist → Index Tip
   - Wrist → Middle Tip
   - Wrist → Ring Tip
   - Wrist → Pinky Tip
   
   Interpretation: How "open" each finger is relative to wrist
   Example: High values indicate extended fingers; low values indicate bent/closed fingers

b) Finger Inter-joint Distances (12 features)
   
   For each finger (Thumb, Index, Middle, Ring, Pinky):
   - Joint 1 → Joint 2 distance
   - Joint 2 → Joint 3 distance
   - Joint 3 → Joint 4 (Tip) distance
   
   Interpretation: How much each joint segment is extended
   Example: Thumb inter-joint distances tell us if thumb is bent or straight

c) MCP Joints Spread (4 features)
   - Thumb MCP → Index MCP distance
   - Index MCP → Middle MCP distance
   - Middle MCP → Ring MCP distance
   - Ring MCP → Pinky MCP distance
   
   Interpretation: How spread out the fingers are
   Example: Large values indicate wide finger spread; small values indicate fingers close together


2. JOINT ANGLES (15 features)
================================================================================

These features measure the bending/flexion state of each finger by computing
angles between finger segments. Important for distinguishing between gestures
like "peace" (fingers extended, high angles) vs "fist" (fingers bent, low angles).

Angles computed:

For each finger (Thumb, Index, Middle, Ring, Pinky):
   - Angle at intermediate joint (e.g., MCP-PIP-DIP)
   - Angle at distal joint (e.g., PIP-DIP-TIP)
   - Overall angle (e.g., MCP-PIP-TIP) - measures overall bending

Total: 3 angles × 5 fingers = 15 angles

Range: [0°, 180°]
   - 180°: Joint fully extended (straight)
   - 90°: Joint at right angle (slightly bent)
   - 0°: Joint fully flexed (maximum bend)

Interpretation:
   - "Peace" gesture: Most angles close to 180° (extended)
   - "Fist" gesture: Angles smaller (bent)
   - "OK" gesture: Different angles for different fingers


3. HAND SPAN METRICS (4 features)
================================================================================

These features capture the overall size and shape of the hand.

Metrics:

a) Hand Bounding Box Width
   - Maximum X coordinate - Minimum X coordinate
   - Interpretation: Horizontal spread of hand

b) Hand Bounding Box Height
   - Maximum Y coordinate - Minimum Y coordinate
   - Interpretation: Vertical extent of hand

c) Hand Aspect Ratio
   - Width / Height
   - Interpretation: Overall hand shape (wide vs tall)
   - Example: High ratio means hand is wider than tall

d) Maximum Reach from Wrist
   - Maximum distance from wrist to any other joint
   - Interpretation: How extended the hand is overall
   - Example: Large value indicates open hand, small value indicates closed hand


4. RELATIVE POSITIONS (6 features)
================================================================================

These features capture the relative positioning of key finger tips within the
hand bounding box. All values normalized to [0, 1].

Positions:

a) Wrist Relative Y Position
   - (Wrist Y - Min Y) / Height
   - Interpretation: Position of wrist in hand
   - Example: 0 means wrist at top, 1 means wrist at bottom

b-c) Thumb Tip Position
   - Thumb tip X coordinate (relative to bbox): Horizontal position
   - Thumb tip Y coordinate (relative to bbox): Vertical position

d-e) Index Tip Position
   - Index tip X coordinate (relative to bbox)
   - Index tip Y coordinate (relative to bbox)

f) Pinky Tip Position
   - Pinky tip X coordinate (relative to bbox)

Interpretation:
   - These features help distinguish gestures based on finger positioning
   - Example: "Thumbs up" has thumb tip at high Y (bottom), "Thumbs down" has it at low Y (top)


==============================================================================
Usage Examples
==============================================================================

EXAMPLE 1: Basic Feature Extraction
────────────────────────────────────

from src.hand_landmarks import HandLandmarkDetector
from src.feature_extractor import HandGestureFeatureExtractor

# Initialize detector and extractor
detector = HandLandmarkDetector()
extractor = HandGestureFeatureExtractor(normalize=True)

# Capture frame from webcam
frame = capture_webcam_frame()  # numpy array [H, W, 3]

# Detect hand landmarks
landmarks, handedness = detector.detect(frame)

# Extract features
features = extractor.extract(landmarks)  # shape: (46,)

print(f"Feature vector: {features}")
print(f"Feature shape: {features.shape}")


EXAMPLE 2: Detailed Feature Breakdown
──────────────────────────────────────

# Extract features with detailed breakdown
result = extractor.extract(landmarks, return_dict=True)

# Access individual feature groups
distances = result['distances']      # (21,) - inter-joint distances
angles = result['angles']            # (15,) - joint angles
spans = result['spans']              # (4,)  - hand span metrics
positions = result['positions']      # (6,)  - relative positions

# Get feature names for interpretation
names = extractor.get_feature_names()  # List of 46 descriptive names

# Print feature information
for name, value in zip(names, result['vector']):
    print(f"{name:35s}: {value:7.4f}")


EXAMPLE 3: Processing Video Sequence
─────────────────────────────────────

from examples_feature_extraction import RealtimeFeatureExtractor

# Initialize real-time extractor
extractor = RealtimeFeatureExtractor(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Process video file
results = extractor.process_video(
    video_path="gesture_video.mp4",
    output_path="output_with_landmarks.mp4",
    display=True
)

# Extract feature vectors
feature_vectors = [features for features, _ in results if features is not None]
feature_matrix = np.array(feature_vectors)  # shape: (n_frames, 46)

# Export to CSV for model training
extractor.export_features("extracted_features.csv")


EXAMPLE 4: Real-time Webcam Feature Extraction
───────────────────────────────────────────────

# Process webcam stream
results = extractor.process_webcam(duration_seconds=60)

# Get statistics
print(f"Total frames: {extractor.frame_count}")
print(f"Hands detected: {extractor.detection_count}")
print(f"Detection rate: {extractor.detection_count/extractor.frame_count*100:.1f}%")

# Export features for training
extractor.export_features("webcam_features.csv")


EXAMPLE 5: Training a Gesture Classifier
─────────────────────────────────────────

from sklearn.ensemble import RandomForestClassifier
from src.data_utils import GestureDataLoader

# Load collected gesture data
loader = GestureDataLoader(data_dir="data/collected_gestures")

# Get feature vectors and labels
X, y, gesture_names = loader.get_feature_vectors(
    aggregation='mean',  # Aggregate frames within each sample
    normalize=True
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Get feature importance
feature_names = extractor.get_feature_names()
for i, (name, importance) in enumerate(zip(feature_names, clf.feature_importances_)):
    if importance > 0.01:  # Show top features
        print(f"{name:35s}: {importance:.4f}")


==============================================================================
Feature Normalization and Scale Invariance
==============================================================================

The feature extractor automatically normalizes distances by the hand bounding
box diagonal, making it scale-invariant. This means:

✓ Same gesture at different distances from camera → Same features
✓ Same gesture performed with different hand sizes → Same features
✓ Robust to varying hand-to-camera distance

Benefits:
- Models trained on one camera can work on different cameras
- Gestures are recognized regardless of hand size
- More stable feature values for training


==============================================================================
Coordinate Systems
==============================================================================

LANDMARK COORDINATES:
- Input: Normalized coordinates [0, 1] from MediaPipe
- (0, 0) = top-left corner
- (1, 1) = bottom-right corner

DISTANCE FEATURES:
- Unnormalized: Raw Euclidean distances in normalized space
- Normalized: Divided by hand bounding box diagonal (default behavior)
- Range: [0, ~0.7] for normalized distances

ANGLE FEATURES:
- Always in degrees [0, 180]
- Calculated using dot product formula
- 0° = fully flexed, 180° = fully extended


==============================================================================
Performance Considerations
==============================================================================

COMPUTATION TIME:
- Feature extraction: ~5-10 milliseconds per frame
- Suitable for real-time applications (30+ FPS)
- Optimized using NumPy vectorized operations

MEMORY USAGE:
- Feature vector: 46 float32 values = 184 bytes per frame
- Very lightweight for storage and transmission
- Efficient for large-scale data collection

OPTIMIZATION TIPS:
1. Use batch processing for multiple frames
2. Disable normalization if not needed (use_normalize=False)
3. Pre-allocate arrays for repeated feature extraction
4. Use the feature extraction module in parallel pipelines


==============================================================================
Common Gesture Feature Patterns
==============================================================================

PEACE SIGN (Index and Middle extended):
- Index finger: Large angles (~180°), high wrist-to-tip distance
- Middle finger: Large angles (~180°), high wrist-to-tip distance
- Other fingers: Small angles (~90°), low wrist-to-tip distances
- Hand span: Medium (not fully open)

THUMBS UP:
- Thumb: Extended, pointing upward (high Y position)
- Other fingers: Bent (low angles)
- Index/Middle/Ring/Pinky: Clustered together (low MCP distances)

OPEN HAND:
- All fingers: Extended (angles ~180°)
- All wrist-to-tip distances: High
- Hand span: Maximum
- Relative positions: Spread out (high position values)

FIST:
- All fingers: Bent (low angles)
- All wrist-to-tip distances: Low
- Hand span: Minimum
- Inter-joint distances: Low
- Relative positions: Clustered

OK SIGN (Thumb and Index finger meeting):
- Thumb-Index distance: Low
- Other finger distances: Variable
- Index angle: Medium (bent at MCP)
- Other angles: Variable


==============================================================================
Troubleshooting
==============================================================================

ISSUE: Feature values are all zeros
CAUSE: Hand not detected properly
SOLUTION: Check hand detection confidence, ensure good lighting

ISSUE: NaN values in features
CAUSE: Invalid landmarks or zero-length vectors
SOLUTION: The module includes safeguards; check input landmarks are valid

ISSUE: Features are not distinguishing between gestures
CAUSE: Features may not be discriminative for your gestures
SOLUTION: Consider augmenting with additional features or improving data quality

ISSUE: Slow feature extraction
CAUSE: Processing high resolution images
SOLUTION: Reduce input image resolution or use batch processing


==============================================================================
Integration with ML Pipelines
==============================================================================

SKLEARN Pipeline:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])
    
    # Features are already scaled well due to normalization
    pipeline.fit(X_train, y_train)

TENSORFLOW Pipeline:
    import tensorflow as tf
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(46,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_gestures, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

XGBOOST Pipeline:
    import xgboost as xgb
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    model.fit(X_train, y_train)


==============================================================================
API Reference
==============================================================================

HandGestureFeatureExtractor.__init__(normalize=True, fill_value=0.0)
    Initialize the feature extractor
    
    Parameters:
        normalize (bool): If True, normalize distances by hand span
        fill_value (float): Value for missing/invalid features

HandGestureFeatureExtractor.extract(landmarks, return_dict=False)
    Extract features from hand landmarks
    
    Parameters:
        landmarks (np.ndarray): Shape (21, 2), normalized coordinates
        return_dict (bool): If True, return dict with feature groups
    
    Returns:
        If return_dict=False: np.ndarray of shape (46,)
        If return_dict=True: dict with keys 'vector', 'distances', 'angles', 'spans', 'positions'

HandGestureFeatureExtractor.get_feature_names()
    Get descriptive names for all features
    
    Returns:
        list: 46 feature names in order

HandGestureFeatureExtractor._compute_inter_joint_distances(landmarks, hand_span)
    Compute inter-joint distances
    
    Returns:
        np.ndarray of shape (21,)

HandGestureFeatureExtractor._compute_joint_angles(landmarks)
    Compute joint angles
    
    Returns:
        np.ndarray of shape (15,)

HandGestureFeatureExtractor._compute_hand_span_metrics(landmarks)
    Compute hand span metrics
    
    Returns:
        np.ndarray of shape (4,)

HandGestureFeatureExtractor._compute_relative_positions(landmarks)
    Compute relative positions
    
    Returns:
        np.ndarray of shape (6,)


==============================================================================
Version History
==============================================================================

Version 1.0.0 (Initial Release)
- 46-dimensional feature vector
- Scale-invariant distance normalization
- 15 joint angle features
- 4 hand span metrics
- 6 relative position features
- Full documentation and examples
- 30+ unit tests
- Real-time integration examples


==============================================================================
"""

# This documentation file serves as a comprehensive guide for the feature engineering module
print(__doc__)
