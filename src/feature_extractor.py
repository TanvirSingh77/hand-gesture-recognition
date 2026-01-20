"""
Hand Gesture Feature Extractor Module

This module extracts meaningful features from raw hand landmarks for gesture recognition.
It computes:
1. Normalized distances between key joints (inter-joint distances)
2. Angles between finger segments (joint angles)
3. Relative hand metrics (hand span, finger lengths)
4. Returns a flattened numerical feature vector for ML models

The feature vector is consistent and can be used for both training and inference.

Example:
    from src.hand_landmarks import HandLandmarkDetector
    from src.feature_extractor import HandGestureFeatureExtractor
    
    detector = HandLandmarkDetector()
    extractor = HandGestureFeatureExtractor()
    
    frame = capture_frame()  # numpy array [H, W, 3]
    landmarks, handedness = detector.detect(frame)
    
    features = extractor.extract(landmarks)  # 1D array of features
    print(f"Feature vector shape: {features.shape}")
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings


class HandGestureFeatureExtractor:
    """
    Extracts and computes features from hand landmarks for gesture recognition.
    
    Features computed:
    1. Inter-joint distances (21 distances): Normalized Euclidean distances between key joints
    2. Joint angles (15 angles): Angles between finger segments (thumb, index, middle, ring, pinky)
    3. Hand span metrics (4 values): Overall hand measurements
    4. Relative positions (6 values): Position ratios relative to hand bounding box
    
    Total feature vector: 46 features (21 distances + 15 angles + 4 span + 6 positions)
    
    All distances are normalized by hand bounding box diagonal to be scale-invariant.
    All angles are in degrees (0-180).
    """
    
    def __init__(self, normalize: bool = True, fill_value: float = 0.0):
        """
        Initialize the feature extractor.
        
        Args:
            normalize: If True, normalize distances by hand span for scale invariance
            fill_value: Value to use if hand not detected (default: 0.0)
        """
        self.normalize = normalize
        self.fill_value = fill_value
        
        # Hand landmark indices (from MediaPipe 21-point hand model)
        # Wrist
        self.WRIST = 0
        
        # Thumb: CMC(1), MCP(2), IP(3), TIP(4)
        self.THUMB_CMC = 1
        self.THUMB_MCP = 2
        self.THUMB_IP = 3
        self.THUMB_TIP = 4
        
        # Index: MCP(5), PIP(6), DIP(7), TIP(8)
        self.INDEX_MCP = 5
        self.INDEX_PIP = 6
        self.INDEX_DIP = 7
        self.INDEX_TIP = 8
        
        # Middle: MCP(9), PIP(10), DIP(11), TIP(12)
        self.MIDDLE_MCP = 9
        self.MIDDLE_PIP = 10
        self.MIDDLE_DIP = 11
        self.MIDDLE_TIP = 12
        
        # Ring: MCP(13), PIP(14), DIP(15), TIP(16)
        self.RING_MCP = 13
        self.RING_PIP = 14
        self.RING_DIP = 15
        self.RING_TIP = 16
        
        # Pinky: MCP(17), PIP(18), DIP(19), TIP(20)
        self.PINKY_MCP = 17
        self.PINKY_PIP = 18
        self.PINKY_DIP = 19
        self.PINKY_TIP = 20
        
        # Feature indices for easy reference
        self.DISTANCE_FEATURES_START = 0
        self.DISTANCE_FEATURES_END = 21
        self.ANGLE_FEATURES_START = 21
        self.ANGLE_FEATURES_END = 36
        self.SPAN_FEATURES_START = 36
        self.SPAN_FEATURES_END = 40
        self.POSITION_FEATURES_START = 40
        self.POSITION_FEATURES_END = 46
    
    def extract(
        self,
        landmarks: np.ndarray,
        return_dict: bool = False
    ) -> np.ndarray | dict:
        """
        Extract features from hand landmarks.
        
        Args:
            landmarks: Hand landmarks array of shape (21, 2) with normalized coordinates [0, 1]
            return_dict: If True, return dict with feature names and values
        
        Returns:
            If return_dict=False: Feature vector of shape (46,) as numpy array
            If return_dict=True: Dict with keys 'vector', 'distances', 'angles', 'spans', 'positions'
        
        Raises:
            ValueError: If landmarks shape is not (21, 2)
        """
        if landmarks.shape != (21, 2):
            raise ValueError(f"Expected landmarks shape (21, 2), got {landmarks.shape}")
        
        # Compute hand span (normalization factor)
        hand_span = self._compute_hand_span(landmarks)
        
        # Extract feature groups
        distances = self._compute_inter_joint_distances(landmarks, hand_span)
        angles = self._compute_joint_angles(landmarks)
        spans = self._compute_hand_span_metrics(landmarks)
        positions = self._compute_relative_positions(landmarks)
        
        # Concatenate all features into single vector
        feature_vector = np.concatenate([distances, angles, spans, positions])
        
        if return_dict:
            return {
                'vector': feature_vector,
                'distances': distances,
                'angles': angles,
                'spans': spans,
                'positions': positions,
                'hand_span': hand_span
            }
        
        return feature_vector
    
    def _compute_hand_span(self, landmarks: np.ndarray) -> float:
        """
        Compute the hand bounding box diagonal as a normalization factor.
        
        This makes distances scale-invariant - important for detecting the same gesture
        at different distances from the camera.
        
        Returns:
            Diagonal length of hand bounding box in normalized coordinates
        """
        x_min, x_max = landmarks[:, 0].min(), landmarks[:, 0].max()
        y_min, y_max = landmarks[:, 1].min(), landmarks[:, 1].max()
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Avoid division by zero
        span = np.sqrt(width**2 + height**2)
        return span if span > 1e-6 else 1.0
    
    def _compute_inter_joint_distances(
        self,
        landmarks: np.ndarray,
        hand_span: float
    ) -> np.ndarray:
        """
        Compute normalized distances between key joints.
        
        These distances capture the overall geometry and spread of the hand.
        Normalized by hand_span to be scale-invariant.
        
        Distances computed (21 features):
        1. Wrist to each fingertip (5): Captures hand opening/closing
        2. Each finger's inter-joint distances (12): Captures finger bending
        3. Distances between MCP joints (4): Captures finger spread
        
        Returns:
            Array of shape (21,) with normalized distances
        """
        distances = []
        norm_factor = hand_span if self.normalize else 1.0
        
        # 1. Wrist to each fingertip (5 distances)
        # These indicate how "open" each finger is
        for tip_idx in [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, 
                        self.RING_TIP, self.PINKY_TIP]:
            dist = self._euclidean_distance(
                landmarks[self.WRIST],
                landmarks[tip_idx]
            ) / norm_factor
            distances.append(dist)
        
        # 2. Thumb inter-joint distances (3 distances)
        # CMC->MCP, MCP->IP, IP->TIP
        for j in range(3):
            joint_a = self.THUMB_CMC + j
            joint_b = self.THUMB_CMC + j + 1
            dist = self._euclidean_distance(
                landmarks[joint_a],
                landmarks[joint_b]
            ) / norm_factor
            distances.append(dist)
        
        # 3. Index finger inter-joint distances (3 distances)
        # MCP->PIP, PIP->DIP, DIP->TIP
        for j in range(3):
            joint_a = self.INDEX_MCP + j
            joint_b = self.INDEX_MCP + j + 1
            dist = self._euclidean_distance(
                landmarks[joint_a],
                landmarks[joint_b]
            ) / norm_factor
            distances.append(dist)
        
        # 4. Middle finger inter-joint distances (3 distances)
        for j in range(3):
            joint_a = self.MIDDLE_MCP + j
            joint_b = self.MIDDLE_MCP + j + 1
            dist = self._euclidean_distance(
                landmarks[joint_a],
                landmarks[joint_b]
            ) / norm_factor
            distances.append(dist)
        
        # 5. Ring finger inter-joint distances (3 distances)
        for j in range(3):
            joint_a = self.RING_MCP + j
            joint_b = self.RING_MCP + j + 1
            dist = self._euclidean_distance(
                landmarks[joint_a],
                landmarks[joint_b]
            ) / norm_factor
            distances.append(dist)
        
        # 6. Pinky finger inter-joint distances (3 distances)
        for j in range(3):
            joint_a = self.PINKY_MCP + j
            joint_b = self.PINKY_MCP + j + 1
            dist = self._euclidean_distance(
                landmarks[joint_a],
                landmarks[joint_b]
            ) / norm_factor
            distances.append(dist)
        
        # 7. Distances between MCP joints (4 distances)
        # Thumb-Index, Index-Middle, Middle-Ring, Ring-Pinky
        # These capture finger spread and hand width
        mcp_joints = [
            self.THUMB_MCP, self.INDEX_MCP, self.MIDDLE_MCP,
            self.RING_MCP, self.PINKY_MCP
        ]
        for j in range(len(mcp_joints) - 1):
            dist = self._euclidean_distance(
                landmarks[mcp_joints[j]],
                landmarks[mcp_joints[j + 1]]
            ) / norm_factor
            distances.append(dist)
        
        return np.array(distances, dtype=np.float32)
    
    def _compute_joint_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute angles between finger segments.
        
        These angles capture the bending/flexion state of each finger.
        Important for distinguishing gestures like "peace" (fingers extended)
        from "fist" (fingers bent).
        
        Angles computed (15 features):
        1. Thumb angles (3): CMC-MCP-IP, MCP-IP-TIP, CMC-MCP-TIP
        2. Index angles (3): MCP-PIP-DIP, PIP-DIP-TIP, MCP-PIP-TIP
        3. Middle angles (3): MCP-PIP-DIP, PIP-DIP-TIP, MCP-PIP-TIP
        4. Ring angles (3): MCP-PIP-DIP, PIP-DIP-TIP, MCP-PIP-TIP
        5. Pinky angles (3): MCP-PIP-DIP, PIP-DIP-TIP, MCP-PIP-TIP
        
        Returns:
            Array of shape (15,) with angles in degrees [0, 180]
        """
        angles = []
        
        # Thumb angles (3)
        # Angle at MCP joint: CMC-MCP-IP
        angles.append(self._compute_angle(
            landmarks[self.THUMB_CMC],
            landmarks[self.THUMB_MCP],
            landmarks[self.THUMB_IP]
        ))
        # Angle at IP joint: MCP-IP-TIP
        angles.append(self._compute_angle(
            landmarks[self.THUMB_MCP],
            landmarks[self.THUMB_IP],
            landmarks[self.THUMB_TIP]
        ))
        # Angle at MCP to TIP: CMC-MCP-TIP
        angles.append(self._compute_angle(
            landmarks[self.THUMB_CMC],
            landmarks[self.THUMB_MCP],
            landmarks[self.THUMB_TIP]
        ))
        
        # Index finger angles (3)
        # Angle at PIP joint: MCP-PIP-DIP
        angles.append(self._compute_angle(
            landmarks[self.INDEX_MCP],
            landmarks[self.INDEX_PIP],
            landmarks[self.INDEX_DIP]
        ))
        # Angle at DIP joint: PIP-DIP-TIP
        angles.append(self._compute_angle(
            landmarks[self.INDEX_PIP],
            landmarks[self.INDEX_DIP],
            landmarks[self.INDEX_TIP]
        ))
        # Angle at MCP to TIP: MCP-PIP-TIP
        angles.append(self._compute_angle(
            landmarks[self.INDEX_MCP],
            landmarks[self.INDEX_PIP],
            landmarks[self.INDEX_TIP]
        ))
        
        # Middle finger angles (3)
        angles.append(self._compute_angle(
            landmarks[self.MIDDLE_MCP],
            landmarks[self.MIDDLE_PIP],
            landmarks[self.MIDDLE_DIP]
        ))
        angles.append(self._compute_angle(
            landmarks[self.MIDDLE_PIP],
            landmarks[self.MIDDLE_DIP],
            landmarks[self.MIDDLE_TIP]
        ))
        angles.append(self._compute_angle(
            landmarks[self.MIDDLE_MCP],
            landmarks[self.MIDDLE_PIP],
            landmarks[self.MIDDLE_TIP]
        ))
        
        # Ring finger angles (3)
        angles.append(self._compute_angle(
            landmarks[self.RING_MCP],
            landmarks[self.RING_PIP],
            landmarks[self.RING_DIP]
        ))
        angles.append(self._compute_angle(
            landmarks[self.RING_PIP],
            landmarks[self.RING_DIP],
            landmarks[self.RING_TIP]
        ))
        angles.append(self._compute_angle(
            landmarks[self.RING_MCP],
            landmarks[self.RING_PIP],
            landmarks[self.RING_TIP]
        ))
        
        # Pinky finger angles (3)
        angles.append(self._compute_angle(
            landmarks[self.PINKY_MCP],
            landmarks[self.PINKY_PIP],
            landmarks[self.PINKY_DIP]
        ))
        angles.append(self._compute_angle(
            landmarks[self.PINKY_PIP],
            landmarks[self.PINKY_DIP],
            landmarks[self.PINKY_TIP]
        ))
        angles.append(self._compute_angle(
            landmarks[self.PINKY_MCP],
            landmarks[self.PINKY_PIP],
            landmarks[self.PINKY_TIP]
        ))
        
        return np.array(angles, dtype=np.float32)
    
    def _compute_hand_span_metrics(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute overall hand geometry metrics.
        
        These features capture the overall size and aspect ratio of the hand.
        Useful for distinguishing gestures based on overall hand shape.
        
        Metrics (4 features):
        1. Hand bounding box width
        2. Hand bounding box height
        3. Hand bounding box aspect ratio (width/height)
        4. Wrist to furthest fingertip distance
        
        Returns:
            Array of shape (4,) with hand span metrics
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        width = x_coords.max() - x_coords.min()
        height = y_coords.max() - y_coords.min()
        
        # Aspect ratio with safeguard against division by zero
        aspect_ratio = width / (height + 1e-6)
        
        # Maximum reach from wrist
        wrist = landmarks[self.WRIST]
        distances_from_wrist = np.linalg.norm(landmarks - wrist, axis=1)
        max_reach = distances_from_wrist.max()
        
        return np.array([width, height, aspect_ratio, max_reach], dtype=np.float32)
    
    def _compute_relative_positions(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute relative positions of key joints within the hand bounding box.
        
        These features capture the relative positioning of fingers,
        useful for distinguishing fine-grained gesture differences.
        
        Positions (6 features):
        1. Wrist position (relative Y within bounding box)
        2. Thumb tip relative position (X within bbox)
        3. Thumb tip relative position (Y within bbox)
        4. Index tip relative position (X within bbox)
        5. Index tip relative position (Y within bbox)
        6. Pinky tip relative position (X within bbox)
        
        Returns:
            Array of shape (6,) with relative positions [0, 1]
        """
        x_min, x_max = landmarks[:, 0].min(), landmarks[:, 0].max()
        y_min, y_max = landmarks[:, 1].min(), landmarks[:, 1].max()
        
        width = x_max - x_min if x_max > x_min else 1.0
        height = y_max - y_min if y_max > y_min else 1.0
        
        positions = []
        
        # Wrist Y position (how low in the hand)
        wrist_y_rel = (landmarks[self.WRIST, 1] - y_min) / height
        positions.append(wrist_y_rel)
        
        # Thumb tip position
        thumb_x_rel = (landmarks[self.THUMB_TIP, 0] - x_min) / width
        thumb_y_rel = (landmarks[self.THUMB_TIP, 1] - y_min) / height
        positions.append(thumb_x_rel)
        positions.append(thumb_y_rel)
        
        # Index tip position
        index_x_rel = (landmarks[self.INDEX_TIP, 0] - x_min) / width
        index_y_rel = (landmarks[self.INDEX_TIP, 1] - y_min) / height
        positions.append(index_x_rel)
        positions.append(index_y_rel)
        
        # Pinky tip position
        pinky_x_rel = (landmarks[self.PINKY_TIP, 0] - x_min) / width
        positions.append(pinky_x_rel)
        
        return np.array(positions, dtype=np.float32)
    
    @staticmethod
    def _euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two 2D points.
        
        Args:
            point1: Point as [x, y]
            point2: Point as [x, y]
        
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(point1 - point2))
    
    @staticmethod
    def _compute_angle(
        point_a: np.ndarray,
        point_vertex: np.ndarray,
        point_c: np.ndarray
    ) -> float:
        """
        Compute angle at vertex formed by three points.
        
        Angle is computed using the dot product formula:
        angle = arccos((BA · BC) / (|BA| * |BC|))
        
        where BA and BC are vectors from vertex to A and C respectively.
        
        Args:
            point_a: First point [x, y]
            point_vertex: Vertex point [x, y]
            point_c: Third point [x, y]
        
        Returns:
            Angle in degrees [0, 180]
        """
        # Vectors from vertex to the two other points
        vector_a = point_a - point_vertex
        vector_c = point_c - point_vertex
        
        # Compute magnitudes
        magnitude_a = np.linalg.norm(vector_a)
        magnitude_c = np.linalg.norm(vector_c)
        
        # Avoid division by zero
        if magnitude_a < 1e-6 or magnitude_c < 1e-6:
            return 0.0
        
        # Compute cosine of angle using dot product
        cos_angle = np.dot(vector_a, vector_c) / (magnitude_a * magnitude_c)
        
        # Clamp to [-1, 1] to avoid numerical errors in arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert from radians to degrees
        angle_radians = np.arccos(cos_angle)
        angle_degrees = np.degrees(angle_radians)
        
        return float(angle_degrees)
    
    def get_feature_names(self) -> List[str]:
        """
        Get descriptive names for all features in the feature vector.
        
        Returns:
            List of 46 feature names in the order they appear in the feature vector
        """
        names = []
        
        # Inter-joint distances
        names.extend([
            "wrist_to_thumb_tip",
            "wrist_to_index_tip",
            "wrist_to_middle_tip",
            "wrist_to_ring_tip",
            "wrist_to_pinky_tip",
            "thumb_cmc_mcp_dist",
            "thumb_mcp_ip_dist",
            "thumb_ip_tip_dist",
            "index_mcp_pip_dist",
            "index_pip_dip_dist",
            "index_dip_tip_dist",
            "middle_mcp_pip_dist",
            "middle_pip_dip_dist",
            "middle_dip_tip_dist",
            "ring_mcp_pip_dist",
            "ring_pip_dip_dist",
            "ring_dip_tip_dist",
            "pinky_mcp_pip_dist",
            "pinky_pip_dip_dist",
            "pinky_dip_tip_dist",
            "thumb_index_mcp_dist",
            "index_middle_mcp_dist",
            "middle_ring_mcp_dist",
            "ring_pinky_mcp_dist",
        ])
        
        # Joint angles
        names.extend([
            "thumb_mcp_angle",
            "thumb_ip_angle",
            "thumb_overall_angle",
            "index_pip_angle",
            "index_dip_angle",
            "index_overall_angle",
            "middle_pip_angle",
            "middle_dip_angle",
            "middle_overall_angle",
            "ring_pip_angle",
            "ring_dip_angle",
            "ring_overall_angle",
            "pinky_pip_angle",
            "pinky_dip_angle",
            "pinky_overall_angle",
        ])
        
        # Hand span metrics
        names.extend([
            "hand_bbox_width",
            "hand_bbox_height",
            "hand_aspect_ratio",
            "max_reach_from_wrist",
        ])
        
        # Relative positions
        names.extend([
            "wrist_relative_y",
            "thumb_tip_relative_x",
            "thumb_tip_relative_y",
            "index_tip_relative_x",
            "index_tip_relative_y",
            "pinky_tip_relative_x",
        ])
        
        return names


if __name__ == "__main__":
    """
    Demo: Extract features from random landmarks
    """
    # Create random landmarks (for demo only)
    random_landmarks = np.random.rand(21, 2)
    
    # Initialize extractor
    extractor = HandGestureFeatureExtractor(normalize=True)
    
    # Extract features
    features = extractor.extract(random_landmarks, return_dict=True)
    
    print("="*60)
    print("Hand Gesture Feature Extraction Demo")
    print("="*60)
    print(f"\nFeature vector shape: {features['vector'].shape}")
    print(f"Hand span: {features['hand_span']:.4f}")
    print(f"\nFeature groups:")
    print(f"  - Inter-joint distances: {features['distances'].shape[0]} features")
    print(f"  - Joint angles: {features['angles'].shape[0]} features")
    print(f"  - Hand span metrics: {features['spans'].shape[0]} features")
    print(f"  - Relative positions: {features['positions'].shape[0]} features")
    
    # Print feature names and values
    print(f"\nDetailed feature breakdown:")
    feature_names = extractor.get_feature_names()
    feature_vector = features['vector']
    
    print(f"\nInter-joint Distances (normalized):")
    for i, (name, value) in enumerate(zip(feature_names[:21], feature_vector[:21])):
        print(f"  {i+1:2d}. {name:30s}: {value:7.4f}")
    
    print(f"\nJoint Angles (degrees):")
    for i, (name, value) in enumerate(zip(feature_names[21:36], feature_vector[21:36]), 1):
        print(f"  {i:2d}. {name:30s}: {value:7.2f}°")
    
    print(f"\nHand Span Metrics:")
    for i, (name, value) in enumerate(zip(feature_names[36:40], feature_vector[36:40]), 1):
        print(f"  {i:2d}. {name:30s}: {value:7.4f}")
    
    print(f"\nRelative Positions:")
    for i, (name, value) in enumerate(zip(feature_names[40:46], feature_vector[40:46]), 1):
        print(f"  {i:2d}. {name:30s}: {value:7.4f}")
