"""Utilities for managing collected gesture data.

This module provides functions for:
- Loading collected gesture data
- Converting data formats
- Analyzing data statistics
- Preparing data for training
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv


class GestureDataLoader:
    """Load and manage collected gesture data."""
    
    def __init__(self, data_dir: str = "data/collected_gestures"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing collected gesture data
        """
        self.data_dir = Path(data_dir)
        self.gestures = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all gesture samples from disk."""
        for gesture_dir in self.data_dir.iterdir():
            if gesture_dir.is_dir():
                gesture_name = gesture_dir.name
                samples = []
                
                # Load all sample files for this gesture
                for sample_file in sorted(gesture_dir.glob("sample_*.json")):
                    try:
                        with open(sample_file, 'r') as f:
                            sample_data = json.load(f)
                            samples.append(sample_data)
                    except Exception as e:
                        print(f"Error loading {sample_file}: {e}")
                
                if samples:
                    self.gestures[gesture_name] = samples
    
    def get_gesture_samples(self, gesture_name: str) -> List[dict]:
        """Get all samples for a specific gesture.
        
        Args:
            gesture_name: Name of the gesture
        
        Returns:
            List of sample dictionaries
        """
        return self.gestures.get(gesture_name, [])
    
    def get_all_gestures(self) -> Dict[str, List[dict]]:
        """Get all loaded gestures.
        
        Returns:
            Dictionary mapping gesture names to sample lists
        """
        return self.gestures.copy()
    
    def get_statistics(self) -> dict:
        """Get statistics about loaded data.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": sum(len(samples) for samples in self.gestures.values()),
            "gesture_count": len(self.gestures),
            "gestures": {}
        }
        
        for gesture_name, samples in self.gestures.items():
            gesture_stats = {
                "sample_count": len(samples),
                "total_frames": sum(s.get("frame_count", 0) for s in samples),
                "avg_frames": np.mean([s.get("frame_count", 0) for s in samples])
            }
            stats["gestures"][gesture_name] = gesture_stats
        
        return stats
    
    def export_to_csv(self, output_file: str, include_z: bool = False):
        """Export landmark data to CSV format.
        
        Args:
            output_file: Path to output CSV file
            include_z: Whether to include z-coordinates
        
        This creates a CSV where each row is a gesture sample, with columns for:
        - gesture (class label)
        - Each landmark coordinate (x, y[, z])
        """
        rows = []
        
        for gesture_name, samples in self.gestures.items():
            for sample_idx, sample in enumerate(samples):
                frames = sample.get("frames", [])
                
                # Aggregate landmarks across frames (e.g., take mean or first frame)
                if frames:
                    first_frame = frames[0]
                    landmarks = np.array(first_frame["landmarks"])
                    
                    # Flatten landmarks
                    row = {"gesture": gesture_name, "sample_id": sample_idx}
                    
                    for lm_idx, (x, y) in enumerate(landmarks):
                        row[f"lm_{lm_idx}_x"] = x
                        row[f"lm_{lm_idx}_y"] = y
                    
                    rows.append(row)
        
        if rows:
            # Write CSV
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"Exported {len(rows)} samples to {output_file}")
    
    def export_to_numpy(self, output_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Export landmark data to NumPy format.
        
        Args:
            output_file: Path to output .npz file
        
        Returns:
            Tuple of (features, labels)
            - features: (n_samples, 42) array of normalized landmarks
            - labels: (n_samples,) array of gesture class labels
        """
        features = []
        labels = []
        
        gesture_to_id = {name: idx for idx, name in enumerate(self.gestures.keys())}
        
        for gesture_name, samples in self.gestures.items():
            gesture_id = gesture_to_id[gesture_name]
            
            for sample in samples:
                frames = sample.get("frames", [])
                
                if frames:
                    # Use first frame's landmarks
                    landmarks = np.array(frames[0]["landmarks"], dtype=np.float32)
                    features.append(landmarks.flatten())
                    labels.append(gesture_id)
        
        if features:
            X = np.array(features, dtype=np.float32)
            y = np.array(labels, dtype=np.int32)
            
            # Save with metadata
            np.savez(
                output_file,
                X=X,
                y=y,
                gesture_names=list(gesture_to_id.keys()),
                gesture_ids=list(gesture_to_id.values())
            )
            
            print(f"Exported {len(X)} samples to {output_file}")
            return X, y
        
        return None, None
    
    def get_feature_vectors(self, 
                           aggregate_frames: str = "first") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get feature vectors for all samples.
        
        Args:
            aggregate_frames: How to aggregate multiple frames per sample
                - "first": Use first frame only
                - "mean": Average across all frames
                - "flatten": Flatten all frames into one vector
        
        Returns:
            Tuple of (X, y, gesture_names)
            - X: Feature matrix (n_samples, n_features)
            - y: Label vector (n_samples,)
            - gesture_names: List of gesture class names
        """
        gesture_names = sorted(self.gestures.keys())
        gesture_to_id = {name: idx for idx, name in enumerate(gesture_names)}
        
        X = []
        y = []
        
        for gesture_name, samples in self.gestures.items():
            gesture_id = gesture_to_id[gesture_name]
            
            for sample in samples:
                frames = sample.get("frames", [])
                
                if not frames:
                    continue
                
                if aggregate_frames == "first":
                    landmarks = np.array(frames[0]["landmarks"], dtype=np.float32)
                    feature_vector = landmarks.flatten()
                
                elif aggregate_frames == "mean":
                    all_landmarks = []
                    for frame in frames:
                        landmarks = np.array(frame["landmarks"], dtype=np.float32)
                        all_landmarks.append(landmarks)
                    
                    feature_vector = np.mean(all_landmarks, axis=0).flatten()
                
                elif aggregate_frames == "flatten":
                    all_landmarks = []
                    for frame in frames:
                        landmarks = np.array(frame["landmarks"], dtype=np.float32)
                        all_landmarks.append(landmarks.flatten())
                    
                    feature_vector = np.concatenate(all_landmarks)
                
                else:
                    raise ValueError(f"Unknown aggregate method: {aggregate_frames}")
                
                X.append(feature_vector)
                y.append(gesture_id)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), gesture_names
    
    def print_statistics(self):
        """Print data statistics to console."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Gesture Data Statistics")
        print("="*60)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Total gestures: {stats['gesture_count']}")
        print("\nBreakdown by gesture:")
        print("-"*60)
        
        for gesture_name, gesture_stats in sorted(stats['gestures'].items()):
            display_name = gesture_name.replace("_", " ").title()
            print(f"  {display_name}:")
            print(f"    Samples: {gesture_stats['sample_count']}")
            print(f"    Total frames: {gesture_stats['total_frames']}")
            print(f"    Avg frames per sample: {gesture_stats['avg_frames']:.1f}")
        
        print("="*60 + "\n")


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize landmarks to [-1, 1] range.
    
    Args:
        landmarks: Input landmarks (21, 2) with values in [0, 1]
    
    Returns:
        Normalized landmarks in [-1, 1]
    """
    return landmarks * 2.0 - 1.0


def standardize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Standardize landmarks (zero mean, unit variance).
    
    Args:
        landmarks: Input landmarks (21, 2)
    
    Returns:
        Standardized landmarks
    """
    mean = np.mean(landmarks, axis=0, keepdims=True)
    std = np.std(landmarks, axis=0, keepdims=True)
    std[std == 0] = 1  # Avoid division by zero
    
    return (landmarks - mean) / std


def augment_landmarks(landmarks: np.ndarray, 
                      flip: bool = False,
                      rotate: bool = False,
                      scale: bool = False) -> List[np.ndarray]:
    """Generate augmented landmark variations.
    
    Args:
        landmarks: Input landmarks (21, 2)
        flip: Whether to include horizontal flip
        rotate: Whether to include rotations
        scale: Whether to include scaling
    
    Returns:
        List of augmented landmark arrays
    """
    augmented = [landmarks.copy()]
    
    if flip:
        # Horizontal flip
        flipped = landmarks.copy()
        flipped[:, 0] = 1.0 - flipped[:, 0]
        augmented.append(flipped)
    
    if rotate:
        # Rotate by small angles
        for angle in [-10, 10]:
            angle_rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Center at origin
            centered = landmarks - 0.5
            
            # Rotate
            rotated = np.zeros_like(centered)
            rotated[:, 0] = centered[:, 0] * cos_a - centered[:, 1] * sin_a
            rotated[:, 1] = centered[:, 0] * sin_a + centered[:, 1] * cos_a
            
            # Shift back and clip
            rotated = np.clip(rotated + 0.5, 0, 1)
            augmented.append(rotated)
    
    if scale:
        # Scale by small factors
        for scale_factor in [0.9, 1.1]:
            centered = landmarks - 0.5
            scaled = centered * scale_factor
            scaled = np.clip(scaled + 0.5, 0, 1)
            augmented.append(scaled)
    
    return augmented


if __name__ == "__main__":
    # Example usage
    loader = GestureDataLoader()
    loader.print_statistics()
    
    # Export to different formats
    loader.export_to_csv("data/landmarks.csv")
    loader.export_to_numpy("data/landmarks.npz")
    
    # Get feature vectors
    X, y, gesture_names = loader.get_feature_vectors(aggregate_frames="first")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    print(f"Gesture classes: {gesture_names}")
