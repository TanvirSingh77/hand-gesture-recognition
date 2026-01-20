"""
Feature Extraction Integration Example

Demonstrates how to use the HandGestureFeatureExtractor with the HandLandmarkDetector
for end-to-end feature extraction in a gesture recognition pipeline.

This example shows:
1. Real-time feature extraction from webcam frames
2. Feature visualization and statistics
3. Batch processing of video sequences
4. Exporting features for model training
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Import from project modules
from src.hand_landmarks import HandLandmarkDetector
from src.feature_extractor import HandGestureFeatureExtractor


class RealtimeFeatureExtractor:
    """
    Real-time feature extraction pipeline combining hand detection and feature engineering.
    
    This class integrates the HandLandmarkDetector with HandGestureFeatureExtractor
    to create a complete pipeline for extracting gesture features from video.
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the real-time feature extractor.
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.detector = HandLandmarkDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.extractor = HandGestureFeatureExtractor(normalize=True)
        
        # Statistics tracking
        self.frame_count = 0
        self.detection_count = 0
        self.failed_detections = 0
        self.feature_buffer = []
    
    def process_frame(
        self,
        frame: np.ndarray,
        draw_landmarks: bool = True
    ) -> tuple[np.ndarray | None, dict]:
        """
        Process a single video frame to extract hand gesture features.
        
        Args:
            frame: Input frame as numpy array [H, W, 3] in BGR format
            draw_landmarks: If True, draw hand skeleton on frame
        
        Returns:
            Tuple of (feature_vector, metadata_dict)
            - feature_vector: Array of shape (46,) if hand detected, None otherwise
            - metadata_dict: Contains detection status, handedness, confidence, etc.
        """
        self.frame_count += 1
        
        metadata = {
            'frame_number': self.frame_count,
            'hand_detected': False,
            'handedness': None,
            'feature_vector': None,
            'feature_names': None,
            'hand_span': None,
            'error': None
        }
        
        try:
            # Detect hand landmarks
            landmarks, handedness = self.detector.detect(frame)
            
            if landmarks is None:
                self.failed_detections += 1
                metadata['error'] = 'No hand detected'
                return None, metadata
            
            # Extract features from landmarks
            self.detection_count += 1
            feature_result = self.extractor.extract(landmarks, return_dict=True)
            feature_vector = feature_result['vector']
            
            # Update metadata
            metadata['hand_detected'] = True
            metadata['handedness'] = handedness
            metadata['feature_vector'] = feature_vector
            metadata['feature_names'] = self.extractor.get_feature_names()
            metadata['hand_span'] = feature_result['hand_span']
            
            # Optionally draw landmarks on frame
            if draw_landmarks:
                frame = self._draw_landmarks_with_features(
                    frame,
                    landmarks,
                    feature_result,
                    handedness
                )
            
            # Store in buffer for statistics
            self.feature_buffer.append(feature_vector)
            
            return feature_vector, metadata
        
        except Exception as e:
            self.failed_detections += 1
            metadata['error'] = str(e)
            return None, metadata
    
    def process_video(
        self,
        video_path: str,
        output_path: str | None = None,
        display: bool = True
    ) -> list:
        """
        Process an entire video file and extract features from all frames.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save output video with landmarks drawn
            display: If True, display video with annotations
        
        Returns:
            List of (feature_vector, metadata) tuples for all detected hands
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_results = []
        frame_idx = 0
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Process frame
                features, metadata = self.process_frame(frame, draw_landmarks=True)
                
                # Store results
                all_results.append((features, metadata))
                
                # Write to output video
                if writer:
                    writer.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Feature Extraction', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                if frame_idx % 30 == 0:
                    det_rate = self.detection_count / frame_idx * 100
                    print(f"  Frame {frame_idx}: Detection rate: {det_rate:.1f}%")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        print(f"\nProcessing complete:")
        print(f"  Total frames: {frame_idx}")
        print(f"  Hands detected: {self.detection_count}")
        print(f"  Detection rate: {self.detection_count/frame_idx*100:.1f}%")
        
        return all_results
    
    def process_webcam(self, duration_seconds: int = 30) -> list:
        """
        Process webcam stream and extract features in real-time.
        
        Args:
            duration_seconds: Duration to capture from webcam
        
        Returns:
            List of (feature_vector, metadata) tuples
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Cannot access webcam")
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        all_results = []
        start_time = cv2.getTickCount()
        
        print("Starting webcam capture...")
        print("Press 'q' to quit")
        print("Press 's' to save statistics")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                features, metadata = self.process_frame(frame, draw_landmarks=True)
                all_results.append((features, metadata))
                
                # Add FPS counter
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time) * 1000
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Add statistics
                det_rate = self.detection_count / self.frame_count * 100
                cv2.putText(
                    frame,
                    f"Detections: {self.detection_count}/{self.frame_count} ({det_rate:.1f}%)",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Display
                cv2.imshow('Real-time Feature Extraction', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._print_statistics()
                
                # Check duration
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                if elapsed > duration_seconds:
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"\nCapture complete: {len(all_results)} frames processed")
        return all_results
    
    def _draw_landmarks_with_features(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        features: dict,
        handedness: str
    ) -> np.ndarray:
        """
        Draw hand landmarks and feature information on frame.
        
        Args:
            frame: Input frame to draw on
            landmarks: Hand landmarks (normalized coordinates)
            features: Features dictionary from extractor
            handedness: 'Left' or 'Right'
        
        Returns:
            Frame with drawn landmarks and info
        """
        h, w = frame.shape[:2]
        
        # Convert normalized landmarks to pixel coordinates
        landmarks_px = landmarks * np.array([w, h])
        
        # Draw hand skeleton
        # Connections between joints
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17)
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            start = tuple(landmarks_px[start_idx].astype(int))
            end = tuple(landmarks_px[end_idx].astype(int))
            cv2.line(frame, start, end, (0, 255, 0), 2)
        
        # Draw joints
        for i, (x, y) in enumerate(landmarks_px):
            x, y = int(x), int(y)
            # Different colors for different joint types
            if i == 0:  # Wrist
                color = (255, 0, 0)
                radius = 8
            elif i % 4 == 0:  # Fingertips
                color = (0, 0, 255)
                radius = 6
            else:
                color = (255, 255, 0)
                radius = 4
            
            cv2.circle(frame, (x, y), radius, color, -1)
        
        # Draw information panel
        y_offset = 30
        cv2.putText(frame, f"Hand: {handedness}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Hand Span: {features['hand_span']:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 30
        avg_distance = np.mean(features['distances'])
        cv2.putText(frame, f"Avg Distance: {avg_distance:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 30
        avg_angle = np.mean(features['angles'])
        cv2.putText(frame, f"Avg Angle: {avg_angle:.1f}°", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def _print_statistics(self):
        """Print statistics about feature extraction"""
        print("\n" + "="*60)
        print("Feature Extraction Statistics")
        print("="*60)
        print(f"Frames processed: {self.frame_count}")
        print(f"Hands detected: {self.detection_count}")
        print(f"Detection rate: {self.detection_count/max(self.frame_count, 1)*100:.1f}%")
        print(f"Failed detections: {self.failed_detections}")
        
        if len(self.feature_buffer) > 0:
            features_array = np.array(self.feature_buffer)
            print(f"\nFeature Statistics (over {len(self.feature_buffer)} frames):")
            print(f"  Mean feature vector: {np.mean(features_array, axis=0)[:5]}... (first 5)")
            print(f"  Std feature vector: {np.std(features_array, axis=0)[:5]}... (first 5)")
    
    def export_features(self, output_csv: str):
        """
        Export extracted features to CSV file.
        
        Args:
            output_csv: Path to output CSV file
        """
        if len(self.feature_buffer) == 0:
            print("No features to export")
            return
        
        import csv
        
        features_array = np.array(self.feature_buffer)
        feature_names = self.extractor.get_feature_names()
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame'] + feature_names)
            
            for i, features in enumerate(features_array):
                writer.writerow([i] + features.tolist())
        
        print(f"Exported {len(self.feature_buffer)} feature vectors to {output_csv}")


def main():
    """Example: Real-time feature extraction from webcam"""
    
    # Initialize extractor
    extractor = RealtimeFeatureExtractor(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    print("="*60)
    print("Hand Gesture Feature Extraction - Real-time Demo")
    print("="*60)
    
    # Process webcam for 30 seconds
    results = extractor.process_webcam(duration_seconds=30)
    
    # Print statistics
    extractor._print_statistics()
    
    # Show feature names
    print(f"\nExtracted {len(extractor.get_feature_names())} features:")
    for i, name in enumerate(extractor.get_feature_names(), 1):
        print(f"  {i:2d}. {name}")
    
    # Export features if any were captured
    if len(extractor.feature_buffer) > 0:
        export_path = "features_export.csv"
        extractor.export_features(export_path)


if __name__ == "__main__":
    main()
