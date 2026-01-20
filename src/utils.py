"""Utility functions for the gesture recognition system."""

import cv2
import numpy as np


def draw_text(frame, text, position, font_size=1.0, color=(0, 255, 0)):
    """Draw text on a frame.
    
    Args:
        frame: Input video frame
        text: Text to display
        position: (x, y) position for text
        font_size: Font scale
        color: BGR color tuple
        
    Returns:
        Frame with text drawn
    """
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size, color, 2
    )
    return frame


def draw_fps(frame, fps, position=(30, 30)):
    """Draw FPS counter on frame.
    
    Args:
        frame: Input video frame
        fps: Frames per second value
        position: Position to draw FPS
        
    Returns:
        Frame with FPS drawn
    """
    text = f"FPS: {fps:.1f}"
    return draw_text(frame, text, position)


def extract_landmarks_array(hand_landmarks):
    """Convert MediaPipe landmarks to numpy array.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        
    Returns:
        Numpy array of landmark coordinates
    """
    if hand_landmarks is None:
        return None
    
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    return np.array(landmarks)


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance value
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
