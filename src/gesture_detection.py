"""Hand gesture detection using MediaPipe."""

import mediapipe as mp


class GestureDetector:
    """Detects hand landmarks and gestures in video frames."""
    
    def __init__(self):
        """Initialize the gesture detector with MediaPipe."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect(self, frame):
        """Detect hand landmarks in a frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            results: MediaPipe hand detection results
        """
        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        results = self.hands.process(rgb_frame)
        return results
    
    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on the frame.
        
        Args:
            frame: Input video frame
            results: MediaPipe hand detection results
            
        Returns:
            Annotated frame with landmarks drawn
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame
    
    def close(self):
        """Close the detector resources."""
        self.hands.close()
