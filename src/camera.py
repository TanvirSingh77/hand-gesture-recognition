"""Camera module for capturing video frames in real-time."""

import cv2


class CameraCapture:
    """Handles camera input and frame capture."""
    
    def __init__(self, camera_id=0):
        """Initialize camera capture.
        
        Args:
            camera_id: Index of the camera device (default: 0)
        """
        self.camera_id = camera_id
        self.cap = None
        
    def start(self):
        """Start the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
    def get_frame(self):
        """Get the current frame from camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start() first.")
        return self.cap.read()
    
    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
