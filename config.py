"""Configuration settings for the gesture recognition system."""

# Camera settings
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Detection settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_HANDS = 2

# Display settings
SHOW_FPS = True
SHOW_LANDMARKS = True
WINDOW_NAME = "Hand Gesture Recognition"

# Model settings
MODEL_PATH = "./models/gesture_model.pkl"

# Logging
DEBUG = False
