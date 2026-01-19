
# Camera type groupings for simulator and backend logic
VIOLENCE_CAMERAS = [
    "CAM-042", "CAM-128", "CAM-089", "CAM-156"
]
CRASH_CAMERAS = [
    "CAM-283", "CAM-074", "CAM-195", "CAM-267"
]
PEOPLE_COUNT_CAMERAS = VIOLENCE_CAMERAS  # Or specify a different list if needed
# Global configuration for thresholds and model paths

# Externalized thresholds for tunability
# HIGH ACCURACY MODE: Only trigger on very confident detections
VIOLENCE_THRESHOLD = 0.70  # Higher confidence to reduce false positives
ACCIDENT_THRESHOLD = 0.30  # Increased sensitivity for more crash detections
SMOOTHING_WINDOW = 5

# Default cameras (aligned with dashboard mock metadata)
DEFAULT_CAMERAS = [
    "CAM-042", "CAM-128", "CAM-089", "CAM-156",
    "CAM-283", "CAM-074", "CAM-195", "CAM-267",
    "CAM-341", "CAM-412", "CAM-523", "CAM-604"
]





# Model identification
MODEL_NAME = "Vigil-MobileNetClip-v1"


MODEL_PATHS = {
    "mobilenet_clip": "backend/models/mobilenet_clip_best_ts.pt",  # Violence
    "crash_lstm": "backend/models/mobilenetv2_lstm_finetuned.pt",  # Crash
    "people_counter": "backend/models/yolov8n.pt",                 # People counting
    "x3d_s": "backend/models/x3d_s_best.pth",
}

DATA_PATHS = {
    "demo_video": "demo.mp4"
}
