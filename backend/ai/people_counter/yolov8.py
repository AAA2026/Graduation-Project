"""
YOLOv8 People Counter Integration

This module loads a YOLOv8 model and counts people in a video.
Requires ultralytics package: pip install ultralytics
"""
import time
from pathlib import Path
from ultralytics import YOLO
import cv2

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "yolov8n.pt"

# Global model cache
_yolo_model = None

def load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(MODEL_PATH))
    return _yolo_model

def detect_people_count(video_path: str) -> dict:
    model = load_yolo_model()
    cap = cv2.VideoCapture(video_path)
    people_count = 0
    frame_count = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        # Count people (class 0 in COCO)
        n_people = sum(int(cls == 0) for cls in results[0].boxes.cls.cpu().numpy())
        people_count += n_people
        frame_count += 1
        if frame_count >= 16:
            break
    cap.release()
    avg_count = int(round(people_count / frame_count)) if frame_count else 0
    latency = int((time.time() - start) * 1000)
    return {
        "count": avg_count,
        "confidence": 0.95,  # Placeholder, can be improved
        "model": "yolov8n.pt",
        "latency_ms": latency,
        "timestamp": time.time()
    }
