"""
People Counting Service - Placeholder

This module simulates a people counting model for demo purposes.
Replace with real model integration as needed.
"""
import random
import time

def detect_people_count(video_path: str) -> dict:
    """
    Simulate people counting on a video clip.
    Replace with real model inference.
    Returns:
        dict: {
            "count": int,
            "confidence": float,
            "model": str,
            "latency_ms": int,
            "timestamp": float
        }
    """
    # Simulate detection
    count = random.randint(0, 25)
    confidence = random.uniform(0.80, 0.99)
    latency = random.randint(40, 120)
    return {
        "count": count,
        "confidence": confidence,
        "model": "demo-people-counter",
        "latency_ms": latency,
        "timestamp": time.time()
    }
