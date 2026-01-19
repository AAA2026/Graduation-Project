"""
Backend AI Interface - Unified Entry Point for All Detection Models

This module provides a unified interface for:

Flask routes should import from here to access all AI models.
"""

from __future__ import annotations
from typing import Dict

# Import specialized detectors







print("[DEBUG] Importing violence, crash, and people counting models...")
from backend.ai.violence_detector import detect_violence
from backend.ai.accident_model import detect_crash
from backend.ai.people_counter.yolov8 import detect_people_count
print("[DEBUG] Model imports complete.")

def run_inference(video_path: str, camera_id: str = None) -> dict:
	"""
	Unified inference entry: runs violence, crash, and people counting as needed.
	Returns dict with event, confidence, model, latency, timestamp, and people count if available.
	"""
	result = {}
	violence_result = None
	crash_result = None
	people_result = None
	print(f"[DEBUG] run_inference called: video_path={video_path}, camera_id={camera_id}")
	if camera_id:
		from backend.config import VIOLENCE_CAMERAS, CRASH_CAMERAS, PEOPLE_COUNT_CAMERAS
		if camera_id in VIOLENCE_CAMERAS:
			print(f"[DEBUG] Running violence detection for {camera_id}")
			violence_result = detect_violence(video_path)
			if camera_id in PEOPLE_COUNT_CAMERAS:
				print(f"[DEBUG] Running people counting for {camera_id}")
				people_result = detect_people_count(video_path)
			result = violence_result or {}
			if people_result:
				result['people_count'] = people_result.get('count', 0)
				result['people_confidence'] = people_result.get('confidence', 0)
		elif camera_id in CRASH_CAMERAS:
			print(f"[DEBUG] Running crash detection for {camera_id}")
			crash_result = detect_crash(video_path)
			result = crash_result or {}
		else:
			print(f"[DEBUG] Running default violence detection for {camera_id}")
			violence_result = detect_violence(video_path)
			result = violence_result or {}
	else:
		print(f"[DEBUG] Running violence detection (no camera_id)")
		violence_result = detect_violence(video_path)
		result = violence_result or {}
	print(f"[DEBUG] Inference result: {result}")
	return result


