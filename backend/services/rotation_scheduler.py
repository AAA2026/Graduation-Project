"""
Background Video Rotation Scheduler

This module runs the "surveillance loop" that makes cameras appear live:
- Every 6-10 seconds, rotates each camera to a new random clip
- Runs AI inference on the new clip
- Updates camera state with results

Runs in background threads without blocking Flask.
"""

import random
import time
import threading
from backend.services.camera_manager import (
    rotate_camera_video,
    get_video_absolute_path,
    update_camera_inference,
    get_camera_state
)
from backend.services.incident_service import process_video
from backend.services.incident_storage import add_incident
from backend.config import DEFAULT_CAMERAS

_scheduler_running = False
_scheduler_threads = []
_last_violence_time = 0
_last_crash_time = 0
_violence_cooldown = 40  # Seconds between violence events
_crash_cooldown = 40     # Seconds between crash events
_violence_lock = threading.Lock()
_crash_lock = threading.Lock()


def camera_rotation_loop(camera_id: str):
    """
    Infinite loop for a single camera:
    1. Wait 6-10 seconds (randomized to avoid sync)
    2. Rotate to new video
    3. Run AI inference
    4. Update state
    5. Repeat
    
    Runs in dedicated background thread per camera.
    """
    print(f"[ROTATION] Starting rotation loop for {camera_id}")
    
    while _scheduler_running:
        try:
            # Random interval between 6-10 seconds
            interval = random.uniform(6, 10)
            time.sleep(interval)
            
            if not _scheduler_running:
                break
            
            # Rotate to new video (check violence and crash cooldowns)
            global _last_violence_time, _last_crash_time
            with _violence_lock:
                time_since_violence = time.time() - _last_violence_time
                allow_violence = time_since_violence >= _violence_cooldown
            
            with _crash_lock:
                time_since_crash = time.time() - _last_crash_time
                allow_crash = time_since_crash >= _crash_cooldown
            
            new_video = rotate_camera_video(camera_id, allow_violence=allow_violence, allow_crash=allow_crash)
            print(f"[VIDEO] {camera_id} -> {new_video}")
            
            # Get absolute path for inference
            video_path = get_video_absolute_path(camera_id)
            
            # Run appropriate detector(s) for this camera type
            print(f"[INFERENCE] Running inference on {camera_id}...")
            result = process_video(
                camera_id,
                video_path,
                model_name="mobilenet",
                update_state=False,
            )
            
            # Log results with detector details
            event = result.get("event", "unknown")
            conf = result.get("confidence", 0)
            print(f"[RESULT] {camera_id}: {event.upper()} (confidence: {conf:.2%})")
            
            # Update camera state with inference results
            if result.get("success"):
                update_camera_inference(camera_id, result)
                
                # Log alerts, store incidents, and update cooldowns
                if result.get("event") == "violence":
                    conf = result.get("confidence", 0)
                    print(f"[ALERT] {camera_id} detected violence (confidence: {conf:.2f})")
                    
                    # Store incident for frontend viewing
                    add_incident(
                        camera_id=camera_id,
                        event_type="violence",
                        confidence=conf,
                        video_path=new_video,
                        model=result.get("model", "unknown")
                    )
                    
                    with _violence_lock:
                        _last_violence_time = time.time()
                        print(f"[COOLDOWN] Violence cooldown active for {_violence_cooldown}s")
                        
                elif result.get("event") == "traffic":
                    conf = result.get("confidence", 0)
                    print(f"🚗 ALERT: {camera_id} detected car crash (confidence: {conf:.2f})")
                    
                    add_incident(
                        camera_id=camera_id,
                        event_type="traffic",
                        confidence=conf,
                        video_path=new_video,
                        model=result.get("model", "unknown")
                    )
                    
                    with _crash_lock:
                        _last_crash_time = time.time()
                        print(f"🕐 Crash cooldown active for {_crash_cooldown}s")
        
        except Exception as e:
            print(f"[ERROR] Rotation loop {camera_id}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)  # Backoff on error


def start_rotation_scheduler():
    """
    Start background rotation for all cameras.
    Called once on Flask app startup.
    """
    global _scheduler_running, _scheduler_threads
    
    if _scheduler_running:
        print("⚠️  Scheduler already running")
        return
    
    _scheduler_running = True
    _scheduler_threads = []
    
    # Create one thread per camera
    for camera_id in DEFAULT_CAMERAS:
        thread = threading.Thread(
            target=camera_rotation_loop,
            args=(camera_id,),
            daemon=True,
            name=f"CameraRotation-{camera_id}"
        )
        thread.start()
        _scheduler_threads.append(thread)
    
    print(f"✅ Started rotation scheduler for {len(_scheduler_threads)} cameras")


def stop_rotation_scheduler():
    """
    Stop all rotation threads gracefully.
    """
    global _scheduler_running
    
    print("🛑 Stopping rotation scheduler...")
    _scheduler_running = False
    
    # Wait for threads to exit
    for thread in _scheduler_threads:
        thread.join(timeout=2.0)
    
    print("✅ Scheduler stopped")
