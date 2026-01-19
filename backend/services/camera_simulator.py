import threading
import time
import random
from pathlib import Path
from typing import Optional
from backend.config import DEFAULT_CAMERAS, VIOLENCE_CAMERAS, CRASH_CAMERAS, PEOPLE_COUNT_CAMERAS, VIOLENCE_THRESHOLD, ACCIDENT_THRESHOLD
from backend.services.camera_manager import rotate_camera_video, update_camera_inference, get_video_absolute_path, get_offline_mode_state
from backend.services.incident_storage import add_incident
from backend.ai.inference import run_inference
from backend.ai.accident_model import detect_crash

# Placeholder for model inference import
# from backend.ai.inference import run_inference

class CameraSimulator:
    def __init__(self, camera_ids, video_dir, rotation_interval=120, violence_probability=0.10):
        self.camera_ids = camera_ids
        self.video_dir = Path(video_dir)
        self.rotation_interval = rotation_interval
        self.violence_probability = violence_probability
        self._stop_event = threading.Event()
        self.threads = []
        self.camera_states = {cid: {} for cid in camera_ids}
        # Force initial state: assign a video and status to each camera
        for idx, cid in enumerate(camera_ids):
            # Use rotate_camera_video to respect camera types (crash/violence)
            rel_path = rotate_camera_video(cid)
            self.camera_states[cid] = {
                "camera_id": cid,
                "status": "online",
                "video": rel_path,
                "event": "none",
                "confidence": 0.0,
                "last_update": None
            }
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.inference_count = 0
        self.inference_count = 0
        self.violence_blocked_until = {}  # Timestamp until which violence detection is blocked
        self.crash_blocked_until = {}     # Timestamp until which crash detection is blocked
        self.last_video_rotation = {}      # Track last rotation time per camera
        self.processed_incident_videos = set() # Track videos that have already triggered an incident
        self.last_video_rotation = {}      # Track last rotation time per camera
        self.processed_incident_videos = set() # Track videos that have already triggered an incident
        self.video_rotation_duration = rotation_interval # Use the configured interval
        self.violence_cooldown = 40  # Seconds before allowing next violence (increased for better silence)
        self.crash_cooldown = 40     # Seconds before allowing next crash
    
    def start(self):
        """Start the simulation loop in a background thread."""
        if self.running:
            print("⚠️  Simulator already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.thread.start()
        print(f"🎬 Camera Simulator started (rotation: {self.rotation_interval}s)")
    
    def stop(self):
        """Stop the simulation loop."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            print("🛑 Camera Simulator stopped")
    
    def _simulation_loop(self):
        """
        Main simulation loop - runs in background thread.
        Periodically rotates videos and runs inference.
        """
        print("▶️  Starting camera simulation loop...")
        
        while self.running:
            current_time = time.time()
            for camera_id in DEFAULT_CAMERAS:
                try:
                    # Check if we should rotate the video (video_rotation_duration)
                    now = time.time()
                    last_rot = self.last_video_rotation.get(camera_id, 0)
                    
                    # Force initial rotation if no video, otherwise check duration
                    current_video = self.camera_states.get(camera_id, {}).get("video")
                    
                    if not current_video or (now - last_rot > self.video_rotation_duration):
                        rel_video_path = rotate_camera_video(camera_id)
                        self.last_video_rotation[camera_id] = now
                        # Cache it if needed, or rely on rotate_camera_video returning simple path
                    else:
                        # KEEP EXISTING VIDEO - Re-fetch current from camera_states to be sure
                        # Actually rotate_camera_video logic in this codebase is purely functional/random?
                        # We need to persist the current video path.
                        # The camera_states logic in camera_manager holds the state.
                        # We should just NOT call rotate_camera_video and use existing.
                        # But wait, we need 'rel_video_path' for inference below.
                        # Get current video from camera_states if available
                        state = self.camera_states.get(camera_id, {})
                        rel_video_path = state.get("video")
                        
                        # Fallback if somehow missing
                        if not rel_video_path:
                             rel_video_path = rotate_camera_video(camera_id)
                             self.last_video_rotation[camera_id] = now

                    video_path = str(self.video_dir / rel_video_path) if rel_video_path else ''
                    if rel_video_path and Path(rel_video_path).is_absolute():
                        video_path = rel_video_path
                    # Violence section: run people counting and violence detection
                    if camera_id in VIOLENCE_CAMERAS:
                        try:
                            from backend.ai.people_counter.yolov8 import detect_people_count
                            people_result = detect_people_count(video_path)
                            with self.lock:
                                update_camera_inference(camera_id, people_result)
                                self.inference_count += 1
                                offline = get_offline_mode_state()
                                if not offline and people_result.get("count", 0) > 15:
                                    print(f"   👥 {camera_id} detected high people count ({people_result['count']}) - Merging into Violence Check")
                                    # Do NOT trigger separate incident here.
                                    # Let the main violence/inference loop handle incident creation
                                    # so it respects cooldowns and duplicate checks.
                                elif offline:
                                    print(f"[OFFLINE MODE] Skipping people_count incident for {camera_id}")
                        except Exception as e:
                            print(f"❌ People counting failed for {camera_id}: {e}")
                    # Determine model by video folder
                    try:
                        video_folder = Path(video_path).parent.name.lower()
                        if video_folder in ["crash", "no_crash"]:
                            # Use crash model
                            result = run_inference(video_path, camera_id=camera_id)
                        elif video_folder in ["violence", "no_violence"]:
                            # Use violence model (with people counting)
                            result = run_inference(video_path, camera_id=camera_id)
                            # CRITICAL: Ensure early people_result is preserved if run_inference didn't return it
                            if 'people_count' not in result and 'people_result' in locals() and people_result:
                                result['people_count'] = people_result.get('count', 0)
                        else:
                            # Unknown: skip
                            print(f"[SIMULATOR] Unknown video type for {camera_id}: {video_path}")
                            continue
                        with self.lock:
                            update_camera_inference(camera_id, result)
                            self.inference_count += 1
                            # Incident creation logic
                            event = result.get('event', '').lower()
                            confidence = result.get('confidence', 0)
                            print(f"   [DEBUG] {camera_id} Inference: Event='{event}', Conf={confidence:.2f}, Folder='{video_folder}'")
                            
                            # Enforce thresholds & Offline Mode
                            offline = get_offline_mode_state()
                            
                            if offline:
                                # detailed logging if needed, or just skip
                                if (event == "violence" and confidence >= VIOLENCE_THRESHOLD) or (event == "car_crash" and confidence >= ACCIDENT_THRESHOLD):
                                     print(f"   ⏸️ OFFLINE: Skipping incident detection for {camera_id}")
                            else:
                                if video_folder in ["violence", "no_violence"] and event == "violence":
                                    if confidence >= VIOLENCE_THRESHOLD:
                                        now = time.time()
                                        blocked_until = self.violence_blocked_until.get(camera_id, 0)
                                        
                                        if video_path in self.processed_incident_videos:
                                            print(f"   ℹ️ Skipping duplicate incident for {video_path}")
                                        elif now < blocked_until:
                                            print(f"   ⏳ Snoozing detection for {camera_id} (Blocked until {blocked_until:.1f} vs Now {now:.1f})")
                                        else:
                                            print(f"   [DEBUG] Triggering Violence: {camera_id} | BlockedUntil: {blocked_until:.1f} | Now: {now:.1f}")
                                            # Double check to be absolutely sure
                                            if now < blocked_until:
                                                print("   [CRITICAL ERROR] Race condition detected! Aborting trigger.")
                                                continue
                                            people = result.get('people_count', 0)
                                            label = f"Violence incident involving {people} people" if people else "Violence incident detected"
                                            add_incident(
                                                camera_id,
                                                "violence",
                                                confidence,
                                                rel_video_path,
                                                result.get('model', 'unknown'),
                                                {"label": label, "timestamp": result.get('timestamp'), "people_count": people} if people else {"label": label, "timestamp": result.get('timestamp')}
                                            )
                                            self.processed_incident_videos.add(video_path)
                                            # Set NEXT block time
                                            self.violence_blocked_until[camera_id] = now + self.violence_cooldown
                                    else:
                                        print(f"   Note: Violence detected but confidence {confidence:.2f} < {VIOLENCE_THRESHOLD}")

                                elif video_folder in ["crash", "no_crash"] and event == "car_crash":
                                    if confidence >= ACCIDENT_THRESHOLD:
                                        now = time.time()
                                        blocked_until = self.crash_blocked_until.get(camera_id, 0)

                                        if video_path in self.processed_incident_videos:
                                            print(f"   ℹ️ Skipping duplicate incident for {video_path}")
                                        elif now < blocked_until:
                                            print(f"   ⏳ Snoozing detection for {camera_id} (Blocked for {blocked_until - now:.1f}s)")
                                        else:
                                            add_incident(
                                                camera_id,
                                                "crash",
                                                confidence,
                                                rel_video_path,
                                                result.get('model', 'unknown'),
                                                {"label": "Car crash detected", "timestamp": result.get('timestamp')}
                                            )
                                            self.processed_incident_videos.add(video_path)
                                            self.crash_blocked_until[camera_id] = now + self.crash_cooldown
                                    else:
                                        print(f"   Note: Crash detected but confidence {confidence:.2f} < {ACCIDENT_THRESHOLD}")
                        print(f"   ✓ {camera_id}: {result.get('event', 'unknown').upper()} (conf: {result.get('confidence', 0.0):.2%})")
                    except Exception as e:
                        print(f"❌ Error processing {camera_id}: {e}")
                    # Removed sleep from here to prevent serial blocking
                except Exception as e:
                    print(f"❌ Error in camera loop for {camera_id}: {e}")

            
            # Sleep ONCE per full rotation of all cameras
            time.sleep(self.rotation_interval)


    
    def _demo_inference(self, camera_id: str, video_abs_path: str):
        """
        Fallback demo inference when real inference unavailable.
        Simulates realistic results based on video folder.
        """
        # Determine event type from video path
        lower_path = video_abs_path.lower()
        if "violence" in lower_path:
            event = "violence"
            confidence = random.uniform(0.92, 0.99)
        elif "crash" in lower_path:
            event = "traffic"
            confidence = random.uniform(0.90, 0.99)
        else:
            event = "normal"
            confidence = random.uniform(0.85, 0.99)
        
        result = {
            "event": event,
            "confidence": confidence,
            "model": "demo-mode",
            "latency_ms": random.randint(50, 150),
            "timestamp": time.time()
        }
        
        with self.lock:
            update_camera_inference(camera_id, result)
            self.inference_count += 1
            
            # Check offline mode before creating incidents
            offline = get_offline_mode_state()
            if offline:
                print(f"   🔴 OFFLINE MODE: Skipping incident creation for {camera_id}")
            
            # Store violence incidents with cooldown (unless offline)
            if not offline and event == "violence":
                now = time.time()
                blocked_until = self.violence_blocked_until.get(camera_id, 0)
                
                if now < blocked_until:
                     print(f"   ⏳ [DEMO] Snoozing violence detection for {camera_id}")
                else:
                    conf = result.get("confidence", 0)
                    if conf >= VIOLENCE_THRESHOLD:
                        print(f"   🚨 ALERT: {camera_id} detected violence (confidence: {conf:.2%})")
                        add_incident(
                            camera_id=camera_id,
                            event_type="violence",
                            confidence=conf,
                            video_path=Path(video_abs_path).name if "Videos" not in video_abs_path else str(Path(video_abs_path).relative_to(Path(video_abs_path).parent.parent)),
                            model=result.get("model", "unknown")
                        )
                        # Set NEXT block time
                        self.violence_blocked_until[camera_id] = now + self.violence_cooldown
                    else:
                        print(f"   Note: Demo Violence rejected (conf {conf:.2f} < {VIOLENCE_THRESHOLD})")
            
            # Store crash incidents with cooldown (unless offline)
            if not offline and event == "traffic":
                now = time.time()
                blocked_until = self.crash_blocked_until.get(camera_id, 0)
                
                if now < blocked_until:
                     print(f"   ⏳ [DEMO] Snoozing crash detection for {camera_id}")
                else:
                    conf = result.get("confidence", 0)
                    if conf >= ACCIDENT_THRESHOLD:
                        print(f"   🚗 ALERT: {camera_id} detected car crash (confidence: {conf:.2%})")
                        add_incident(
                            camera_id=camera_id,
                            event_type="crash",
                            confidence=conf,
                            video_path=Path(video_abs_path).name if "Videos" not in video_abs_path else str(Path(video_abs_path).relative_to(Path(video_abs_path).parent.parent)),
                            model=result.get("model", "unknown")
                        )
                        # Set NEXT block time
                        self.crash_blocked_until[camera_id] = now + self.crash_cooldown
                    else:
                        print(f"   Note: Demo Crash rejected (conf {conf:.2f} < {ACCIDENT_THRESHOLD})")
        
        print(f"   ℹ️  {camera_id}: {event.upper()} (demo-mode, conf: {confidence:.2%})")
    
    def get_stats(self) -> dict:
        """Get simulator statistics."""
        with self.lock:
            return {
                "running": self.running,
                "inferences_run": self.inference_count,
                "rotation_interval": self.rotation_interval,
                "violence_probability": self.violence_probability,
                "violence_cooldown_s": self.violence_cooldown,
                "crash_cooldown_s": self.crash_cooldown,
                "cameras_monitored": len(DEFAULT_CAMERAS)
            }

    def clear_processed_video(self, video_path: str, camera_id: str = None):
        """Allow a specific video to trigger an incident again (e.g. after resolve), but snoozed."""
        with self.lock:
            # Try removing both relative and potential variations
            if video_path in self.processed_incident_videos:
                self.processed_incident_videos.remove(video_path)
                print(f"♻️  Re-enabled detection for video: {video_path}")
            
            # If camera_id provided (or inferred?), snooze it to prevent INSTANT reappearance
            if camera_id:
                now = time.time()
                future_block = now + self.violence_cooldown
                self.violence_blocked_until[camera_id] = future_block
                self.crash_blocked_until[camera_id] = future_block
                # Also force regular rotation next loop so we don't just stare at the same video? 
                # Actually, user might want to keep watching. Just snooze is enough.
                print(f"      [DEBUG] Snoozing {camera_id} until {future_block:.1f} after single-ack")

    def clear_all_processed_videos(self):
        """Reset all video detection history but SNOOZE detection for a cooldown period."""
        with self.lock:
            self.processed_incident_videos.clear()
            
            # Set all cooldowns to FUTURE timestamp to prevent immediate re-detection
            # This gives the user a grace period of 'silence' after clearing
            now = time.time()
            future_block = now + self.violence_cooldown # Block for full cooldown duration
            
            # Force all cameras to rotate video on next loop iteration
            for cid in self.camera_ids:
                self.violence_blocked_until[cid] = future_block
                self.crash_blocked_until[cid] = future_block
                self.last_video_rotation[cid] = 0
                print(f"      [DEBUG] Snoozing {cid} until {future_block:.1f} (Now: {now:.1f})")
                
            print(f"♻️  Reset processed videos, forced rotation, and set global snooze for {len(self.camera_ids)} cams.")


# Global simulator instance
_simulator: Optional[CameraSimulator] = None
_simulator_lock = threading.Lock()


def get_simulator() -> CameraSimulator:
    """Get or create global simulator instance."""
    global _simulator
    
    if _simulator is None:
        with _simulator_lock:
            if _simulator is None:
                _simulator = CameraSimulator(
                    rotation_interval=2.0,
                    violence_probability=0.15
                )
    
    return _simulator


def start_camera_simulator():
    """Start the global camera simulator (called from app.py on startup)."""
    simulator = get_simulator()
    if not simulator.running:
        simulator.start()


def stop_camera_simulator():
    """Stop the global camera simulator (called on app shutdown)."""
    if _simulator and _simulator.running:
        _simulator.stop()
