from backend.ai.inference import run_inference
from backend.ai.crash_detector import detect_crash
from backend.services.camera_service import update_camera
from backend.utils.smoothing import smooth_decision


def process_video(
    camera_id: str,
    video_path: str,
    model_name: str = "mobilenet",
    update_state: bool = True,
    enable_crash: bool = False,
    crash_threshold: float = 0.70,
) -> dict:
    """Run AI inference for a camera's video, apply smoothing, and update state.
    
    Automatically selects the right detector(s) based on camera type:
    - Violence cameras: Only run violence detection
    - Crash cameras: Only run crash detection
    - Mixed cameras: Run both and merge
    
    Args:
        camera_id: Camera identifier
        video_path: Path to video file
        model_name: ML model to use ("mobilenet" or "x3d")
        update_state: Whether to update camera_service state (default True)
        enable_crash: Override to force crash detection (for mixed cameras)
        crash_threshold: Minimum confidence to treat crash as an incident
    """
    from backend.config import VIOLENCE_CAMERAS, CRASH_CAMERAS
    
    # Auto-detect which detector to run based on camera type
    is_violence_camera = camera_id in VIOLENCE_CAMERAS
    is_crash_camera = camera_id in CRASH_CAMERAS
    is_mixed_camera = not (is_violence_camera or is_crash_camera)
    
    # Detect video type from path for mixed cameras
    video_path_lower = video_path.lower()
    is_violence_video = "violence" in video_path_lower and "no_violence" not in video_path_lower
    is_no_violence_video = "no_violence" in video_path_lower
    is_crash_video = "crash" in video_path_lower and "no_crash" not in video_path_lower
    is_no_crash_video = "no_crash" in video_path_lower
    
    violence_result = None
    crash_result = None
    
    # Run detection based on camera type AND video type
    if is_violence_camera:
        # Violence camera: always run violence detection only
        violence_result = run_inference(video_path, model_name=model_name)
    elif is_crash_camera:
        # Crash camera: always run crash detection only
        try:
            crash_result = detect_crash(video_path)
            print(f"[CRASH-DEBUG] {camera_id} crash_result: is_crash={crash_result.get('is_crash')}, conf={crash_result.get('confidence'):.2%}")
        except Exception as e:
            crash_result = {"is_crash": False, "confidence": 0.0, "error": str(e)}
            print(f"[CRASH-ERROR] {camera_id}: {e}")
    elif is_mixed_camera:
        # Mixed camera: run detector based on video type
        if is_violence_video or is_no_violence_video:
            # Violence-type video: only run violence detection
            violence_result = run_inference(video_path, model_name=model_name)
        elif is_crash_video or is_no_crash_video:
            # Crash-type video: only run crash detection
            try:
                crash_result = detect_crash(video_path)
            except Exception as e:
                crash_result = {"is_crash": False, "confidence": 0.0, "error": str(e)}
        else:
            # Unknown video type: run both detectors
            violence_result = run_inference(video_path, model_name=model_name)
            try:
                crash_result = detect_crash(video_path)
            except Exception as e:
                crash_result = {"is_crash": False, "confidence": 0.0, "error": str(e)}
    
    # Determine final event based on camera type
    if is_violence_camera:
        # Violence camera: only report violence or normal
        event = violence_result.get("event", "normal")
        confidence = violence_result.get("confidence", 0.0)
        model = violence_result.get("model")
    elif is_crash_camera:
        # Crash camera: only report traffic or normal
        event = "traffic" if (crash_result and crash_result.get("is_crash")) else "normal"
        confidence = crash_result.get("confidence", 0.0) if crash_result else 0.0
        model = crash_result.get("model") if crash_result else ""
    else:
        # Mixed camera: use result from whichever detector ran
        if violence_result:
            event = violence_result.get("event", "normal")
            confidence = violence_result.get("confidence", 0.0)
            model = violence_result.get("model")
        elif crash_result:
            event = "traffic" if crash_result.get("is_crash") else "normal"
            confidence = crash_result.get("confidence", 0.0)
            model = crash_result.get("model", "")
        else:
            event = "normal"
            confidence = 0.0
            model = "unknown"
        
        # If both ran, crash takes precedence if confident enough
        if violence_result and crash_result:
            event = violence_result.get("event", "normal")
            confidence = violence_result.get("confidence", 0.0)
            model = violence_result.get("model")
            
            if crash_result.get("is_crash") and crash_result.get("confidence", 0.0) >= crash_threshold:
                event = "traffic"
                confidence = crash_result.get("confidence", 0.0)
                model = crash_result.get("model", "mobilenet_lstm_crash")

    is_incident, smoothed_conf = smooth_decision(
        camera_id,
        confidence,
    )

    # Only update old camera_service if requested (for backward compat)
    if update_state:
        update_camera(
            camera_id=camera_id,
            incident=is_incident,
            confidence=smoothed_conf,
            model=model,
            latency_ms=violence_result.get("latency_ms", 0) if violence_result else 0,
            timestamp=violence_result.get("timestamp", 0) if violence_result else 0,
        )
    
    # Return enriched result for camera_manager
    return {
        "event": event,
        "confidence": confidence,
        "model": model,
        "latency_ms": violence_result.get("latency_ms", 0) if violence_result else 0,
        "timestamp": (violence_result or crash_result or {}).get("timestamp", 0),
        "success": True,
        "smoothed_confidence": smoothed_conf,
        "is_incident": is_incident,
        "violence_result": violence_result,
        "crash_result": crash_result,
    }
