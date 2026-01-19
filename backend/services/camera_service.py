# Camera state management feeding the frontend

CAMERAS = {
    "CAM-01": {"status": "online", "confidence": 0.0, "last_model": None, "last_latency": 0, "last_timestamp": 0},
    "CAM-02": {"status": "online", "confidence": 0.0, "last_model": None, "last_latency": 0, "last_timestamp": 0},
    "CAM-03": {"status": "online", "confidence": 0.0, "last_model": None, "last_latency": 0, "last_timestamp": 0},
}


def update_camera(camera_id: str, incident: bool, confidence: float, model: str = None, latency_ms: int = 0, timestamp: float = 0) -> None:
    if camera_id not in CAMERAS:
        CAMERAS[camera_id] = {"status": "online", "confidence": 0.0, "last_model": None, "last_latency": 0, "last_timestamp": 0}
    CAMERAS[camera_id]["status"] = "incident" if incident else "online"
    CAMERAS[camera_id]["confidence"] = round(float(confidence), 2)
    if model:
        CAMERAS[camera_id]["last_model"] = model
    if latency_ms:
        CAMERAS[camera_id]["last_latency"] = latency_ms
    if timestamp:
        CAMERAS[camera_id]["last_timestamp"] = timestamp


def get_camera_states() -> list[dict]:
    return [
        {
            "camera_id": cam_id,
            "status": data["status"],
            "confidence": data["confidence"],
            "last_model": data.get("last_model"),
            "last_latency": data.get("last_latency"),
        }
        for cam_id, data in CAMERAS.items()
    ]
