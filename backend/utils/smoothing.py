from collections import deque
from backend.config import SMOOTHING_WINDOW, VIOLENCE_THRESHOLD

_history: dict[str, deque] = {}


def smooth_decision(camera_id: str, confidence: float) -> tuple[bool, float]:
    """Apply temporal smoothing over recent confidences for a camera.

    Returns (is_incident, smoothed_confidence).
    """
    if camera_id not in _history:
        _history[camera_id] = deque(maxlen=SMOOTHING_WINDOW)

    _history[camera_id].append(float(confidence))

    avg_conf = sum(_history[camera_id]) / len(_history[camera_id])
    is_incident = avg_conf >= VIOLENCE_THRESHOLD

    return is_incident, round(avg_conf, 2)
