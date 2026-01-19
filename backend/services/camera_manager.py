# VIGIL Camera Manager - Clean
import threading
from pathlib import Path
import random
from backend.config import VIOLENCE_CAMERAS, CRASH_CAMERAS

def get_video_path(camera_id):
    dataroot = Path(__file__).parent.parent.parent
    video_dir = dataroot / 'Videos'
    if camera_id in VIOLENCE_CAMERAS:
        folder = video_dir / 'violence'
    elif camera_id in CRASH_CAMERAS:
        folder = video_dir / 'crash'
    else:
        folder = video_dir / 'no_violence'
        if not folder.exists() or not list(folder.glob('*.mp4')):
            folder = video_dir / 'no_crash'
    if folder.exists():
        files = [f for f in folder.glob('*.mp4') if f.is_file()]
        if files:
            return str(folder.name + '/' + random.choice(files).name)
    # fallback: pick any video from Videos/
    dataroot = Path(__file__).parent.parent.parent
    video_dir = dataroot / 'Videos'
    for subfolder in ['violence', 'crash', 'no_violence', 'no_crash']:
        folder = video_dir / subfolder
        if folder.exists():
            files = [f for f in folder.glob('*.mp4') if f.is_file()]
            if files:
                return str(folder.name + '/' + random.choice(files).name)
    # If no valid video found, log warning and return None
    import logging
    logging.warning(f"No valid video found for camera {camera_id}")
    return None

def rotate_camera_video(camera_id, allow_violence=True, allow_crash=True):
    """
    Rotates the video for a camera. Returns the new video path.
    """
    from backend.config import VIOLENCE_CAMERAS, CRASH_CAMERAS
    dataroot = Path(__file__).parent.parent.parent
    video_dir = dataroot / 'Videos'
    if camera_id in VIOLENCE_CAMERAS:
        folder = video_dir / 'violence'
    elif camera_id in CRASH_CAMERAS:
        folder = video_dir / 'crash'
    else:
        folder = video_dir / 'no_violence'
        if not folder.exists() or not list(folder.glob('*.mp4')):
            folder = video_dir / 'no_crash'
    if folder.exists():
        files = list(folder.glob('*.mp4'))
        if files:
            chosen = random.choice(files)
            return str(folder.name + '/' + chosen.name)
    return ''

def update_camera_inference(camera_id, inference_result):
    """
    Updates the camera state with inference result.
    """
    state = camera_states.get(camera_id, {})
    state.update(inference_result)
    camera_states.set(camera_id, state)

import threading
from pathlib import Path
import random

class SafeDict:
    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()
    
    def get(self, key, default=None):
        with self._lock:
            return self._d.get(key, default)
    
    def set(self, key, value):
        with self._lock:
            self._d[key] = value
    
    def update(self, other):
        with self._lock:
            self._d.update(other)

    def all(self):
        with self._lock:
            return dict(self._d)

# Camera state storage
camera_states = SafeDict()

# Incident storage
incidents = SafeDict()

# Global system state
_system_state = SafeDict()
_system_state.set("offline_mode", False)

def get_offline_mode_state() -> bool:
    """Get the current offline mode state (boolean)."""
    return _system_state.get("offline_mode", False)

def set_offline_mode_state(enabled: bool):
    """Set the current offline mode state."""
    _system_state.set("offline_mode", enabled)
    import logging
    logging.info(f"System offline mode set to: {enabled}")

def get_video_absolute_path(camera_id: str) -> str:
    """
    Returns the absolute path to the video file for a given camera ID.
    """
    """
    Returns the relative path to a valid video file for a given camera ID, based on camera type.
    """
    from backend.config import VIOLENCE_CAMERAS, CRASH_CAMERAS
    dataroot = Path(__file__).parent.parent.parent
    video_dir = dataroot / 'Videos'
    if camera_id in VIOLENCE_CAMERAS:
        folder = video_dir / 'violence'
    elif camera_id in CRASH_CAMERAS:
        folder = video_dir / 'crash'
    else:
        # Assign non-violence/non-crash videos for other cameras
        folder = video_dir / 'no_violence'
        if not folder.exists() or not list(folder.glob('*.mp4')):
            folder = video_dir / 'no_crash'
    if folder.exists():
        files = list(folder.glob('*.mp4'))
        if files:
            # Return path relative to Videos/ for frontend
            return str(folder.name + '/' + files[0].name)
    # fallback: pick any video from Videos/
    for subfolder in ['violence', 'crash', 'no_violence', 'no_crash']:
        folder = video_dir / subfolder
        if folder.exists():
            files = list(folder.glob('*.mp4'))
            if files:
                return str(folder.name + '/' + files[0].name)
    return ''

# --- MISSING FUNCTIONS FOR SIMULATOR ---
def rotate_camera_video(camera_id, allow_violence=True, allow_crash=True):
    """
    Rotates the video for a camera. Returns the new video path.
    """
    """
    Rotates the video for a camera. Returns the new video path (relative to Videos/).
    Picks a random video from the correct folder for the camera type.
    """
    from backend.config import VIOLENCE_CAMERAS, CRASH_CAMERAS
    dataroot = Path(__file__).parent.parent.parent
    video_dir = dataroot / 'Videos'
    if camera_id in VIOLENCE_CAMERAS:
        folder = video_dir / 'violence'
    elif camera_id in CRASH_CAMERAS:
        folder = video_dir / 'crash'
    else:
        # Randomly assign no_violence or no_crash to generic cameras to vary content
        # biased 60% towards no_crash (cars) to satisfy user preference
        if random.random() < 0.6:
            folder = video_dir / 'no_crash'
            if not folder.exists() or not list(folder.glob('*.mp4')):
                folder = video_dir / 'no_violence'
        else:
            folder = video_dir / 'no_violence'
            if not folder.exists() or not list(folder.glob('*.mp4')):
                folder = video_dir / 'no_crash'
    if folder.exists():
        files = [f for f in folder.glob('*.mp4') if f.is_file()]
        if files:
            chosen = random.choice(files)
            return str(folder.name + '/' + chosen.name)
            
    # If no valid video found, log warning and return None
    import logging
    logging.warning(f"No valid video found for camera {camera_id} in {folder}")
    return ''

def update_camera_inference(camera_id, inference_result):
    """
    Updates the camera state with inference result.
    """
    state = camera_states.get(camera_id, {})
    state.update(inference_result)
    camera_states.set(camera_id, state)