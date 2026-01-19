# VIGIL System Overview

A quick, readable reference for how the VIGIL demo runs its simulated cameras, AI inference, and API responses.

## What the system does
- Simulates 12 CCTV cameras that continuously rotate 5–7 second MP4 clips from `Videos/`
- Runs PyTorch models (violence, crash; optional weapon) on each clip
- Maintains live camera state and incident records in memory
- Serves JSON via Flask for the dashboards that poll every ~3 seconds

## End-to-end loop (background thread)
1) Pick a video for each camera (default: ~85% normal, ~15% violence/crash) with per-camera cooldowns.  
2) Run detector(s) via `backend/ai/inference.py` (returns `event`, `confidence`, `model`, `latency_ms`, `timestamp`).  
3) Update the shared camera state (`backend/services/camera_manager.py`).  
4) If confidence clears threshold, add/merge an incident (`backend/services/incident_storage.py`).

## Core components
- Flask app: `backend/app.py` (routes, starts simulator thread)
- Simulator: `backend/services/camera_simulator.py` (video rotation + inference calls)
- Inference entrypoint: `backend/ai/inference.py` (loads models in `backend/models/`)
- Camera state: `backend/services/camera_manager.py` (thread-safe dict)
- Incident registry: `backend/services/incident_storage.py` (dedupes within a time window)
- Frontend polling: `/api/live-status` every ~3s (JS in `js/`)

## Data flow
- Background: Simulator rotates clips → runs inference → updates state/incident store. No inference on incoming HTTP requests.  
- API read path: `/api/live-status` reads current state; `/api/simulator-stats` reads simulator metadata; `/videos/<path>` serves MP4s.

## Inference details
- `backend/ai/inference.py` centralizes detectors and exposes simple functions (e.g., `detect_violence(video_path)`).
- Each detector returns a dict: `event`, `confidence` (0–1), `model` name, `latency_ms`, `timestamp`.
- Models are loaded from `backend/models/` with PyTorch (see `backend/config.py` for model paths).
- Thresholding and incident gating happen outside the model: the simulator compares `confidence` to configured thresholds before flagging.

## Configuration knobs
Edit `backend/config.py`:
- `VIOLENCE_THRESHOLD` (default 0.90)
- `ACCIDENT_THRESHOLD` (default 0.80)
- `DEFAULT_CAMERAS` list
- Model paths/names

Edit simulator constructor args in `backend/services/camera_simulator.py`:
- `rotation_interval` (seconds between video swaps; default ~7)
- `violence_probability` (default 0.15)
- `violence_cooldown_per_camera` (seconds before same camera can alert again)

## Camera state & incidents
- Camera state fields: `camera_id`, `video`, `event`, `confidence`, `model`, `latency_ms`, `last_update`, `status`.
- Incident deduplication: incidents of the same camera/type within a configured window merge instead of spamming new entries.

## Dataset layout
- `Videos/violence`, `Videos/no_violence`, `Videos/crash`, `Videos/no_crash`
- Paths are stored relative to project root (e.g., `videos/violence/clip_001.mp4`).

## Running locally
1) Install dependencies: `pip install -r requirements.txt` (after activating your env).  
2) Start backend + simulator: `python backend/app.py`.  
3) Health checks: visit `http://127.0.0.1:5000/`, `http://127.0.0.1:5000/api/live-status`, `http://127.0.0.1:5000/api/simulator-stats`.

## Adjusting behavior quickly
- More alerts: lower thresholds in `backend/config.py`; increase `violence_probability` in `camera_simulator.py`.
- Fewer alerts: raise thresholds; lengthen `violence_cooldown_per_camera`.
- Slower rotation: increase `rotation_interval`.

## Adding a new detector
1) Create `backend/ai/<new_detector>.py` with a `detect_<thing>(video_path)` returning the standard dict.  
2) Import and expose it in `backend/ai/inference.py`.  
3) Wire it into the simulator’s rotation/inference logic so it runs per clip.  
4) Optionally add new incident rendering in `js/` dashboard files if you need UI changes.

## Troubleshooting quick notes
- No videos: ensure `Videos/` folders exist with MP4s or run `python check_demo_setup.py`.
- No alerts: confirm model files exist in `backend/models/` and thresholds aren’t too high.
- Simulator not running: check the console for the startup message from `camera_simulator.py`.
- Frontend stuck: inspect browser console and verify `/api/live-status` returns valid JSON.
