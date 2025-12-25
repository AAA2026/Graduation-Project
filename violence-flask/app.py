import os
import uuid
import tempfile
from pathlib import Path
from typing import Deque, Dict
from collections import deque

import cv2
import numpy as np
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

# --------------------------------------------------------------------
# PyTorch + video models (for violence detection)
# --------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from pytorchvideo.models.hub import x3d_s as x3d_s_pv
    from torchvision import models as tv_models
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except Exception:
    torch = None
    nn = None
    x3d_s_pv = None
    tv_models = None
    AutoImageProcessor = None
    AutoModelForImageClassification = None

# YOLOv8 for people counting
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# --------------------------------------------------------------------
# Paths and config
# --------------------------------------------------------------------
APP_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = APP_ROOT / "models"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "violence_people_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# YOLOv8n weights in the ROOT of the project (parent of this folder)
YOLO_WEIGHTS = (APP_ROOT.parent / "yolov8n.pt").resolve()

# Weapon classifier folder (Swin)
WEAPON_ROOT = (APP_ROOT.parent / "weapon-detector").resolve()
WEAPON_CLS_DIR = WEAPON_ROOT / "classifier"
WEAPON_THRESHOLD = 0.80  # prob threshold to say "there are weapons"

# Violence models: NEW models (.pth)
CONFIGS = {
    "x3d": {
        "pth": MODELS_DIR / "x3d_s_best.pth",
        "T": 13,
        "SIZE": 160,
    },
    "mobilenet": {
        "pth": MODELS_DIR / "mobilenet_clip_best.pth",
        "T": 13,
        "SIZE": 160,
    },
}
DEFAULT_MODEL_KEY = "mobilenet"  # use your best model by default

# Violence thresholds + smoothing (hysteresis)
TH_HIGH = 0.60   # 🔴 ALERT when EMA > 0.60
TH_LOW  = 0.30
ENTER_N = 5
EXIT_M  = 8
EMA_ALPHA = 0.8

# Optional label map (not strictly needed but we keep it)
LABEL_MAP = {"0": "NonViolence", "1": "Violence"}
lm_path = MODELS_DIR / "label_map.json"
if lm_path.exists():
    import json
    with open(lm_path, "r", encoding="utf-8") as f:
        LABEL_MAP = json.load(f)

app = Flask(__name__, static_folder="static", template_folder="templates")

# --------------------------------------------------------------------
# Global model caches
# --------------------------------------------------------------------
TORCH_MODELS: Dict[str, nn.Module] = {}
YOLO_MODEL = None
WEAPON_PROCESSOR = None
WEAPON_MODEL = None

if torch is not None:
    DEVICE_VIOLENCE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE_VIOLENCE = "cpu"

# --------------------------------------------------------------------
# Generic softmax (used for weapon classifier logits)
# --------------------------------------------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

# --------------------------------------------------------------------
# Preprocessing for frames
# --------------------------------------------------------------------
def preprocess_frame(bgr: np.ndarray, size: int) -> np.ndarray:
    """
    Resize keeping aspect, center-crop to size x size, BGR->RGB, float32/255, normalize (ImageNet).
    Returns CHW float array of shape (3, size, size).
    """
    h, w = bgr.shape[:2]
    scale = int(size * 1.14)  # slightly bigger, then center-crop
    short = min(h, w)
    r = scale / float(short)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    y = max((nh - size) // 2, 0)
    x = max((nw - size) // 2, 0)
    crop = resized[y:y + size, x:x + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1))  # C,H,W
    return chw

# --------------------------------------------------------------------
# Violence models via PyTorch
# --------------------------------------------------------------------
def get_violence_model(model_key: str) -> nn.Module:
    """
    Lazy-load PyTorch violence model (X3D-S or MobileNetClip) from .pth.
    """
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is not available on the server")

    if model_key not in CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")

    if model_key in TORCH_MODELS:
        return TORCH_MODELS[model_key]

    if model_key == "x3d":
        if x3d_s_pv is None:
            raise RuntimeError(
                "pytorchvideo is not installed. Install it with 'pip install pytorchvideo'."
            )
        # Build X3D-S and replace head to 2 classes (same as training)
        model = x3d_s_pv(pretrained=False)
        in_features = model.blocks[5].proj.in_features
        model.blocks[5].proj = nn.Linear(in_features, 2)

        state = torch.load(CONFIGS["x3d"]["pth"], map_location=DEVICE_VIOLENCE)
        model.load_state_dict(state)
        model.to(DEVICE_VIOLENCE)
        model.eval()

    elif model_key == "mobilenet":
        if tv_models is None:
            raise RuntimeError(
                "torchvision is not available. Install it with 'pip install torchvision'."
            )

        # Base MobileNetV2
        base_model = tv_models.mobilenet_v2(weights=None)
        in_features = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(in_features, 2)

        # Wrap into MobileNetClip: expect [B, T, 3, H, W]
        class MobileNetClip(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

            def forward(self, x):
                # x: [B, T, C, H, W]
                B, T, C, H, W = x.shape
                x = x.view(B * T, C, H, W)
                logits_frame = self.backbone(x)      # [B*T, 2]
                logits_frame = logits_frame.view(B, T, 2)
                logits = logits_frame.mean(dim=1)    # [B, 2]
                return logits

        model = MobileNetClip(base_model)

        state = torch.load(CONFIGS["mobilenet"]["pth"], map_location=DEVICE_VIOLENCE)
        model.load_state_dict(state)
        model.to(DEVICE_VIOLENCE)
        model.eval()

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    TORCH_MODELS[model_key] = model
    print(f"[INIT] Loaded PyTorch violence model '{model_key}' from {CONFIGS[model_key]['pth']}")
    return model


def run_violence_model(model_key: str, window: Deque[np.ndarray]) -> float:
    """
    Given a deque of preprocessed frames (CHW) of length T,
    run the chosen model and return P(violence).
    """
    model = get_violence_model(model_key)
    T = CONFIGS[model_key]["T"]

    if len(window) < T:
        return 0.0

    # window entries: (3, H, W)
    frames_np = np.stack(list(window), axis=0)  # [T, 3, H, W]
    with torch.no_grad():
        if model_key == "x3d":
            # X3D expects [B, C, T, H, W]
            x = np.transpose(frames_np, (1, 0, 2, 3))  # [3, T, H, W]
            x = torch.from_numpy(x).unsqueeze(0).to(DEVICE_VIOLENCE)
        else:
            # MobileNetClip expects [B, T, C, H, W]
            x = torch.from_numpy(frames_np).unsqueeze(0).to(DEVICE_VIOLENCE)

        logits = model(x)  # [1, 2]
        probs = torch.softmax(logits, dim=1)
        p_violence = float(probs[0, 1].item())

    return p_violence

# --------------------------------------------------------------------
# YOLOv8 people detector
# --------------------------------------------------------------------
def get_yolo_model():
    """Lazy-load YOLOv8n for people detection, using weights in project root."""
    global YOLO_MODEL
    if YOLO is None:
        return None
    if YOLO_MODEL is not None:
        return YOLO_MODEL

    if YOLO_WEIGHTS.exists():
        print(f"[INIT] Loading YOLOv8n from {YOLO_WEIGHTS}")
        YOLO_MODEL = YOLO(str(YOLO_WEIGHTS))
    else:
        # Fallback: let ultralytics download yolov8n.pt if not found
        print(f"[WARN] YOLO weights not found at {YOLO_WEIGHTS}, falling back to 'yolov8n.pt'")
        YOLO_MODEL = YOLO("yolov8n.pt")

    return YOLO_MODEL


def detect_people(frame_bgr: np.ndarray) -> int:
    """
    Run YOLOv8n on the frame to detect persons.
    Returns people_count (no drawing needed for final summary).
    """
    model = get_yolo_model()
    if model is None:
        return 0

    people_count = 0
    # YOLO expects RGB
    results = model(frame_bgr[..., ::-1], verbose=False)

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names.get(cls_id, str(cls_id))
            if "person" not in label.lower():
                continue
            if conf < 0.4:
                continue
            people_count += 1

    return people_count

# --------------------------------------------------------------------
# Weapon classifier (Swin)
# --------------------------------------------------------------------
def get_weapon_classifier():
    """
    Lazy-load Swin image classifier that predicts weapon types.
    We only care if any frame has prob >= WEAPON_THRESHOLD.
    """
    global WEAPON_PROCESSOR, WEAPON_MODEL
    if AutoImageProcessor is None or AutoModelForImageClassification is None or torch is None:
        return None, None
    if not WEAPON_CLS_DIR.exists():
        print(f"[WARN] Weapon classifier folder not found at {WEAPON_CLS_DIR}")
        return None, None
    if WEAPON_MODEL is not None and WEAPON_PROCESSOR is not None:
        return WEAPON_PROCESSOR, WEAPON_MODEL

    print(f"[INIT] Loading weapon classifier from {WEAPON_CLS_DIR}")
    WEAPON_PROCESSOR = AutoImageProcessor.from_pretrained(WEAPON_CLS_DIR)
    WEAPON_MODEL = AutoModelForImageClassification.from_pretrained(WEAPON_CLS_DIR)
    if torch.cuda.is_available():
        WEAPON_MODEL.to("cuda")
    WEAPON_MODEL.eval()
    return WEAPON_PROCESSOR, WEAPON_MODEL


def weapon_in_frame(frame_bgr: np.ndarray) -> bool:
    """
    Return True if this frame is classified as "weapon" with high confidence.
    """
    processor, model = get_weapon_classifier()
    if processor is None or model is None:
        return False

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    logits_np = logits.detach().cpu().numpy()
    probs = softmax(logits_np)[0]
    max_prob = float(np.max(probs))
    return max_prob >= WEAPON_THRESHOLD

# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", label_map=LABEL_MAP)


@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload video, returns job_id.
    Front-end will then:
      1) Show video using /video/<job_id>
      2) Call /analyze to get summary
    """
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(f.filename).suffix.lower() or ".mp4"
    job_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{job_id}{suffix}"
    f.save(out_path)

    return jsonify({"job_id": job_id, "filename": f.filename})


@app.route("/video/<job_id>")
def video(job_id):
    """
    Serve the original uploaded video so the browser can play it
    with pause/continue from the same point.
    """
    candidates = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not candidates:
        abort(404, "Job not found")
    path = candidates[0]
    return send_from_directory(path.parent, path.name, as_attachment=False)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze the whole video and return a summary:
      - violence_detected (bool)
      - max_violence_score (float)
      - people_involved (int)
      - weapons_present (bool)
      - mode (str): "violence_only" or "full"
    """
    data = request.get_json() or request.form

    # default to your best model (mobilenet clip)
    model_key = (data.get("model") or DEFAULT_MODEL_KEY).lower()

    # backward-compat: if someone still sends "tsm", map it to mobilenet
    if model_key == "tsm":
        model_key = "mobilenet"

    # analysis mode:
    #   "violence_only" -> just violence model
    #   "full"          -> violence + people + weapons
    mode = (data.get("mode") or "full").lower()
    if mode not in ("violence_only", "full"):
        mode = "full"

    job_id = data.get("job_id", "")

    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400
    if model_key not in CONFIGS:
        return jsonify({"error": f"Unknown model '{model_key}'"}), 400

    candidates = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not candidates:
        return jsonify({"error": "Job not found"}), 404
    video_path = str(candidates[0])

    if torch is None:
        return jsonify({"error": "PyTorch is not available on this server"}), 500

    # Try to initialize the model once (will raise if something is wrong)
    try:
        _ = get_violence_model(model_key)
    except Exception as e:
        return jsonify({"error": f"Could not load violence model: {e}"}), 500

    T = CONFIGS[model_key]["T"]
    SIZE = CONFIGS[model_key]["SIZE"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 500

    window: Deque[np.ndarray] = deque(maxlen=T)
    ema = 0.0
    alert = False
    streak_hi = 0
    streak_lo = 0

    violence_detected = False
    max_ema = 0.0
    max_people_overall = 0
    max_people_violent = 0
    weapons_present = False

    frame_idx = 0

    # turn sub-modules on/off based on mode
    do_people = mode != "violence_only"
    do_weapons = mode != "violence_only"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # ---------------- Violence detection ----------------
        chw = preprocess_frame(frame, SIZE)
        window.append(chw)
        if len(window) == T:
            prob = run_violence_model(model_key, window)  # P(violence)

            ema = EMA_ALPHA * ema + (1 - EMA_ALPHA) * prob
            max_ema = max(max_ema, ema)

            if not alert:
                if ema > TH_HIGH:
                    streak_hi += 1
                    if streak_hi >= ENTER_N:
                        alert = True
                        streak_hi = 0
                else:
                    streak_hi = 0
            else:
                if ema < TH_LOW:
                    streak_lo += 1
                    if streak_lo >= EXIT_M:
                        alert = False
                        streak_lo = 0
                else:
                    streak_lo = 0

            if ema > TH_HIGH:
                violence_detected = True

        # ---------------- People detection (optional) ----------------
        if do_people:
            people_count = detect_people(frame)
            max_people_overall = max(max_people_overall, people_count)
            if ema > TH_HIGH:
                max_people_violent = max(max_people_violent, people_count)

        # ---------------- Weapon detection (optional, every 5th frame) ----------------
        if do_weapons and frame_idx % 5 == 0 and not weapons_present:
            if weapon_in_frame(frame):
                weapons_present = True

    cap.release()

    # If violence exists, "people involved" = most people in any violent frame.
    # Otherwise fall back to global max people count.
    if do_people:
        people_involved = max_people_violent if violence_detected else max_people_overall
    else:
        people_involved = 0

    if not do_weapons:
        weapons_present = False

    result = {
        "job_id": job_id,
        "model": model_key,
        "mode": mode,
        "violence_detected": bool(violence_detected),
        "max_violence_score": round(max_ema, 3),
        "people_involved": int(people_involved),
        "weapons_present": bool(weapons_present),
    }
    return jsonify(result)


if __name__ == "__main__":
    # For LAN use host="0.0.0.0"
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
