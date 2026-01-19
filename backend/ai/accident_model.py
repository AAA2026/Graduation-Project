# VIGIL Accident Model - Clean
# TODO: Implement crash detection

# Placeholder for actual accident detection model integration


import time
from typing import Dict


import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torchvision.transforms.functional import to_pil_image
from backend.ai.crash_detector.model_architecture import MobileNetV2_LSTM
from backend.ai.crash_detector.transforms_setup import get_test_transform
from backend.ai.crash_detector.sampling import sample_frame_indices
from pathlib import Path

class AccidentModel:
    def __init__(self, model_path: str | None = None):
        if model_path is None:
            # Default to the known model path in crash_detector
            model_path = str(Path(__file__).parent / "crash_detector" / "mobilenetv2_lstm_finetuned.pt")
        
        self.model_path = model_path
        self.device = "cpu"  # Force CPU for demo stability
        self.model = None
        self.test_transform = get_test_transform(224)
        
        # Lazy load in predict to avoid startup delay, or load now?
        # Let's load now for clarity, but inside a try-except block just in case
        try:
            self.load_model()
        except Exception as e:
            print(f"[ACCIDENT_MODEL] Warning: Model load failed at startup: {e}")

    def load_model(self):
        print(f"[ACCIDENT_MODEL] Loading model from {self.model_path}")
        self.model = MobileNetV2_LSTM()
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print("[ACCIDENT_MODEL] Model loaded successfully.")

    def predict(self, video_path: str) -> float:
        """
        Returns a probability (0.0 - 1.0) of accident in the video.
        """
        if self.model is None:
            self.load_model()
            
        try:
            # Video loading logic
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            
            # Sample 16 frames
            idxs = sample_frame_indices(total, 16)
            frames_np = vr.get_batch(idxs).asnumpy()
            
            frame_tensors = []
            for img in frames_np:
                pil_img = to_pil_image(img)
                frame_tensors.append(self.test_transform(pil_img))
            
            # Stack and infer
            video_tensor = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out = self.model(video_tensor)
                probs = F.softmax(out, dim=1)[0].cpu().numpy()
            
            # Index 1 is accident, Index 0 is normal
            accident_prob = float(probs[1])
            return accident_prob
            
        except Exception as e:
            print(f"[ACCIDENT_MODEL] Inference error on {video_path}: {e}")
            return 0.0

# Singleton instance
_accident_model = AccidentModel()


def detect_crash(video_path: str) -> Dict:
    """
    Run accident detection on the given video and return result dict.
    """
    start = time.time()
    confidence = _accident_model.predict(video_path)
    latency_ms = int((time.time() - start) * 1000)
    return {
        "event": "car_crash",
        "confidence": confidence,
        "model": "mobilenetv2_lstm",
        "latency_ms": latency_ms,
        "timestamp": time.time()
    }
