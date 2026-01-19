# Export for external import
__all__ = ["run_inference"]
"""
Pure ML inference module - no Flask dependencies.
Loads MobileNetClip and X3D-S models for violence detection.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Model paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
MOBILENET_PATH = MODEL_DIR / "mobilenet_clip_best.pth"
X3D_PATH = MODEL_DIR / "x3d_s_best.pth"

# Inference settings
# Force CPU to avoid CUDA DLL loading issues in Flask
DEVICE = torch.device("cpu")
FRAME_COUNT = 16  # Number of frames to sample
IMG_SIZE = 224    # Input image size
THRESHOLD = 0.6   # Violence detection threshold

# Global model cache
_models = {
    "mobilenet": None,
    "x3d": None
}


def load_mobilenet_model():
    """Load MobileNetClip model from disk."""
    if not MOBILENET_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MOBILENET_PATH}")
    
    # MobileNetV2 backbone with custom head for binary classification
    from torchvision import models
    model = models.mobilenet_v2(pretrained=False)
    
    # Load trained weights first to check structure
    state_dict = torch.load(MOBILENET_PATH, map_location=DEVICE)
    
    # Check if weights have 'backbone.' prefix (wrapped model)
    if any(key.startswith('backbone.') for key in state_dict.keys()):
        # Remove 'backbone.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_state_dict[k.replace('backbone.', '')] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # Check classifier output size
    classifier_weight_key = 'classifier.1.weight'
    if classifier_weight_key in state_dict:
        num_classes = state_dict[classifier_weight_key].shape[0]
    else:
        num_classes = 2  # Default
    
    # Build classifier to match trained model
    if num_classes == 2:
        # 2-class classification (violence, normal)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, 2)
        )
    else:
        # Binary output with sigmoid
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, 1),
            nn.Sigmoid()
        )
    
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    return model


def load_x3d_model():
    """Load X3D-S model from disk."""
    if not X3D_PATH.exists():
        raise FileNotFoundError(f"Model not found: {X3D_PATH}")
    
    # X3D-S is a 3D CNN for video classification
    # This is a simplified placeholder - actual X3D would use pytorchvideo
    model = nn.Sequential(
        nn.Conv3d(3, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        nn.ReLU(),
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(24, 1),
        nn.Sigmoid()
    )
    
    state_dict = torch.load(X3D_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    return model


def get_model(model_name: str = "mobilenet"):
    """Get or load model from cache."""
    if _models[model_name] is None:
        if model_name == "mobilenet":
            _models[model_name] = load_mobilenet_model()
        elif model_name == "x3d":
            _models[model_name] = load_x3d_model()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    return _models[model_name]


def extract_frames(video_path: str, num_frames: int = FRAME_COUNT) -> np.ndarray:
    """Extract uniformly sampled frames from video.
    
    Returns:
        numpy array of shape (num_frames, H, W, 3)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = total_frames
    
    # Uniformly sample frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")
    
    return np.array(frames)


def preprocess_frames_mobilenet(frames: np.ndarray) -> torch.Tensor:
    """Preprocess frames for MobileNet (2D CNN).
    
    Args:
        frames: (num_frames, H, W, 3) numpy array
    
    Returns:
        tensor of shape (num_frames, 3, 224, 224)
    """
    processed = []
    
    for frame in frames:
        # Resize to 224x224
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        # HWC to CHW
        frame = np.transpose(frame, (2, 0, 1))
        processed.append(frame)
    
    # Stack to (num_frames, 3, 224, 224)
    tensor = torch.from_numpy(np.array(processed)).float()
    return tensor.to(DEVICE)


def preprocess_frames_x3d(frames: np.ndarray) -> torch.Tensor:
    """Preprocess frames for X3D (3D CNN).
    
    Args:
        frames: (num_frames, H, W, 3) numpy array
    
    Returns:
        tensor of shape (1, 3, num_frames, 224, 224)
    """
    processed = []
    
    for frame in frames:
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45])
        std = np.array([0.225, 0.225, 0.225])
        frame = (frame - mean) / std
        processed.append(frame)
    
    # Stack to (num_frames, H, W, 3)
    video = np.array(processed)
    # THWC to CTHW: (3, num_frames, 224, 224)
    video = np.transpose(video, (3, 0, 1, 2))
    # Add batch dimension: (1, 3, num_frames, 224, 224)
    tensor = torch.from_numpy(video).unsqueeze(0).float()
    return tensor.to(DEVICE)


def run_inference(video_path: str, model_name: str = "mobilenet") -> Dict:
    """
    Run violence detection inference on a video file.
    
    Args:
        video_path: Path to video file (MP4, AVI, etc.)
        model_name: "mobilenet" or "x3d"
    
    Returns:
        {
            "event": "violence" | "normal",
            "confidence": float (0-1),
            "model": str,
            "latency_ms": int,
            "timestamp": float
        }
    """
    start_time = time.time()
    
    try:
        # Extract frames from video
        frames = extract_frames(video_path, num_frames=FRAME_COUNT)
        
        # Get model
        model = get_model(model_name)
        
        # Preprocess
        if model_name == "mobilenet":
            inputs = preprocess_frames_mobilenet(frames)
            # Average predictions across all frames
            with torch.no_grad():
                outputs = model(inputs)  # (num_frames, 2) or (num_frames, 1)
                
                # Handle 2-class output
                if outputs.shape[1] == 2:
                    # Apply softmax and get violence probability (class 1)
                    probs = torch.softmax(outputs, dim=1)
                    violence_prob = probs[:, 1].mean().item()  # Violence class
                    normal_prob = probs[:, 0].mean().item()   # Normal class
                else:
                    # Binary sigmoid output
                    violence_prob = outputs.mean().item()
                    normal_prob = 1.0 - violence_prob
        
        elif model_name == "x3d":
            inputs = preprocess_frames_x3d(frames)
            with torch.no_grad():
                outputs = model(inputs)  # (1, 1)
                violence_prob = outputs.item()
                normal_prob = 1.0 - violence_prob
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Determine event
        event = "violence" if violence_prob >= THRESHOLD else "normal"
        
        # Return confidence of the PREDICTED class
        confidence = violence_prob if event == "violence" else normal_prob
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "event": event,
            "confidence": round(float(confidence), 3),
            "model": model_name,
            "latency_ms": latency_ms,
            "timestamp": time.time()
        }
    
    except Exception as e:
        # Return error state
        return {
            "event": "error",
            "confidence": 0.0,
            "model": model_name,
            "latency_ms": int((time.time() - start_time) * 1000),
            "timestamp": time.time(),
            "error": str(e)
        }


# ============================================
# VALIDATION TEST (can be run standalone)
# ============================================

if __name__ == "__main__":
    """
    Test inference on a sample video.
    Usage: python inference.py
    """
    import sys
    
    # Check if model files exist
    if not MOBILENET_PATH.exists():
        print(f"⚠️  Model not found: {MOBILENET_PATH}")
        print("Place trained .pth files in backend/models/")
        sys.exit(1)
    
    # Test with a dummy video path (replace with actual video)
    test_video = "demo.mp4"
    
    if Path(test_video).exists():
        print(f"🎥 Running inference on: {test_video}")
        print(f"🖥️  Device: {DEVICE}")
        print()
        
        result = run_inference(test_video, model_name="mobilenet")
        
        print("✅ Inference Result:")
        print(f"   Event: {result['event']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Model: {result['model']}")
        print(f"   Latency: {result['latency_ms']} ms")
        print(f"   Timestamp: {result['timestamp']}")
        
        if "error" in result:
            print(f"   ⚠️ Error: {result['error']}")
    else:
        print(f"⚠️  Test video not found: {test_video}")
        print("Create a sample video or update the test_video path")
