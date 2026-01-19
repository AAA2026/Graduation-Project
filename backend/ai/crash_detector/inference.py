import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torchvision.transforms.functional import to_pil_image

from model_architecture import MobileNetV2_LSTM
from transforms_setup import get_test_transform
from sampling import sample_frame_indices

def load_model(weights_path, device="cpu"):
    model = MobileNetV2_LSTM()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_video(model, video_path, num_frames=16, img_size=224, device="cpu"):
    test_transform = get_test_transform(img_size)

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)

    idxs = sample_frame_indices(total, num_frames)
    frames_np = vr.get_batch(idxs).asnumpy()

    frame_tensors = []
    for img in frames_np:
        pil_img = to_pil_image(img)
        frame_tensors.append(test_transform(pil_img))

    video_tensor = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(video_tensor)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()

    return {
        "normal": float(probs[0]),
        "accident": float(probs[1]),
        "prediction": "accident" if probs[1] > probs[0] else "normal"
    }
