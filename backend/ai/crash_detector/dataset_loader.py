import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from torchvision.transforms.functional import to_pil_image
from sampling import sample_frame_indices

class VideoDataset(Dataset):
    def __init__(self, samples, transform, num_frames=16):
        self.samples = samples
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)

        idxs = sample_frame_indices(total, self.num_frames)
        frames_np = vr.get_batch(idxs).asnumpy()

        frame_tensors = []
        for img in frames_np:
            pil = to_pil_image(img)
            frame_tensors.append(self.transform(pil))

        video_tensor = torch.stack(frame_tensors, dim=0)
        return video_tensor, torch.tensor(label, dtype=torch.long)
