import numpy as np

def sample_frame_indices(num_total_frames, num_frames):
    if num_total_frames <= 0:
        return np.zeros(num_frames, dtype=np.int64)

    if num_total_frames <= num_frames:
        return np.linspace(0, num_total_frames - 1, num_frames).astype(np.int64)

    segment_length = num_total_frames // num_frames
    indices = []
    for i in range(num_frames):
        start = i * segment_length
        end = min(start + segment_length, num_total_frames)
        if start >= end:
            indices.append(num_total_frames - 1)
        else:
            indices.append(np.random.randint(start, end))
    return np.array(indices, dtype=np.int64)
