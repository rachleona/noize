import numpy as np
import torch


def compute_embeddings(
    encoder, x, num_frames=250, num_eval=10, return_mean=True, l2_norm=True
):
    """
    Generate embeddings for a batch of utterances
    x: 1xTxD
    """
    # map to the waveform size
    x = x.unsqueeze(0)
    if encoder.use_torch_spec:
        num_frames = num_frames * encoder.audio_config["hop_length"]

    max_len = x.shape[1]

    if max_len < num_frames:
        num_frames = max_len

    offsets = np.linspace(0, max_len - num_frames, num=num_eval)

    frames_batch = []
    for offset in offsets:
        offset = int(offset)
        end_offset = int(offset + num_frames)
        frames = x[:, offset:end_offset]
        frames_batch.append(frames)

    frames_batch = torch.cat(frames_batch, dim=0)
    embeddings = encoder.forward(frames_batch, l2_norm=l2_norm)

    if return_mean:
        embeddings = torch.mean(embeddings, dim=0, keepdim=True)
    return embeddings
