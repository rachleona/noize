import librosa
import torch
import torch.nn.functional as F


def utt_make_frames(x, frame_size):
    remains = x.size(0) % frame_size
    if remains != 0:
        x = F.pad(x, (0, remains))
    out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
    return out


def get_spectrograms(y, hp, sr, device):
    # Preemphasis
    y = torch.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = torch.stft(
        y=y,
        n_fft=hp.n_fft,
        hop_length=int(sr * hp.frame_shift),
        win_length=int(sr * hp.frame_length),
        return_complex=True,
    )
    linear = torch.view_as_real(linear)

    # magnitude spectrogram
    mag = torch.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel_basis = torch.from_numpy(mel_basis).to(device)
    mel = torch.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * torch.log10(torch.maximum(1e-5, mel))
    mag = 20 * torch.log10(torch.maximum(1e-5, mag))

    # normalize
    mel = torch.clamp((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = torch.clamp((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.to(torch.float32)  # (T, n_mels)
    mag = mag.T.to(torch.float32)  # (T, 1+n_fft//2)

    return mel, mag
