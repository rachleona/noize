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
    y = y - torch.cat((torch.zeros(1).to(device), hp.preemphasis * y[:-1]))

    # stft
    win_length = int(sr * hp.frame_length)
    linear = torch.stft(
        y,
        n_fft=hp.n_fft,
        hop_length=int(sr * hp.frame_shift),
        win_length=win_length,
        pad_mode="constant",
        window=torch.hann_window(win_length).to(device),
        return_complex=True,
    )
    # magnitude spectrogram
    mag = torch.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=hp.n_fft, n_mels=hp.n_mels
    )  # (n_mels, 1+n_fft//2)
    mel_basis = torch.from_numpy(mel_basis).to(device)
    mel = mel_basis @ mag  # (n_mels, t)

    # to decibel
    mel = 20 * torch.log10(torch.maximum(torch.FloatTensor([1e-5]).to(device), mel))
    mag = 20 * torch.log10(torch.maximum(torch.FloatTensor([1e-5]).to(device), mag))

    # normalize
    mel = torch.clamp((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = torch.clamp((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.to(torch.float32)  # (T, n_mels)
    mag = mag.T.to(torch.float32)  # (T, 1+n_fft//2)

    return mel, mag
