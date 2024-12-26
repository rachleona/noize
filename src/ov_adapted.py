import torch

def extract_se(audio_ref_tensor, perturber):
    hps = perturber.hps
    y = audio_ref_tensor.unsqueeze(0)
    y = spectrogram_torch(y, 
                          hps.data.filter_length,
                          hps.data.hop_length, 
                          hps.data.win_length,
                          perturber.hann_window,
                          center=False).to(perturber.DEVICE)
    
    g = perturber.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

    return g

def convert(y, src_se, tgt_se, perturber):
    hps = perturber.hps
    device = perturber.DEVICE

    y = y.unsqueeze(0)
    spec = spectrogram_torch(y, hps.data.filter_length,
                            hps.data.hop_length, hps.data.win_length,
                            perturber.hann_window,
                            center=False).to(device)
    spec_lengths = torch.LongTensor([spec.size(-1)]).to(device)
    audio = perturber.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=0.3)[0][
                0, 0].data.to(device)
    return audio

# adapted from spectrogram_torch from openvoice
def spectrogram_torch(y, n_fft, hop_size, win_size, hann_window, center=False):
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec
