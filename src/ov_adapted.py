import torch

from openvoice.mel_processing import spectrogram_torch


def extract_se(audio_ref_tensor, perturber):
    """
    Extracts OpenVoice Tone Colour Embeddings from a given audio tensor
    Adapted from original OpenVoice code to process audio waveform data directly instead of from a file

    Parameters
    ----------
    audio_ref_tensor : torch.Tensor
        the audio waveform data in tensor form
    perturber : PerturbationGenerator
        PerturbationGenerator object containing the model and config needed for the extraction

    Returns
    -------
    torch.Tensor
        the tone colour embeddings
    """
    dps = perturber.data_params
    y = audio_ref_tensor.unsqueeze(0)
    y = spectrogram_torch(
        y,
        dps.filter_length,
        dps.hop_length,
        dps.win_length,
        perturber.hann_window,
        center=False,
    ).to(perturber.DEVICE)

    g = perturber.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

    return g


def convert(y, src_se, tgt_se, perturber):
    """
    Converts the voice in a given audio clip from src_se to tgt_se using the OpenVoice model
    Adapted from original OpenVoice code to process audio waveform data directly instead of from a file

    Parameters
    ----------
    y : torch.Tensor
        the audio waveform data of the clip to voice convert (in tensor form)
    src_se : torch.Tensor
        the tone colour embeddings of y
    tgt_se : torch.Tensor
        the tone colour embeddings of the target voice
    perturber : PerturbationGenerator
        PerturbationGenerator object containing the model and config needed for the voice conversion

    Returns
    -------
    torch.Tensor
        a new version of y with the target voice
    """
    dps = perturber.data_params
    device = perturber.DEVICE

    y = y.unsqueeze(0)
    spec = spectrogram_torch(
        y,
        dps.filter_length,
        dps.hop_length,
        dps.win_length,
        perturber.hann_window,
        center=False,
    ).to(device)
    spec_lengths = torch.LongTensor([spec.size(-1)]).to(device)
    audio = perturber.model.voice_conversion(
        spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=0.3
    )[0][0, 0].data.to(device)
    return audio
