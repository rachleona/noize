import io
import sys
import torch
import torchaudio

from rachleona_noize.openvoice.mel_processing import spectrogram_torch
from rachleona_noize.adaptive_voice_conversion.model import SpeakerEncoder as AvcEncoder
from rachleona_noize.freevc.speaker_encoder import SpeakerEncoder as FvcEncoder
from rachleona_noize.yourtts.compute_embeddings import compute_embeddings as ytts_emb
from TTS.api import TTS


class EncoderLoss:
    """
    Unifies interface for interacting with different encoders used in perturbation loss function

    ...

    Attributes
    ----------
    src_emb : torch.Tensor
        voice embedding from the source audio to be used in calculating embedding distance
    emb_f: function (torch.Tensor) -> torch.Tensor
        function for extracting the embedding of a given audio time series tensor
    weight: float
        the weight to multiply the calculated embedding distance with
    logger: Logger or None, default None
        logger for logging distance values
    log_name: str, default ""
        the key to the log value is logger is given

    Methods
    -------
    loss(new_tensor)
        takes an audio time series tensor and calculates the appropriate loss term
        uses simple euclidean distance

    """

    def __init__(self, src_emb, tgt_emb, f, weight, threshold, log_name="", logger=None):
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.emb_f = f
        self.log_name = log_name
        self.weight = weight
        self.logger = logger
        self.threshold = threshold

    def loss(self, new_tensor):
        new_emb = self.emb_f(new_tensor)
        euc_dist = torch.linalg.vector_norm(self.src_emb - new_emb)
        tgt_dist = 0 if self.tgt_emb is None else torch.linalg.vector_norm(self.tgt_emb - new_emb)
        elu = torch.nn.ELU()

        if self.logger is not None:
            self.logger.log(self.log_name, euc_dist)

        return self.weight * (elu(self.threshold - euc_dist) + tgt_dist)
    

def generate_openvoice_loss(src, perturber):
    """
    Generates EncoderLoss instance based on current source clip and perturber config
    Uses OpenVoice tone colour extractor

    Parameters
    ----------
    src_se : torch.Tensor
        OpenVoice tone colour embedding of the source clip
    perturber : PerturbationGenerator
        perturbation generator instance that is going to use the loss function

    Returns
    -------
    EncoderLoss
    """

    get_emb = lambda a: ov_extract_se(
        perturber.model,
        a,
        perturber.data_params.filter_length,
        perturber.data_params.hop_length,
        perturber.data_params.win_length,
        perturber.hann_window,
        perturber.DEVICE)
    
    src_emb = get_emb(src).detach()
    tgt_emb = None if perturber.target is None else perturber.target["ov_embed"]
    return EncoderLoss(
        src_emb,
        tgt_emb,
        get_emb,
        perturber.DISTANCE_WEIGHT,
        35,
        "dist",
        perturber.logger,
    )


def generate_yourtts_loss(src, perturber):
    """
    Generates EncoderLoss instance based on current source clip and perturber config
    Uses coqui ViTS speaker encoder
    (H/ASP speaker recognition model based on ResNet architecture)

    Parameters
    ----------
    src : torch.Tensor
        source audio time series
    perturber : PerturbationGenerator
        perturbation generator instance that is going to use the loss function

    Returns
    -------
    EncoderLoss
    """
    # suppress verbose output from model initialisation
    text_trap = io.StringIO()
    sys.stdout = text_trap

    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", gpu=perturber.DEVICE!="cpu")

    # restore normal stdout
    sys.stdout = sys.__stdout__
    model = tts.synthesizer.tts_model.speaker_manager.encoder 
    
    src_emb = ytts_emb(model, src).detach()
    tgt_emb = None if perturber.target is None else perturber.target['ytts_embed']
    return EncoderLoss(
        src_emb,
        tgt_emb,
        lambda n: ytts_emb(model, n),
        perturber.YOURTTS_WEIGHT,
        0.75,
        "yourtts",
        perturber.logger,
    )


def generate_freevc_loss(src, perturber):
    """
    Generates EncoderLoss instance based on current source clip and perturber config
    Uses coqui freeVC speaker encoder

    Parameters
    ----------
    src : torch.Tensor
        source audio time series
    perturber : PerturbationGenerator
        perturbation generator instance that is going to use the loss function

    Returns
    -------
    EncoderLoss
    """
    model = FvcEncoder(perturber.DEVICE, False)
    get_emb = lambda n: model.embed_utterance(n, perturber.data_params.sampling_rate)
    
    src_emb = get_emb(src).detach()
    tgt_emb = None if perturber.target is None else perturber.target['fvc_embed']
    return EncoderLoss(
        src_emb,
        tgt_emb,
        get_emb,
        perturber.FREEVC_WEIGHT,
        0.71,
        "freevc",
        perturber.logger,
    )


def generate_avc_loss(src, perturber):
    """
    Generates EncoderLoss instance based on current source clip and perturber config
    Uses speaker encoder from adaIN model

    Parameters
    ----------
    src : torch.Tensor
        source audio time series
    perturber : PerturbationGenerator
        perturbation generator instance that is going to use the loss function

    Returns
    -------
    EncoderLoss
    """
    model = AvcEncoder(**perturber.avc_enc_params).to(perturber.DEVICE)
    model.load_state_dict(
        torch.load(
            perturber.avc_ckpt,
            map_location=torch.device(perturber.DEVICE),
            weights_only=True,
        )
    )
    get_emb = lambda x: model.get_speaker_embeddings(
        x, perturber.avc_hp, perturber.data_params.sampling_rate, perturber.DEVICE
    )
    
    src_emb = get_emb(src).detach()
    tgt_emb = None if perturber.target is None else perturber.target['avc_embed']
    return EncoderLoss(src_emb, tgt_emb, get_emb, perturber.AVC_WEIGHT, 2, "avc", perturber.logger)


def generate_xtts_loss(src, perturber):
    """
    Generates EncoderLoss instance based on current source clip and perturber config
    Uses coqui ViTS speaker encoder
    (H/ASP speaker recognition model based on ResNet architecture)

    Parameters
    ----------
    src : torch.Tensor
        source audio time series
    perturber : PerturbationGenerator
        perturbation generator instance that is going to use the loss function

    Returns
    -------
    EncoderLoss
    """
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(perturber.DEVICE)

    model = tts.synthesizer.tts_model
    get_emb = lambda n: xtts_get_emb(model, n, perturber.data_params.sampling_rate)
    
    src_emb = get_emb(src).detach()
    tgt_emb = None if perturber.target is None else perturber.target['xtts_embed']
    return EncoderLoss(
        src_emb,
        tgt_emb,
        get_emb,
        perturber.YOURTTS_WEIGHT,
        1.2,
        "xtts",
        perturber.logger,
    )

def ov_extract_se(model, audio_ref_tensor, filter_length, hop_length, win_length, hann_window, device):
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
    y = audio_ref_tensor.unsqueeze(0)
    y = spectrogram_torch(
        y,
        filter_length,
        hop_length,
        win_length,
        hann_window,
        center=False,
    ).to(device)

    g = model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

    return g

def xtts_get_emb(model, audio, sr):
    audio = torch.unsqueeze(audio, 0)
    audio = audio[:, : sr * 30].to(model.device)
    
    audio_16k = torchaudio.functional.resample(audio, sr, 16000)
    return (
        model.hifigan_decoder.speaker_encoder.forward(audio_16k.to(model.device), l2_norm=True)
        .unsqueeze(-1)
        .to(model.device)
    )