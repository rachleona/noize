import io
import sys
import torch

from rachleona_noize.ov_adapted import extract_se
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

    def __init__(self, src_emb, f, weight, log_name="", logger=None):
        self.src_emb = src_emb
        self.emb_f = f
        self.log_name = log_name
        self.weight = weight
        self.logger = logger

    def loss(self, new_tensor):
        new_emb = self.emb_f(new_tensor)
        euc_dist = torch.linalg.vector_norm(self.src_emb - new_emb)

        if self.logger is not None:
            self.logger.log(self.log_name, euc_dist)

        return -self.weight * euc_dist


def generate_openvoice_loss(src_se, perturber):
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
    return EncoderLoss(
        src_se,
        lambda n: extract_se(n, perturber),
        perturber.DISTANCE_WEIGHT,
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

    tts = TTS(
        "tts_models/multilingual/multi-dataset/your_tts"
    ).to(perturber.DEVICE)

    # restore normal stdout
    sys.stdout = sys.__stdout__
    model = tts.synthesizer.tts_model.speaker_manager.encoder
    src_emb = ytts_emb(model, src).detach()

    return EncoderLoss(
        src_emb,
        lambda n: ytts_emb(model, n),
        perturber.YOURTTS_WEIGHT,
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
    src_emb = model.embed_utterance(src, perturber.data_params.sampling_rate).detach()

    return EncoderLoss(
        src_emb,
        lambda n: model.embed_utterance(n, perturber.data_params.sampling_rate),
        perturber.FREEVC_WEIGHT,
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

    return EncoderLoss(src_emb, get_emb, perturber.AVC_WEIGHT, "avc", perturber.logger)
