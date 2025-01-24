import io
import sys
import torch

from rachleona_noize.ov_adapted import extract_se
from rachleona_noize.adaptive_voice_conversion.model import SpeakerEncoder as AvcEncoder
from rachleona_noize.freevc.speaker_encoder import SpeakerEncoder as FvcEncoder
from rachleona_noize.yourtts.compute_embeddings import compute_embeddings as ytts_emb
from TTS.api import TTS


class EncoderLoss:
    def __init__(self, src_emb, f, log_name, weight, logger):
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
    return EncoderLoss(
        src_se,
        lambda n: extract_se(n, perturber),
        "dist",
        perturber.DISTANCE_WEIGHT,
        perturber.logger,
    )


def generate_yourtts_loss(src, perturber):
    # suppress verbose output from model initialisation
    text_trap = io.StringIO()
    sys.stdout = text_trap

    tts = TTS(
        "tts_models/multilingual/multi-dataset/your_tts",
        gpu=(perturber.DEVICE != "cpu"),
    )

    # restore normal stdout
    sys.stdout = sys.__stdout__
    model = tts.synthesizer.tts_model.speaker_manager.encoder
    src_emb = ytts_emb(model, src)

    return EncoderLoss(
        src_emb,
        lambda n: ytts_emb(model, n),
        "yourtts",
        perturber.YOURTTS_WEIGHT,
        perturber.logger,
    )


def generate_freevc_loss(src, perturber):
    model = FvcEncoder(perturber.DEVICE, False)
    src_emb = model.embed_utterance(src, perturber.data_params.sampling_rate)

    return EncoderLoss(
        src_emb,
        lambda n: model.embed_utterance(n, perturber.data_params.sampling_rate),
        "freevc",
        perturber.FREEVC_WEIGHT,
        perturber.logger,
    )


def generate_avc_loss(src, perturber):
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
    src_emb = get_emb(src)

    return EncoderLoss(src_emb, get_emb, "avc", perturber.AVC_WEIGHT, perturber.logger)
