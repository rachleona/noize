import json
import librosa
import os
import torch
import sys

from glob import glob
from faster_whisper import WhisperModel
from rachleona_noize.openvoice.utils import HParams


def block_print():
    sys.stdout = open(os.devnull, "w")


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


def split_audio(audio_srs, device, sampling_rate):
    """
    Uses whisper model to split audio clips into multiple segments
    Adapted from OpenVoice code to process audio data directly instead of from files

    Parameters
    ----------
    audio_srs : np.ndarray
        the time series representing the audio clip to apply protection to
    device : str
         device to be used for tensor computations (cpu or cuda)
    sampling_rate : int
        sampling rate of audio_srs

    Returns
    -------
    list of dict
        list of audio segment objects containing the following information about each segment
            1. start index in the original time series
            2. end index in the original time series
            3. the wave data of the audio segment itself as a tensor
            4. an id number
    """

    sr_constant = sampling_rate / 1000

    if device == "cpu":
        compute_type = "float32"
    if device == "cuda":
        compute_type = "float16"

    model = WhisperModel("medium", device=device, compute_type=compute_type)

    max_len = len(audio_srs)
    segments, _ = model.transcribe(audio_srs, beam_size=5, word_timestamps=True)
    segments = list(segments)

    start = None
    res = []

    for k, seg in enumerate(segments):
        # process with the time
        if k == 0:
            start = int(max(0, seg.start * 1000) * sr_constant)

        end = int(min(max_len, seg.end * 1000 + 80) * sr_constant)

        # left 0.08s for each audios
        audio_seg = audio_srs[start:end]
        audio_seg, ind = librosa.effects.trim(audio_seg)

        if len(audio_seg) == 0:
            continue

        seg_tensor = torch.FloatTensor(audio_seg).to(device)

        sr_constant = sampling_rate / 1000
        res.append(
            {
                "start": start + ind[0],
                "end": start + ind[1],
                "tensor": seg_tensor,
                "id": k + 1,
            }
        )

        if k < len(segments) - 1:
            start = int(max(0, segments[k + 1].start * 1000 - 80) * sr_constant)

    return res


def get_tgt_embs(target_id, pths_location, device):
    """
    Loads saved target voice embeddings

    Parameters
    ----------
    target_id : str
        id of the target voice to load, this will be the name of the directory containing all the saved tensors
    pths_location : Path
        path to the misc directory containing saved voices
    device : str
        device to load the tensors to

    Returns
    -------
    bool
        True only if d has all of *args as keys and False otherwise
    """
    voices_dir = os.path.join(pths_location, "voices", target_id, "")
    paths = glob(f"{ voices_dir }*.pth")

    # todo check target exists?

    target = {}
    for p in paths:
        name = os.path.basename(p)[:-4]
        emb = torch.load(
            p,
            map_location=torch.device(device),
            weights_only=True,
        )
        target[name] = emb.detach()

    return target


def dict_has_keys(d, *args):
    """
    Checks if a given dictionary has a specific list of keys

    Parameters
    ----------
    d : dict
        the dictionary to check for keys
    *args : list of str
        list of keys to check for

    Returns
    -------
    bool
        True only if d has all of *args as keys and False otherwise
    """
    if not isinstance(d, dict):
        return False

    for arg in args:
        if arg in d:
            continue
        else:
            return False
    return True


class ConfigError(Exception):
    """
    Custom exception for config related issues

    Attributes
    ----------
    message: str
    """

    def __init__(self, message, *args):
        super().__init__(*args)
        self.message = message


def get_hparams_from_file(config_path):
    """
    Extracts key configs from the file given at config_path
    Uses HParams class from OpenVoice code

    Parameters
    ----------
    config_path: str
        the file path to the config file

    Returns
    -------
    data_params: HParams
        a subset of configs for handling audio data e.g. sampling rate
    model_params: HParams
        a subset of configs for setting up the core OpenVoice synthesiser model
    misc_config: HParams
        object containing all other configs not covered in the above
    avc_enc_params: dict
        encoder parameters needed for initialising avc speaker encoder
    avc_hp: HParams
        configs used for preprocessing audio data for avc speaker encoder

    Raises
    ------
    ConfigError
        If the config file does not contain the expected data necessary for setting up the perturbation generator
    """

    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    if not dict_has_keys(config, "ov_data", "ov_model", "avc_encoder", "avc_hp"):
        # error
        raise ConfigError("Config file does not contain key sections")

    if not dict_has_keys(
        config["ov_data"],
        "sampling_rate",
        "filter_length",
        "hop_length",
        "win_length",
        "n_speakers",
    ):
        raise ConfigError("Data config is missing key entries")

    if not dict_has_keys(
        config["ov_model"],
        "zero_g",
        "inter_channels",
        "hidden_channels",
        "filter_channels",
        "n_heads",
        "n_layers",
        "kernel_size",
        "p_dropout",
        "resblock",
        "resblock_kernel_sizes",
        "resblock_dilation_sizes",
        "upsample_rates",
        "upsample_initial_channel",
        "upsample_kernel_sizes",
        "gin_channels",
    ):
        raise ConfigError("Model config is missing key entries")

    data_params = HParams(**config["ov_data"])
    model_params = HParams(**config["ov_model"])
    avc_hp = HParams(**config["avc_hp"])
    avc_enc_params = config["avc_encoder"]

    rest = {
        key: val
        for key, val in config.items()
        if key not in ["data", "model", "avc_encoder", "avc_hp"]
    }
    misc_config = HParams(**rest)

    return data_params, model_params, misc_config, avc_enc_params, avc_hp
