import json
import torch

from openvoice.utils import HParams
from pathlib import Path
from faster_whisper import WhisperModel


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

        if len(audio_seg) == 0:
            continue

        seg_tensor = torch.FloatTensor(audio_seg).to(device)

        sr_constant = sampling_rate / 1000
        res.append({"start": start, "end": end, "tensor": seg_tensor, "id": k + 1})

        if k < len(segments) - 1:
            start = int(max(0, segments[k + 1].start * 1000 - 80) * sr_constant)

    return res


# adapted from load_audio from CDPAM
# prepares audio waveform tensors directly to format required
def cdpam_prep(audio):
    """
    Transform audio time series tensors into shape expected by CDPAM
    Adapted from CDPAM code to process audio wave data directly instead of through files

    Parameters
    ----------
    audio : torch.Tensor
        The audio tensor to be transformed

    Returns
    -------
    torch.Tensor
        The transformed tensor to be used in CDPAM value calculation
    """

    audio = audio.to(torch.float64) * 32768
    audio = torch.reshape(audio, (-1, 1))
    shape = audio.shape
    audio = torch.reshape(audio, (1, shape[0]))
    return audio


def choose_target(src_se, voices):
    """
    Chooses from a list of tone colour embedding tensors the one most different from src_se
    (Uses simple euclidean distance)

    Parameters
    ----------
    src_se : torch.Tensor
        source audio tensor to compare to
    voices : torch.Tensor
        list of voice embeddings available stacked into one tensor

    Returns
    -------
    torch.Tensor
        the tensor in the voices given that is furthest away from src_se in the feature space
    """
    diff = voices - src_se
    s = torch.sum(diff**2, 2)
    i = torch.argmax(s)
    return voices[i]


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
    pths_location: str
        the path to the directory where the OpenVoice checkpoint and target voice tensors are stored
    misc_config: HParams
        object containing all other configs not covered in the above

    Raises
    ------
    ConfigError
        If the config file does not contain the expected data necessary for setting up the perturbation generator
    """

    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    if not dict_has_keys(config, "data", "model", "pths_location"):
        # error
        raise ConfigError("Config file does not contain key sections")

    if not dict_has_keys(
        config["data"],
        "sampling_rate",
        "filter_length",
        "hop_length",
        "win_length",
        "n_speakers",
    ):
        raise ConfigError("Data config is missing key entries")

    if not dict_has_keys(
        config["model"],
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

    data_params = HParams(**config["data"])
    model_params = HParams(**config["model"])
    pths_location = Path(config["pths_location"])
    rest = {
        key: val
        for key, val in config.items()
        if key != "data" and key != "model" and key != "pths_location"
    }
    misc_config = HParams(**rest)

    if not pths_location.exists():
        return ConfigError("Constant tensors directory not found")

    return data_params, model_params, pths_location, misc_config
