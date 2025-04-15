import librosa
import torch
import time
import csv
from faster_whisper import WhisperModel
from glob import glob

device = "cuda" if torch.cuda.is_available() else "cpu"

def split_audio(audio_srs, sampling_rate):
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

clips = glob("clips3/*.mp3")

for clip in clips:
    start = time.perf_counter()
    audio, sr = librosa.load(clip)
    segments = split_audio(audio, sr)
    time_taken = time.perf_counter() - start

    with open("segments.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([clip, len(segments), time_taken])