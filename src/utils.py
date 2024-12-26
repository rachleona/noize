import librosa
import torch
from faster_whisper import WhisperModel

# adapted from split_audio_whisper from openvoice
# processes waveform tensors directly instead of using files
def split_audio(audio_srs, perturber):
    device = perturber.DEVICE
    sampling_rate = perturber.hps.data.sampling_rate
    sr_constant = sampling_rate / 1000

    if device == "cpu": 
        compute_type="float32"
    if device == "cuda": 
        compute_type="float16"

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
        audio_seg = audio_srs[start : end]

        if (len(audio_seg) == 0): continue

        seg_tensor = torch.FloatTensor(audio_seg).to(device)

        sr_constant = sampling_rate / 1000
        res.append({
            'start': start,
            'end': end,
            'tensor': seg_tensor,
            'id': k + 1
        })

        if k < len(segments) - 1:
            start = int(max(0, segments[k+1].start * 1000 - 80) * sr_constant)
    
    return res

# adapted from load_audio from CDPAM
# prepares audio waveform tensors directly to format required
def cdpam_prep(audio):
    audio = audio.to(torch.float64) * 32768
    audio = torch.reshape(audio, (-1, 1))
    shape = audio.shape
    audio = torch.reshape(audio, (1, shape[0]))
    return audio

def load_audio(filename, perturber):
    srs, _ = librosa.load(filename, sr=perturber.hps.data.sampling_rate)
    tensor = torch.from_numpy(srs).to(perturber.DEVICE)

    return srs, tensor
