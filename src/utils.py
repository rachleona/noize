import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment

# adapted from split_audio_whisper from openvoice
# processes waveform tensors directly instead of using files
def split_audio(audio_srs, perturber):
    device = perturber.DEVICE
    sampling_rate = perturber.hps.data.sampling_rate

    if device == "cpu": 
        compute_type="float32"
        np_type = np.float32
    if device == "cuda": 
        compute_type="float16"
        np_type = np.float16

    model = WhisperModel("medium", device=device, compute_type=compute_type)

    scaled_srs = np.array(audio_srs * (1<<15), dtype=np.int16)
    audio = AudioSegment(
        scaled_srs.tobytes(), 
        frame_rate=sampling_rate,
        sample_width=scaled_srs.dtype.itemsize, 
        channels=1
    )

    max_len = len(audio)
    segments, _ = model.transcribe(audio_srs, beam_size=5, word_timestamps=True)
    segments = list(segments)

    start_time = None
    res = []

    for k, seg in enumerate(segments):
        # process with the time
        if k == 0:
            start_time = max(0, int(seg.start * 1000))

        end_time = min(max_len, int(seg.end * 1000) + 80)

        # left 0.08s for each audios
        audio_seg = audio[start_time : end_time]

        if (len(audio_seg) == 0): continue

        channel_sounds = audio_seg.split_to_mono()
        samples = [s.get_array_of_samples() for s in channel_sounds]

        audio_seg = np.array(samples).T.astype(np_type)        
        audio_seg /= np.iinfo(samples[0].typecode).max
        audio_seg = audio_seg.reshape(-1)

        seg_tensor = torch.FloatTensor(audio_seg).to(device)

        sr_constant = sampling_rate / 1000
        res.append({
            'start': int(start_time * sr_constant),
            'end': int(end_time * sr_constant),
            'tensor': seg_tensor
        })

        if k < len(segments) - 1:
            start_time = max(0, int(segments[k+1].start * 1000 - 80))
    
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
