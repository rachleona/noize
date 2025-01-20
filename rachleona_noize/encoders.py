import os
import torch

from rachleona_noize.ov_adapted import extract_se
from rachleona_noize.adaptive_voice_conversion.model import AE as AvcEncoder
from rachleona_noize.adaptive_voice_conversion.utils import utt_make_frames, get_spectrograms

def generate_openvoice_loss(src_se, perturber):

    def f (new_tensor):
        euc_dist = torch.sum((src_se - extract_se(new_tensor, perturber)) ** 2)

        if perturber.logger is not None:
            perturber.logger("dist", euc_dist)
        
        return -perturber.DISTANCE_WEIGHT * euc_dist
    
    return f

def generate_yourtts_loss(src, perturber):

    def f(new_tensor):
        return
    
    return f

def generate_freevc_loss(src, perturber):

    def f(new_tensor):
        return
    
    return f

def generate_avc_loss(src, perturber):
    model = AvcEncoder(**perturber.avc_enc_params).to(perturber.DEVICE)
    model.load_state_dict(torch.load(os.path.join(perturber.pths_location, "vctk_model.ckpt")))

    src_mel, _ = get_spectrograms(src['tensor'], perturber.avc_hp, perturber.data_params.sampling_rate, perturber.DEVICE)
    x = utt_make_frames(src_mel, perturber.avc_hp.frame_size)
    src_emb = model.get_speaker_embeddings(x)
    
    def f(new_tensor):
        new_mel, _ = get_spectrograms(new_tensor, perturber.avc_hp, perturber.data_params.sampling_rate, perturber.DEVICE)
        x = utt_make_frames(new_mel, perturber.avc_hp.frame_size)
        new_emb = model.get_speaker_embeddings(x)
        euc_dist = torch.sum((src_emb - new_emb) ** 2)

        if perturber.logger is not None:
            perturber.logger("avc", euc_dist)
        
        return -perturber.AVC_WEIGHT * euc_dist
    
    return f