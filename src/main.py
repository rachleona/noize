import os
import re
import soundfile as sf
import typer

from ov_adapted import extract_se
from perturb import PerturbationGenerator
from utils import load_audio, split_audio

def main(filepath: str, 
         output_dir: str,
         perturbation_level: int = 5, 
         c_weight: int = 50, 
         d_weight: int = 2, 
         learning_rate: float = 0.02, 
         iterations: int = 500):
    
    perturber = PerturbationGenerator(
        perturbation_level / 1000, 
        c_weight, 
        d_weight, 
        learning_rate, 
        iterations)

    src_srs, src_tensor = load_audio(filepath, perturber)
    src_segments = split_audio(src_srs, perturber)

    # todo choose target speaker
    target_speaker = "testing/sof_01208_20s.wav"
    tgt_srs, tgt_tensor = load_audio(target_speaker, perturber)
    tgt_se = extract_se(tgt_tensor, perturber)

    p = perturber.generate_perturbations(src_segments, tgt_se, len(src_srs))
    target_audio_srs = src_srs + p

    filename = re.search('[\\w-]+?(?=\\.)', filepath).group(0)
    os.makedirs(output_dir, exist_ok=True)
    sf.write(f"{ output_dir }/protected_{ filename }.wav", target_audio_srs, perturber.hps.data.sampling_rate)

if __name__ == "__main__":
    typer.run(main)