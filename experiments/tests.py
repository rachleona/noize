from rachleona_noize.main import main as noize
import os
from glob import glob

os.makedirs("logs", exist_ok=True)

pairs = [('sof_00610', 'som_09799'), ('sof_04310', 'som_07505')]
for x, y in pairs:
    noize(f"clips2/{x}.wav", "outputs",
            output_filename=f"protected_{x}_tgt.wav",
            snr_weight=0.025,
            perturbation_norm_weight=0.25,
            frequency_weight=1.5,
            logs=True,
            log_file=f"logs/{x}_tgt.csv",
            iterations=500,
            perturbation_level=5,
            learning_rate=0.1,
            target=y,
            cdpam=False)

    noize(f"clips2/{y}.wav", "outputs",
            output_filename=f"protected_{y}_tgt.wav",
            snr_weight=0.025,
            perturbation_norm_weight=0.25,
            frequency_weight=1.5,
            logs=True,
            log_file=f"logs/{y}_tgt.csv",
            iterations=500,
            perturbation_level=5,
            learning_rate=0.1,
            target=x,
            cdpam=False)