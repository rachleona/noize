from rachleona_noize.main import main as noize
import os
from glob import glob

os.makedirs("logs", exist_ok=True)

pairs = [('sof_00610', 'som_01523'), ('sof_04310', 'som_01523'), ('som_09799', 'sof_08886'), ('som_07505', 'sof_08886')]
for x, y in pairs:
    noize(f"clips2/{x}.wav", "outputs",
            output_filename=f"protected_{x}_o.wav",
            snr_weight=0.025,
            perturbation_norm_weight=0.25,
            frequency_weight=1.5,
            logs=True,
            log_file=f"logs/{x}_o.csv",
            iterations=500,
            perturbation_level=5,
            target=y,
            cdpam=False,
            yourtts=False,
            avc=False,
            freevc=False,
            xtts=False)
    
    noize(f"clips2/{x}.wav", "outputs",
            output_filename=f"protected_{x}_ox.wav",
            snr_weight=0.025,
            perturbation_norm_weight=0.25,
            frequency_weight=1.5,
            logs=True,
            log_file=f"logs/{x}_ox.csv",
            iterations=500,
            perturbation_level=5,
            target=y,
            cdpam=False,
            yourtts=False,
            avc=False,
            freevc=False)
    
    noize(f"clips2/{x}.wav", "outputs",
            output_filename=f"protected_{x}_oa.wav",
            snr_weight=0.025,
            perturbation_norm_weight=0.25,
            frequency_weight=1.5,
            logs=True,
            log_file=f"logs/{x}_oa.csv",
            iterations=500,
            perturbation_level=5,
            target=y,
            cdpam=False,
            yourtts=False,
            freevc=False,
            xtts=False)
    
    noize(f"clips2/{x}.wav", "outputs",
            output_filename=f"protected_{x}_of.wav",
            snr_weight=0.025,
            perturbation_norm_weight=0.25,
            frequency_weight=1.5,
            logs=True,
            log_file=f"logs/{x}_of.csv",
            iterations=500,
            perturbation_level=5,
            target=y,
            cdpam=False,
            yourtts=False,
            avc=False,
            xtts=False)
    
    noize(f"clips2/{x}.wav", "outputs",
            output_filename=f"protected_{x}_oafx.wav",
            snr_weight=0.025,
            perturbation_norm_weight=0.25,
            frequency_weight=1.5,
            logs=True,
            log_file=f"logs/{x}_oafx.csv",
            iterations=500,
            perturbation_level=5,
            target=y,
            cdpam=False,
            yourtts=False)