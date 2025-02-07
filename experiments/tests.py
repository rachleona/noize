from rachleona_noize.main import main as noize
import os
from glob import glob

os.makedirs("logs", exist_ok=True)

for f in glob("clips2/*.wav"):
    x = os.path.basename(f)[:-4]
    noize(f, "outputs",
            output_filename=f"protected_{x}_elu.wav",
            snr_weight=0.025,
            perturbation_norm_weight=0.25,
            frequency_weight=1.5,
            logs=True,
            log_file=f"logs/{x}_elu.csv",
            iterations=500,
            perturbation_level=5,
            learning_rate=0.1,
            cdpam=False,
            yourtts=False,
            avc=False,
            freevc=False)