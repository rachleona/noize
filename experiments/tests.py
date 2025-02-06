from rachleona_noize.main import main as noize
import os

os.makedirs("logs", exist_ok=True)

pairs = [("sof_00295", "som_01523"),
         ("som_01523", "sof_00295"),
         ("sof_02121", "som_05223"),
         ("som_05223", "sof_02121"),
         ("sof_08886", "som_01208"),
         ("som_03502", "sof_04415")]

for x, y in pairs:
    for i in range(20,101,10):
        noize(f"clips/{x}.wav", "outputs",
                output_filename=f"protected_{x}_x{i}.wav",
                snr_weight=0.025,
                perturbation_norm_weight=0.25,
                frequency_weight=1.5,
                logs=True,
                log_file=f"logs/{x}_x{i}.csv",
                yourtts_weight=i,
                iterations=500,
                learning_rate=0.1,
                cdpam=False,
                yourtts=False,
                avc=False,
                freevc=False)