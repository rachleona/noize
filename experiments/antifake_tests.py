from rachleona_noize.cli.main import main as noize
import os

os.makedirs("logs", exist_ok=True)

pairs = [("sof_00295", "som_01523"),
         ("som_01523", "sof_00295"),
         ("sof_02121", "som_05223"),
         ("som_05223", "sof_02121"),
         ("sof_08886", "som_01208"),
         ("som_03502", "som_04415")]

# cdpam weight
for x, y in pairs:
    for i in range(1,11):
        noize(f"clips/{x}.wav", "outputs",
                output_filename=f"protected_{x}_a{i}.wav",
                snr_weight=0.005 * i,
                perturbation_norm_weight=0.05 * i,
                frequency_weight=0.3 * i,
                logs=True,
                iterations=1000,
                log_file=f"logs/{x}_a{i}.csv",
                target=f"voices/{y}.pth",
                cdpam=False,
                yourtts=False,
                avc=False,
                freevc=False)