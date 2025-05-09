from noize.cli.main import main as noize
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
    for i in range(1, 21, 5):
        noize(f"clips/{x}.wav", "outputs",
              output_filename=f"protected_{x}_c{i / 10}.wav",
              logs=True,
              cdpam_weight=i / 100,
              iterations=1000,
              log_file=f"logs/{x}_c{i / 10}.csv",
              target=f"voices/{y}.pth")
        
    for i in range(0, 201, 50):
        noize(f"clips/{x}.wav", "outputs",
              output_filename=f"protected_{x}_c{i}.wav",
              logs=True,
              cdpam_weight=i,
              iterations=1000,
              log_file=f"logs/{x}_c{i}.csv",
              target=f"voices/{y}.pth")
        
        

        