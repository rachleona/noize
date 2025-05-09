from noize.cli.main import protect as noize
import os

os.makedirs("res", exist_ok=True)

pairs = [('sof_00610', 'som_05223'), ('sof_04310', 'som_05223'), ('som_09799', 'sof_08886'), ('som_07505', 'sof_08886')]
for x, y in pairs:
    for _ in range(10): 
        noize(f"clips2/{x}.wav", "outputs", 
                resource_log="res/resource_log_o.csv",
                target=y,
                avc=False,
                freevc=False,
                xtts=False)
        
        noize(f"clips2/{x}.wav", "outputs", 
                resource_log="res/resource_log_oa.csv",
                target=y,
                freevc=False,
                xtts=False)
        
        noize(f"clips2/{x}.wav", "outputs", 
                resource_log="res/resource_log_of.csv",
                target=y,
                avc=False,
                xtts=False)
        
        noize(f"clips2/{x}.wav", "outputs", 
                resource_log="res/resource_log_ox.csv",
                target=y,
                avc=False,
                freevc=False)
        
        noize(f"clips2/{x}.wav", "outputs", 
                resource_log="res/resource_log_oaf.csv",
                target=y,
                xtts=False)
        
        noize(f"clips2/{x}.wav", "outputs", 
                resource_log="res/resource_log_oax.csv",
                target=y,
                freevc=False)
        
        noize(f"clips2/{x}.wav", "outputs", 
                resource_log="res/resource_log_ofx.csv",
                target=y,
                avc=False)
        
        noize(f"clips2/{x}.wav", "outputs", 
                resource_log="res/resource_log_oafx.csv",
                target=y)