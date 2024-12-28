import os
import re
import soundfile as sf
import typer
import warnings

from pathlib import Path
from perturb import PerturbationGenerator
from rich import print
from typing_extensions import Annotated
from utils import load_audio, split_audio, with_spinner

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.*")
def main(filepath: Annotated[Path, typer.Argument()] = None, 
         output_dir: Annotated[str, typer.Argument()] = None, 
         perturbation_level: int = 5, 
         c_weight: int = 50, 
         d_weight: int = 2, 
         learning_rate: float = 0.02, 
         iterations: int = 500):

    # todo error reporting

    perturber = with_spinner("Initialising perturbation generator...",
                             PerturbationGenerator,
                             perturbation_level / 1000,
                             c_weight,
                             d_weight,
                             learning_rate,
                             iterations)

    if filepath is None:
        filepath = typer.prompt("Path of audio file to be protected")

    if output_dir is None:
        output_dir = typer.prompt("Output directory")


    # todo file type check

    src_srs, _ = load_audio(filepath, perturber)
    src_segments = with_spinner("Loading audio file to be protected...",
                                split_audio, src_srs, perturber)
    print(f":white_check_mark: [bold green]{ len(src_segments) } audio segments[/] to process")


    print("[bold magenta]Beginning perturbation calculation")
    p = perturber.generate_perturbations(src_segments, len(src_srs))
    target_audio_srs = src_srs + p

    filename = re.search('[\\w-]+?(?=\\.)', os.path.basename(filepath)).group(0)
    os.makedirs(output_dir, exist_ok=True)
    res_filename = os.path.join(output_dir, f"protected_{ filename }.wav")
    
    with_spinner("Writing protected wav to output file...",
                 sf.write,
                 res_filename,
                 target_audio_srs,
                 perturber.hps.data.sampling_rate)
    
    print(":sparkles: [bold]Perturbation application complete[/] :sparkles:")

if __name__ == "__main__":
    typer.run(main)