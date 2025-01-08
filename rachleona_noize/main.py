import rachleona_noize.cli as cli
import os
import re
import soundfile as sf
import typer
import warnings

from pathlib import Path
from rachleona_noize.perturb import PerturbationGenerator
from typing import Optional
from typing_extensions import Annotated
from rachleona_noize.utils import split_audio, ConfigError

app = typer.Typer()

# unclutter CLI by hiding warnings from libraries we have no direct control over
# comment during development
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.*",
)
dirpath = os.path.dirname(__file__)
default_config_path = os.path.join(dirpath, "..", "config.json")


@app.command()
def main(
    filepath: Annotated[Path, typer.Argument()] = None,
    output_dir: Annotated[str, typer.Argument()] = None,
    config_file: Annotated[Optional[Path], typer.Option()] = default_config_path,
    output_filename: str = None,
    perturbation_level: int = 5,
    cdpam_weight: int = 50,
    distance_weight: int = 2,
    learning_rate: float = 0.02,
    iterations: int = 500,
):

    cli.check_file_exist(config_file, "config", True)

    try:
        perturber = cli.with_spinner(
            "Initialising perturbation generator...",
            PerturbationGenerator,
            config_file,
            perturbation_level / 1000,
            cdpam_weight,
            distance_weight,
            learning_rate,
            iterations,
        )
    except ConfigError as err:
        cli.report_config_error(err)

    src_srs = cli.get_audiofile(filepath, perturber.data_params.sampling_rate)
    if output_dir is None:
        output_dir = cli.get_output_dir()

    src_segments = cli.with_spinner(
        "Loading audio file to be protected...",
        split_audio,
        src_srs,
        perturber.DEVICE,
        perturber.data_params.sampling_rate,
    )

    cli.report_perturbation_start(len(src_segments))

    p = perturber.generate_perturbations(src_segments, len(src_srs))
    target_audio_srs = src_srs + p

    if output_filename == None:
        output_filename = filepath
    filename = re.search("[\\w-]+?(?=\\.)", os.path.basename(output_filename)).group(0)

    os.makedirs(output_dir, exist_ok=True)
    res_filename = os.path.join(output_dir, f"protected_{ filename }.wav")

    cli.with_spinner(
        "Writing protected wav to output file...",
        sf.write,
        res_filename,
        target_audio_srs,
        perturber.data_params.sampling_rate,
    )

    cli.report_perturbation_complete()
