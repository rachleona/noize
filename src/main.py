import cli
import os
import re
import soundfile as sf
import typer
import warnings

from pathlib import Path
from perturb import PerturbationGenerator
from typing import Optional
from typing_extensions import Annotated
from utils import split_audio, ConfigError


# unclutter CLI by hiding warnings from libraries we have no direct control over
# comment during development
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.*",
)


def main(
    filepath: Annotated[Path, typer.Argument()] = None,
    output_dir: Annotated[str, typer.Argument()] = None,
    config_file: Annotated[Optional[Path], typer.Option()] = "config.json",
    output_filename: str = None,
    perturbation_level: int = 5,
    cdpam_weight: int = 50,
    distance_weight: int = 2,
    learning_rate: float = 0.02,
    iterations: int = 500,
):
    """
    Receives arguments and options provided in the CLI and calls the PerturbationGenerator
    Main function that ties the entire application together and handles final protected file output

    Parameters
    ----------
    filepath : str, optional
        the path to the audio file to apply protection for
        if not given, the user will be prompted in the CLI later for a value
    output_dir : str, optional
        the directory to put the final protected audio file in
        if not given, the user will be prompted in the CLI later for a value
    output_filename : str, optional
        the file name for the final output
        if not given, will default to "protected_<basename of filepath>"

    Other Parameters
    ----------
    The following parameters are maps directly to the arguments needed for initialising PerturbationGenerator
    Users are welcomed to tweak these if necessary but the default values are what we decided were best for general cases
    - config_file : str
    - perturbation_level : int (scaled before passing to PerturbationGenerator)
    - cdpam_weight : int
    - distance_weight : int
    - learning_rate : float
    - iterations : int
    """

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


if __name__ == "__main__":
    typer.run(main)
