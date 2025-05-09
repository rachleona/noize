import atexit
import rachleona_noize.cli.cli as cli
import os
import re
import tracemalloc
import typer
import soundfile as sf
import warnings

from pathlib import Path
from rachleona_noize.perturb.perturb import PerturbationGenerator
from rachleona_noize.web.server import app as server
from rachleona_noize.web.queue import activate_worker, stop_worker
from typing import Optional
from typing_extensions import Annotated
from rachleona_noize.utils.utils import split_audio, ConfigError
import rachleona_noize.cli.voices as voices

app = typer.Typer(pretty_exceptions_enable=False)

# unclutter CLI by hiding warnings from libraries we have no direct control over
# comment during development
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.*",
)


# default constants
dirpath = os.path.dirname(os.path.dirname(__file__))
default_config_path = os.path.join(dirpath, "config.json")


@app.command()
def protect(
    filepath: Annotated[
        Path, typer.Argument(help=" path to the audio file to apply protection on")
    ] = None,
    output_dir: Annotated[
        str,
        typer.Argument(
            help="name of output directory, will be created if not already exists"
        ),
    ] = None,
    perturbation_level: Annotated[
        int,
        typer.Option(
            help="values between 1-10, controls how strong the noise applied is"
        ),
    ] = 5,
    target: Annotated[
        str, typer.Option(help="id of a saved voice for target-based optimisation")
    ] = None,
    config_file: Annotated[
        Optional[Path],
        typer.Option(help="path to the config file, use official repo one by default"),
    ] = default_config_path,
    output_filename: Annotated[
        str,
        typer.Option(
            help="name of protected file, uses protected_<filpath> by default",
            show_default=False,
        ),
    ] = None,
    avc: Annotated[
        bool,
        typer.Option(help="whether to use adaptVC encoder in perturbation calculation"),
    ] = True,
    freevc: Annotated[
        bool,
        typer.Option(help="whether to use FreeVC encoder in perturbation calculation"),
    ] = True,
    xtts: Annotated[
        bool,
        typer.Option(help="whether to use XTTS encoder in perturbation calculation"),
    ] = True,
    logs: Annotated[
        bool, typer.Option(help="whether to log intermediate loss values")
    ] = False,
    log_file: Annotated[
        str, typer.Option(help="name of log file is logs are requested")
    ] = "log.csv",
    resource_log: Annotated[
        Optional[Path],
        typer.Option(help="name of resource log file if logs are requested"),
    ] = None,
    learning_rate: Annotated[
        float, typer.Option(help="learning rate to use for optimisation process")
    ] = 0.02,
    iterations: Annotated[
        int, typer.Option(help="optimisation iterations (per audio segment)")
    ] = 300,
    distance_weight: Annotated[
        float, typer.Option(help="weight used in loss functions")
    ] = 2,
    snr_weight: Annotated[
        float, typer.Option(help="weight used in loss functions")
    ] = 0.025,
    perturbation_norm_weight: Annotated[
        float, typer.Option(help="weight used in loss functions")
    ] = 0.25,
    frequency_weight: Annotated[
        float, typer.Option(help="weight used in loss functions")
    ] = 1.5,
    avc_weight: Annotated[
        float, typer.Option(help="weight used in loss functions")
    ] = 25,
    freevc_weight: Annotated[
        float, typer.Option(help="weight used in loss functions")
    ] = 25,
    xtts_weight: Annotated[
        float, typer.Option(help="weight used in loss functions")
    ] = 25,
):
    """
    Apply adversarial perturbation to produce protected audio file
    """

    cli.check_file_exist(config_file, "config", True)

    try:
        perturber = cli.with_spinner(
            "Initialising perturbation generator...",
            PerturbationGenerator,
            config_file,
            avc,
            freevc,
            xtts,
            perturbation_level / 1000,
            distance_weight,
            snr_weight,
            perturbation_norm_weight,
            frequency_weight,
            avc_weight,
            freevc_weight,
            xtts_weight,
            learning_rate,
            iterations,
            logs,
            target,
            resource_log,
        )
    except ConfigError as err:
        cli.report_config_error(err)

    src_srs = cli.get_audiofile(filepath, perturber.data_params.sampling_rate)
    if output_dir is None:
        output_dir = cli.get_output_dir()

    if resource_log is not None:
        tracemalloc.start()
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

    if logs:
        perturber.logger.save(log_file)

    if output_filename == None:
        output_filename = re.search(
            "[\\w-]+?(?=\\.)", os.path.basename(filepath)
        ).group(0)
        filename = f"protected_{ output_filename }"
    else:
        filename = re.search(
            "[\\w-]+?(?=\\.)", os.path.basename(output_filename)
        ).group(0)

    os.makedirs(output_dir, exist_ok=True)
    res_filename = os.path.join(output_dir, f"{ filename }.wav")

    cli.with_spinner(
        "Writing protected wav to output file...",
        sf.write,
        res_filename,
        target_audio_srs,
        perturber.data_params.sampling_rate,
    )

    cli.report_operation_complete("Perturbation application complete")
    if resource_log is not None:
        tracemalloc.stop()


@app.command()
def web(
    port: Annotated[int, typer.Option(help="port for server to listen")] = 5000,
    debug: Annotated[
        bool, typer.Option(help="whether to start server in debug mode")
    ] = False,
):
    """
    Starts Flask server for web interface
    """

    activate_worker()
    atexit.register(stop_worker)
    server.run(port=port, debug=debug, host="0.0.0.0")


app.add_typer(
    voices.app,
    name="voices",
    help="Manage saved target voices to be used in protection application",
)
