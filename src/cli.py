import librosa
import os
import typer

from audioread import NoBackendError
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt


def with_spinner(desc, func, *args):
    res = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=desc, total=None)
        res = func(*args)

    return res


def report_perturbation_start(segment_num):
    print(
        f":white_check_mark: [bold green]{ segment_num } audio segments[/] to process"
    )
    print("[bold magenta]Beginning perturbation calculation")


def get_audiofile(filepath, sampling_rate):
    while True:
        if filepath is None:
            filepath = Prompt.ask("[cyan bold]Path of audio file to be protected")

        if not check_file_exist(filepath, "audio"):
            continue

        try:
            src_srs, _ = librosa.load(filepath, sr=sampling_rate)
            return src_srs
        except NoBackendError:
            print(
                ":exclamation:[bold red] Error: Unrecognised format[/]\n[yellow]Please provide a valid audio file with an accepted format."
            )


def get_output_dir():
    return Prompt.ask("[cyan bold]Output directory")


def check_file_exist(path, name, stop=False):
    if not os.path.isfile(path):
        print(
            f":exclamation:[bold red] Error: {name} file not found[/]\n[yellow]Please make sure you have provided the correct path to the {name} file."
        )
        if stop:
            raise typer.Exit(1)
        else:
            return False
    return True


def report_config_error(err):
    # todo also give documentation?
    print(
        f":exclamation:[bold red] Error: {err.message}[/]\n[yellow]Please ensure your config file is valid and provide the correct path to the config file."
    )
    raise typer.Exit(2)


def report_perturbation_complete():
    print(":sparkles: [bold]Perturbation application complete[/] :sparkles:")


def warn(msg):
    print(f":warning-emoji: [bold yellow]{msg}")
