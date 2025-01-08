"""
This file contains all code that directly deals with the CLI e.g. printing error messages and prompts.
They have been isolated here to avoid cluttering the rest of the codebase
"""

import librosa
import os
import typer

from audioread import NoBackendError
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt


def with_spinner(desc, func, *args):
    """
    Calls the given function with a loading spinner with task descriptions
    Loading animation goes away once the function finishes running

    Parameters
    ----------
    desc : str
        the description of the current task to be shown alongside the spinner
    func : function
        the function to be called and to monitor progress for
    *args
        list of arguments to call func with

    Returns
    -------
    any
        the return of func
    """

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=desc, total=None)
        res = func(*args)

    return res


def report_perturbation_start(segment_num):
    """
    Prints CLI messages reporting the start of perturbation calculation

    Parameters
    ----------
    segment_num : int
        the number of audio segments to be processed in the overall perturbation calculation process
    """

    print(
        f":white_check_mark: [bold green]{ segment_num } audio segments[/] to process"
    )
    print("[bold magenta]Beginning perturbation calculation")


def get_audiofile(filepath, sampling_rate):
    """
    Loads the audio file at filepath and prompts the user for one if filepath is None
    Handles error reporting if file doesn't exist or is an unaccepted format

    Parameters
    ----------
    filepath : str or None
        the filepath given as argument when the CLI is called
    sampling_rate : int
        the sampling rate to load the audio file with

    Returns
    -------
    np.ndarray
        the audio waveform data as a numpy array
    """

    while filepath is None or not check_file_exist(filepath, "audio"):
        filepath = Prompt.ask("[cyan bold]Path of audio file to be protected")

    try:
        src_srs, _ = librosa.load(filepath, sr=sampling_rate)
        return src_srs
    except NoBackendError:
        print(
            ":exclamation:[bold red] Error: Unrecognised format[/]\n[yellow]Please provide a valid audio file with an accepted format."
        )


def get_output_dir():
    """
    Prompts for an output directory

    Returns
    -------
    str
        The path of the directory to put protected audio files in
    """
    return Prompt.ask("[cyan bold]Output directory")


def check_file_exist(path, name, stop=False):
    """
    Checks if the path given leads to an existing file
    Handles error reporting if the file does not exists and optionally terminates the CLI

    Parameters
    ----------
    path : str
        the path to check
    name : str
        an approriate descriptor of the file to be used in the error message
    stop : bool, optional
        whether to terminate the CLI is file doesn't exist

    Returns
    -------
    bool
        True if the path given leads to an existing file, False otherwise
    """

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
    """
    Prints out an error message associated with a ConfigError
    The CLI will then be terminated to allow user to correct their configurations

    Parameters
    ----------
    err : ConfigError
    """
    # todo also give documentation?
    print(
        f":exclamation:[bold red] Error: {err.message}[/]\n[yellow]Please ensure your config file is valid and provide the correct path to the config file."
    )
    raise typer.Exit(2)


def report_perturbation_complete():
    """
    Prints out success message for when perturbation application is complete
    """
    print(":sparkles: [bold]Perturbation application complete[/] :sparkles:")


def warn(msg):
    """
    Pretty prints a warning

    Parameters
    ----------
    msg : str
    """
    print(f":warning-emoji: [bold yellow]{msg}")
