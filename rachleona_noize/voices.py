import librosa
import numpy as np
import os
import pygame
import shutil
import torch
import torchaudio
import typer

from pathlib import Path
from rachleona_noize.adaptive_voice_conversion.model import SpeakerEncoder as AvcEncoder
from rachleona_noize.cli import warn, with_spinner, report_operation_complete, check_file_exist
from rachleona_noize.encoders import xtts_get_emb, ov_extract_se, init_ov
from rachleona_noize.freevc.speaker_encoder import SpeakerEncoder as FvcEncoder
from rachleona_noize.utils import get_hparams_from_file
from rich import print
from time import sleep
from TTS.api import TTS
from typing import Optional
from typing_extensions import Annotated


app = typer.Typer()

dirpath = os.path.dirname(__file__)
default_config_path = os.path.join(dirpath, "config.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@app.command()
def list():
    """
    Lists the ids of all saved voices that can be used as perturbation targets, with descriptions
    """
    voices_path = os.path.join(dirpath, "misc", "voices")
    voices = os.listdir(voices_path)

    if (len(voices) == 0):
        warn("No saved voices found")
        raise typer.Exit(1)

    print(f":white_check_mark: [bold green]{ len(voices) } saved voices found")
    for v in voices:
        with open(os.path.join(voices_path, v, "desc.txt"), "r") as f:
            desc = f.read()
            print(f"[bold blue]{v}: [/]{desc}")

@app.command()
def play(id:str):
    """
    Plays 10s of the reference clip given for generating the voice embeddings of given id

    Parameters
    ----------
    id : str
        id of the saved voice to play
        a warning will be printed id given does not exist
    """
    try:
        audio_path = os.path.join(dirpath, "misc", "voices", id, "src.wav")
        audio, sr = librosa.load(audio_path)

        audio = audio[:sr * 10]
        audio *= 32768 / np.max(audio)
        audio = audio.astype(np.int16)

        pygame.mixer.pre_init(sr, channels=1)
        pygame.mixer.init()
        sound = pygame.sndarray.make_sound(audio)

        print(f":sound: [cyan]Playing example clip of voice { id }")
        sound.play()
        sleep(0.01)
        
    except FileNotFoundError:
        warn(f"No saved voice with id {str}")
    

@app.command()
def add(
    id: str,
    filepath: Annotated[Path, typer.Argument()] = None,
    desc: str = "",
    config_file: Annotated[Optional[Path], typer.Option()] = default_config_path,
    device: str = DEVICE
):
    """
    Add a new saved voice for use as perturbation calculation target

    Parameters
    ----------
    id : str
        id/name for the new voice
    filepath : Path
        path to the WAV file to be used as reference to extract embeddings from
    desc : str
        description string to be saved alongside the embeddings
    config_file : str
        path to the config file containing metadata and hyperparameters needed for voice models used
    device : str
        device to conduct tensor calculations on
    """
    check_file_exist(filepath, "voice reference", True)
    pths_location = Path(os.path.join(dirpath, "misc"))
    target_dir = os.path.join(pths_location, "voices", id)

    #todo check file is wav

    try:
        os.makedirs(target_dir)
    except FileExistsError:
        warn("A voice with this id already exists, please try again with a different name")
        raise typer.Exit(1)
    
    #todo max length for descriptions

    shutil.copyfile(filepath, os.path.join(target_dir, "src.wav"))
    with open(os.path.join(target_dir, "desc.txt"), "w") as f:
        f.write(desc)

    with_spinner(
        "Calculating audio embeddings",
        calculate_embs, 
        id,
        config_file, 
        device    
    )

    report_operation_complete(f"New voice embeddings saved under id { id }")

    
@app.command()
def reset(
    id: str,
    config_file: Annotated[Optional[Path], typer.Option()] = default_config_path,
    device: str = DEVICE
):
    """
    Recalculates embeddings for a saved voice

    Parameters
    ----------
    id : str
        id of the voice to calculate embeddings for
    config_file : str
        path to the config file containing metadata and hyperparameters needed for voice models used
    device : str
        device to conduct tensor calculations on
    """
    try:
        with_spinner(
            "Calculating audio embeddings",
            calculate_embs, 
            id,
            config_file, 
            device    
        )
        report_operation_complete(f"Voice embeddings recalculated for { id }")
    except FileNotFoundError:
        warn("No voice exists under this id, please use the [bold]add[/] command to add a new voice instead")


@app.command()
def resetall(
    config_file: Annotated[Optional[Path], typer.Option()] = default_config_path,
    device: str = DEVICE
):
    """
    Recalculates embeddings for all saved voices

    Parameters
    ----------
    config_file : str
        path to the config file containing metadata and hyperparameters needed for voice models used
    device : str
        device to conduct tensor calculations on
    """
    voices_path = os.path.join(dirpath, "misc", "voices")
    voices = os.listdir(voices_path)
    for v in voices:
        reset(v, config_file, device)

@app.command()
def delete(id:str):
    """
    Deletes a saved voice from voice library 

    Parameters
    ----------
    id : str
        id of the voice to delete
    """
    target_dir = os.path.join(dirpath, "misc", "voices", id)
    if os.path.isdir(target_dir):
        os.rmdir(target_dir)
        report_operation_complete(f"All voice embeddings under id { id } deleted successfully")
    else:
        warn(f"No saved voice with id { id }")



def calculate_embs(id, config_file, device):
    """
    Calculates embeddings for saved voice of a given id and saves them to voice library
    Assumes a src.wav file exists to extract embeddings from 

    Parameters
    ----------
    id : str
        id of the voice to calculate embeddings for
    config_file : str
        path to the config file containing metadata and hyperparameters needed for voice models used
    device : str
        device to conduct tensor calculations on
    """
    pths_location = Path(os.path.join(dirpath, "misc"))
    data_params, model_params, misc, avc_enc_params, avc_hp = get_hparams_from_file(
        config_file
    )

    # load audio 
    audio, sr = torchaudio.load(os.path.join(pths_location, "voices", id, "src.wav"))
    audio = torchaudio.functional.resample(audio, sr, data_params.sampling_rate)

    # initialise openvoice
    openvoice = init_ov(misc, data_params, model_params, device, os.path.join(pths_location, "checkpoint.pth"))
    
    # initialise avc
    avc = AvcEncoder(**avc_enc_params).to(device)
    avc.load_state_dict(
        torch.load(
            os.path.join(pths_location, "avc_model.ckpt"),
            map_location=torch.device(device),
            weights_only=True,
        )
    )

    # initialise xtts
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    xtts = tts.synthesizer.tts_model

    # initialise freevc
    fvc = FvcEncoder(device, False)

    # embeddings
    embs = {}
    embs['ov_embed'] = ov_extract_se(openvoice, audio, data_params.filter_length, data_params.hop_length, data_params.win_length, {}, device)
    embs['xtts_embed'] = xtts_get_emb(xtts, audio, data_params.filter_length)
    embs['avc_embed'] = avc.get_speaker_embeddings(audio, avc_hp, data_params.sampling_rate, device)
    embs['fvc_embed'] = fvc.embed_utterance(audio, data_params.sampling_rate)

    for emb in embs:
        torch.save(embs[emb], os.path.join(pths_location, "voices", id, emb + ".pth"))
    

    

