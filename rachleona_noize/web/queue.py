import librosa
import os
import re
import shutil
import soundfile as sf
import time
import uuid

from collections import deque
from rachleona_noize.perturb.perturb import PerturbationGenerator
from rachleona_noize.utils.utils import split_audio
from threading import Thread
from werkzeug.utils import secure_filename

task_queue = deque([])
in_progress = None
results = {}
worker_active = False

dirpath = os.path.dirname(os.path.dirname(__file__))
default_config_path = os.path.join(dirpath, "config.json")
src_path = os.path.join("/tmp", "noize_src")
outputs_path = os.path.join("/tmp", "noize_outputs")


def activate_worker():
    global worker_active
    worker_active = True
    os.makedirs(src_path, exist_ok=True)
    os.makedirs(outputs_path, exist_ok=True)
    thread = Thread(target=worker, daemon=True)
    thread.start()


def stop_worker():
    global worker_active
    global task_queue
    global results_list
    global in_progress

    worker_active = False
    task_queue.clear()
    in_progress = None
    results_list = {}
    shutil.rmtree(src_path)
    shutil.rmtree(outputs_path)


def worker():
    global task_queue
    global results_list
    global worker_active
    global in_progress

    while worker_active:
        if task_queue:
            job = task_queue[0]
            perturber = job.perturber
            src_srs, _ = librosa.load(
                job.filepath, sr=perturber.data_params.sampling_rate
            )
            src_segments = split_audio(
                src_srs, perturber.DEVICE, perturber.data_params.sampling_rate
            )
            job.src_srs = src_srs
            job.src_segments = src_segments
            job.current_seg = 1
            job.current_progress = 0

            job = task_queue.popleft()
            in_progress = job

            p = job.perturber.generate_perturbations(
                job.src_segments, len(job.src_srs), job.progress_tracker
            )
            target_audio_srs = job.src_srs + p
            output_filename = re.search(
                "[\\w-]+?(?=\\.)", os.path.basename(job.oriname)
            ).group(0)
            output_filename = os.path.join(
                outputs_path, f"protected_{ output_filename }.wav"
            )
            tmp_filename = os.path.join(outputs_path, job.id.hex + ".wav")

            sf.write(
                tmp_filename, target_audio_srs, job.perturber.data_params.sampling_rate
            )
            results[job.id.hex] = ProtectJobResult(job, output_filename)
            in_progress = None

            os.remove(job.filepath)

        time.sleep(0.1)


class ProtectJob:
    """
    Defines a protection application job instance

    ...

    Attributes
    ----------
    id : uuid
        unique id of the job
    perturbation_level: int
    target: str
    avc: bool
    freevc: bool
    xtts: bool
    iterations: int
        user input for perturber settings
    encoders: list[str]
        string list of encoders to be used on this job
    filepath: Path
        path where input file has been saved to
    oriname: str
        original file name of input file
    current_seg: int
        tracks which segment is currently being processed
    current_progress: int
        tracks progress of current segment

    Methods
    -------
    get_metadata()
        returns dictionary containing main job metadata like perturber connfig, filepath etc
    progress_tracker(r, seg_id)
        progress tracker function to be used with optimisation loop
    get_progress()
        returns dictionary containing progress details

    """

    def __init__(
        self, audio_file, perturbation_level, target, avc, freevc, xtts, iterations
    ):
        self.id = uuid.uuid4()
        self.perturbation_level = int(perturbation_level)
        self.target = target
        self.avc = avc == "on"
        self.freevc = freevc == "on"
        self.xtts = xtts == "on"
        self.iterations = int(iterations)
        self.encoders = ["OpenVoice"]

        if self.avc:
            self.encoders.append("AdaptVC")
        if self.freevc:
            self.encoders.append("FreeVC")
        if self.xtts:
            self.encoders.append("XTTS")

        perturber = PerturbationGenerator(
            default_config_path,
            self.avc,
            self.freevc,
            self.xtts,
            self.perturbation_level,
            target=self.target,
            iterations=self.iterations,
            learning_rate=0.02,
            distance_weight=2,
            snr_weight=0.025,
            perturbation_norm_weight=0.25,
            frequency_weight=1.5,
            avc_weight=25,
            freevc_weight=25,
            xtts_weight=25,
            logs=False,
            resource_log=None,
        )
        self.perturber = perturber
        self.filepath = os.path.join(src_path, self.id.hex + ".wav")
        audio_file.save(self.filepath)
        self.oriname = secure_filename(audio_file.filename)

    def get_metadata(self):
        return {
            "id": self.id.hex,
            "perturbation_level": self.perturbation_level,
            "iterations": self.iterations,
            "target": self.target,
            "src_file": self.oriname,
            "encoders": ", ".join(self.encoders),
        }

    def progress_tracker(self, r, seg_id):
        self.current_seg = seg_id
        for i in r:
            self.current_progress = i + 1
            yield i

    def get_progress(self):
        return {
            "job_id": self.id.hex,
            "nseg": len(self.src_segments),
            "current_seg": self.current_seg,
            "current_progress": self.current_progress,
            "iterations": self.iterations,
        }


class ProtectJobResult:
    """
    Result object containing data of finished jobs

    ...

    Attributes
    ----------
    job_id : uuid
        unique id of the job
    perturbation_level: int
    target: str
    encoders: list[str]
    iterations: int
        configuration used to run the job
    output_filename: str
        name of protected file

    Methods
    -------
    get_metadata()
        returns dictionary containing all attributes
    """

    def __init__(self, job, output_filename):
        self.job_id = job.id
        self.perturbation_level = job.perturbation_level
        self.target = job.target
        self.encoders = job.encoders
        self.iterations = job.iterations
        self.output = output_filename

    def get_metadata(self):
        return {
            "job_id": self.job_id.hex,
            "perturbation_level": self.perturbation_level,
            "iterations": self.iterations,
            "target": self.target,
            "output_filename": os.path.basename(self.output),
            "encoders": self.encoders,
        }
