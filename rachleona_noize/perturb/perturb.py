import csv
import os
import time
import torch
import tracemalloc

from pathlib import Path
from rachleona_noize.perturb.encoders import *
from rachleona_noize.utils.logging import Logger
from rachleona_noize.perturb.quality import generate_antifake_quality_func
from rich.progress import track
from rachleona_noize.utils.utils import (
    get_tgt_embs,
    get_hparams_from_file,
)


class PerturbationGenerator:
    """
    Central class containing details and methods needed to calculate perturbation

    ...

    Attributes
    ----------
    data_params : HParams
        a subset of configs for handling audio data e.g. sampling rate
    avc_enc_params: dict
        a subset of configs for setting up avc speaker encoder
    avc_hp: HParams
        a subset of configs for audio preprocessing needed by avc
    avc_ckpt: Path
        the path to the checkpoints for the avc speaker encoder model
    af_points: Path
        the path to data used in antifake quality term for frequency filtering
    hann_window: dict
        a dictionary for recording hann_window values used in OpenVoice spectrogram related calculations
    model : SynthesizerTrn
        OpenVoice core voice extraction and conversion model object to be used in loss function
    target : torch.Tensor or None
        the OpenVoice tone colour embedding corresponding to the target voice used to initialise perturbations
    voices : torch.Tensor
        matrix containing all saved voice tensors to be used in selecting initial parameters
    logger : Logger or None
        logger instance for recording loss values (and its components) every iteration
    DISTANCE_WEIGHT : int
        the weight for the distance between OpenVoice voice embeddings in the loss function
    SNR_WEIGHT : float
        the weight for the noise-to-ratio term in quality calculations
    PERTURBATION_NORM_WEIGHT : float
        the weight for the overall perturbation magnitude term in quality calculations
    FREQUENCY_WEIGHT : float
        the weight for the frequence filter term in quality calculations
    AVC_WEIGHT : float
        the weight of the distance between AVC voice embedings in the loss function
    FREEVC_WEIGHT : float
        the weight of the distance between freeVC voice embedings in the loss function
    XTTS_WEIGHT : float
        the weight of the distance between XTTS voice embedings in the loss function
    LEARNING_RATE : float
        learning rate for main loss function that optimises perturbation value
    PERTURBATION_LEVEL : float
        magnitude of perturbation allowed
    DEVICE : str
        device to be used for tensor computations (cpu or cuda)

    Methods
    -------
    generate_loss_function(src)
        Returns the appropriate loss function and voice embedding needed to calculate initial parameters used in perturbation calculation
        Varies based on source audio segment given

    minimize(function, initial_parameters, segment_num)
        Uses torch optimiser to minimise loss value based on loss function and initial parameters given
        Each step of the optimiser is tied to a progress tracker to be shown on the CLI

    generate_perturbations(src_segments, l)
        Given a list of audio segments and expected shape of the perturbation tensor,
        calls the two methods above to generate perturbation for each segment and unifies
        them into one numpy array/time series
    """

    def __init__(
        self,
        config_file,
        avc,
        freevc,
        xtts,
        perturbation_level,
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
    ):

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        data_params, model_params, misc, avc_enc_params, avc_hp = get_hparams_from_file(
            config_file
        )
        self.data_params = data_params
        self.avc_enc_params = avc_enc_params
        self.avc_hp = avc_hp
        self.hann_window = {}

        # misc folder location
        dirname = os.path.dirname(os.path.dirname(__file__))
        pths_location = Path(os.path.join(dirname, "misc"))

        # set up OpenVoice model
        self.model = init_ov(
            misc,
            data_params,
            model_params,
            self.DEVICE,
            os.path.join(pths_location, "checkpoint.pth"),
        )

        # set up paths for weights and info needed for other models
        self.avc_ckpt = os.path.join(pths_location, "avc_model.ckpt")
        self.xtts_config = os.path.join(pths_location, "xtts", "config.json")
        self.xtts_ckpt_dir = os.path.join(pths_location, "xtts", "")

        # log total loss and openvoice embeddings difference by default
        log_values = ["loss", "dist"]
        log_values.append("snr")
        log_values.append("freq")
        log_values.append("p_norm")

        self.loss_generators = [generate_openvoice_loss]
        if avc:
            self.loss_generators.append(generate_avc_loss)
            log_values.append("avc")
        if freevc:
            self.loss_generators.append(generate_freevc_loss)
            log_values.append("freevc")
        if xtts:
            self.loss_generators.append(generate_xtts_loss)
            log_values.append("xtts")

        if target is None:
            self.target = None
        else:
            self.target = get_tgt_embs(target, pths_location, self.DEVICE)

        # tunable weights
        self.DISTANCE_WEIGHT = distance_weight
        self.SNR_WEIGHT = snr_weight
        self.PERTURBATION_NORM_WEIGHT = perturbation_norm_weight
        self.FREQUENCY_WEIGHT = frequency_weight
        self.AVC_WEIGHT = avc_weight
        self.FREEVC_WEIGHT = freevc_weight
        self.XTTS_WEIGHT = xtts_weight
        self.LEARNING_RATE = learning_rate
        self.PERTURBATION_LEVEL = perturbation_level
        self.ITERATIONS = iterations

        # set up logging if asked for
        if logs:
            self.logger = Logger(*log_values)
        else:
            self.logger = None

        self.quality_func = generate_antifake_quality_func(
            os.path.join(pths_location, "points.csv"),
            data_params.sampling_rate,
            self.SNR_WEIGHT,
            self.PERTURBATION_NORM_WEIGHT,
            self.FREQUENCY_WEIGHT,
            self.logger,
            self.DEVICE,
        )

        if resource_log is not None:
            self.resource_log = {
                "filename": resource_log,
                "segment_time": [],
                "segment_mem": [],
            }
        else:
            self.resource_log = None

    def generate_loss_function(self, src):
        """
        Returns the appropriate loss function and voice embedding needed to calculate initial parameters used in perturbation calculation
        Varies based on source audio segment given

        Parameters
        ----------
        src : dict
            The audio segment object containing its start and end index in the original time series,
            the tensor representing the data within the segment and an id number

        Returns
        -------
        loss : function
            The loss function for calculating perturbation needed for protecting this audio segment
        """

        # initialise all necessary encoders for calculating distance loss
        loss_modules = []

        for f in self.loss_generators:
            loss_modules.append(f(src["tensor"], self))

        # combine all to create loss function to be used for this segment
        def loss(perturbation):
            scaled = perturbation / torch.max(perturbation) * self.PERTURBATION_LEVEL
            new_srs_tensor = src["tensor"] + scaled

            quality_term = self.quality_func(new_srs_tensor, scaled)
            dist_term = 0
            for m in loss_modules:
                dist_term += m.loss(new_srs_tensor)

            loss = dist_term + quality_term

            if self.logger is not None:
                self.logger.log("loss", loss)

            return loss

        return loss

    def minimize(self, function, initial_parameters, segment_id, track_func=None):
        """
        Uses torch optimiser to minimise loss value based on loss function and initial parameters given
        Each step of the optimiser is tied to a progress tracker to be shown on the CLI

        Parameters
        ----------
        function : function
            loss function to optimise over
        initial_parameters : torch.Tensor
            the initial values to start optimisation with
        segment_id : int
            the id of the current segment we are calculating perturbation for
            used in progress tracker to report overall progress
        track_func : function
            progress tracking function to use with optimisation loop

        Returns
        -------
        torch.Tensor
            the perturbation tensor to be added to the segment tensor to apply protection
        """

        params = initial_parameters
        params.requires_grad_()
        optimizer = torch.optim.Adam([params], lr=self.LEARNING_RATE)

        if track_func is None:
            track_func = lambda x, y: track(x, description=f"Processing segment {y}")

        for _ in track_func(range(self.ITERATIONS), segment_id):
            optimizer.zero_grad()
            loss = function(params)
            loss.backward()
            optimizer.step()

        return params / torch.max(params) * self.PERTURBATION_LEVEL

    def generate_perturbations(self, src_segments, l, track_func=None):
        """
        Given a list of audio segments and expected shape of the perturbation tensor,
        calls the two methods above to generate perturbation for each segment and unifies
        them into one numpy array/time series

        Parameters
        ----------
        src_segments : list
            A list of audio segment objects to produce perturbation for
        l : int
            size of the perturbation array to be returned
        track_func : function
            progress tracking function to use with optimisation loop

        Returns
        -------
        np.ndarray
            perturbation to be added to the original audio time series to protect against voice cloning
        """
        total_perturbation = torch.zeros(l).to(self.DEVICE)

        for segment in src_segments:

            if self.resource_log is not None:
                start = time.perf_counter()

            loss_f = self.generate_loss_function(segment)
            initial_params = torch.ones(segment["tensor"].shape).to(self.DEVICE)

            perturbation = self.minimize(
                loss_f, initial_params, segment["id"], track_func=track_func
            )
            padding = torch.nn.ZeroPad1d((segment["start"], max(0, l - segment["end"])))
            padded = padding(perturbation.detach())
            total_perturbation += padded[:l]

            if self.resource_log is not None:
                if self.DEVICE == "cpu":
                    peak_mem = tracemalloc.get_traced_memory()[1]
                    tracemalloc.reset_peak()
                else:
                    peak_mem = torch.cuda.max_memory_allocated()

                self.resource_log["segment_time"].append(time.perf_counter() - start)
                self.resource_log["segment_mem"].append(peak_mem)

        if self.resource_log is not None:
            with open(self.resource_log["filename"], "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for k in range(len(src_segments)):
                    writer.writerow(
                        [
                            self.resource_log["segment_time"][k],
                            self.resource_log["segment_mem"][k],
                        ]
                    )

        return total_perturbation.cpu().detach().numpy()
