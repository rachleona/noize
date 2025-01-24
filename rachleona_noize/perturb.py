import os
import torch

from glob import glob
from rachleona_noize.cli import warn
from rachleona_noize.encoders import *
from rachleona_noize.logging import Logger
from rachleona_noize.quality import *
from rachleona_noize.openvoice.models import SynthesizerTrn
from rachleona_noize.ov_adapted import *
from rich.progress import track
from rachleona_noize.utils import (
    choose_target,
    get_hparams_from_file,
    ConfigError,
)


class PerturbationGenerator:
    """
    Central class containing details and methods needed to calculate perturbation

    ...

    Attributes
    ----------
    data_params : dict
        a subset of configs for handling audio data e.g. sampling rate
    hann_window: dict
        a dictionary for recording hann_window values used in spectrogram related calculations
    model : SynthesizerTrn
        OpenVoice core voice extraction and conversion model object to be used in loss function
    voices : torch.Tensor
        matrix containing all saved voice tensors to be used in selecting initial parameters
    CDPAM_WEIGHT : int
        the weight for the cdpam value between original clip and perturbed clip in the loss function
    DISTANCE_WEIGHT : int
        the weight for the distance between voice embeddings in the loss function
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
        cdpam,
        avc,
        freevc,
        yourtts,
        perturbation_level,
        cdpam_weight,
        distance_weight,
        snr_wieght,
        perturbation_norm_weight,
        frequency_weight,
        avc_weight,
        freevc_weight,
        yourtts_weight,
        learning_rate,
        iterations,
        logs,
        target,
    ):

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        data_params, model_params, pths_location, misc, avc_enc_params, avc_hp = (
            get_hparams_from_file(config_file)
        )
        self.data_params = data_params
        self.avc_enc_params = avc_enc_params
        self.avc_hp = avc_hp
        self.hann_window = {}

        self.model = SynthesizerTrn(
            len(getattr(misc, "symbols", [])),
            data_params.filter_length // 2 + 1,
            n_speakers=data_params.n_speakers,
            **model_params,
        ).to(self.DEVICE)
        try:
            checkpoint_dict = torch.load(
                os.path.join(pths_location, "checkpoint.pth"),
                map_location=torch.device(self.DEVICE),
                weights_only=True,
            )
        except FileNotFoundError:
            raise ConfigError(
                "Cannot find checkpoint tensor in directory given in config file"
            )

        self.model.load_state_dict(checkpoint_dict["model"], strict=False)
        self.avc_ckpt = os.path.join(pths_location, "avc_model.ckpt")
        self.af_points = os.path.join(pths_location, "points.csv")

        voice_bank_dir = os.path.join(pths_location, "voices", "*.pth")
        files = glob(voice_bank_dir)

        if len(files) == 0:
            warn(
                "No reference voice tensors found in tensors directory given in config file"
            )

        if target is None:
            self.voices = torch.zeros(max(1, len(files)), 1, 256, 1).to(self.DEVICE)
            for k, v in enumerate(files):
                se = torch.load(
                    v, map_location=torch.device(self.DEVICE), weights_only=True
                )
                if not torch.is_tensor(se) or se.shape != torch.Size((1, 256, 1)):
                    warn(f"Data loaded from {v} is not a valid voice tensor")
                    continue
                self.voices[k] += se
            self.target = None
        else:
            self.target = torch.load(
                target, map_location=torch.device(self.DEVICE), weights_only=True
            )
        # modules to include in loss function
        self.CDPAM_QUALITY = cdpam
        self.AVC_LOSS = avc
        self.FREEVC_LOSS = freevc
        self.YOURTTS_LOSS = yourtts

        # tunable weights
        self.CDPAM_WEIGHT = cdpam_weight
        self.DISTANCE_WEIGHT = distance_weight
        self.SNR_WEIGHT = snr_wieght
        self.PERTURBATION_NORM_WEIGHT = perturbation_norm_weight
        self.FREQUENCY_WEIGHT = frequency_weight
        self.AVC_WEIGHT = avc_weight
        self.FREEVC_WEIGHT = freevc_weight
        self.YOURTTS_WEIGHT = yourtts_weight
        self.LEARNING_RATE = learning_rate
        self.PERTURBATION_LEVEL = perturbation_level
        self.ITERATIONS = iterations

        if logs:
            # log total loss and openvoice embeddings difference by default
            log_values = ["loss", "dist"]

            if avc:
                log_values.append("avc")
            if freevc:
                log_values.append("freevc")
            if yourtts:
                log_values.append("yourtts")
            if cdpam:
                log_values.append("cdpam")
            else:
                log_values.append("snr")
                log_values.append("freq")
                log_values.append("p_norm")

            self.logger = Logger(*log_values)
        else:
            self.logger = None

    def generate_loss_function(self, src, src_se):
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
        source_se : torch.Tensor
            OpenVoice tone colour embedding tensor extracted from the audio segment given
            Will be used in calculating starting parameter for loss minimisation
        """

        if self.CDPAM_QUALITY:
            quality_func = generate_cdpam_quality_func(src, self)
        else:
            quality_func = generate_antifake_quality_func(self)

        loss_modules = [generate_openvoice_loss(src_se, self)]

        if self.AVC_LOSS:
            generate_avc_loss(src["tensor"], self)
        if self.FREEVC_LOSS:
            generate_freevc_loss(src["tensor"], self)
        if self.YOURTTS_LOSS:
            generate_yourtts_loss(src["tensor"], self)

        def loss(perturbation):
            scaled = perturbation / torch.max(perturbation) * self.PERTURBATION_LEVEL
            new_srs_tensor = src["tensor"] + scaled

            quality_term = quality_func(new_srs_tensor, scaled)
            dist_term = 0
            for m in loss_modules:
                dist_term += m.loss(new_srs_tensor)

            loss = dist_term + quality_term

            if self.logger is not None:
                self.logger.log("loss", loss)

            return loss

        return loss

    def minimize(self, function, initial_parameters, segment_id):
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

        Returns
        -------
        torch.Tensor
            the perturbation tensor to be added to the segment tensor to apply protection
        """

        params = initial_parameters
        params.requires_grad_()
        optimizer = torch.optim.Adam([params], lr=self.LEARNING_RATE)

        for _ in track(
            range(self.ITERATIONS), description=f"Processing segment {segment_id}"
        ):
            optimizer.zero_grad()
            loss = function(params)
            loss.backward()
            optimizer.step()

        return params / torch.max(params) * self.PERTURBATION_LEVEL

    def generate_perturbations(self, src_segments, l):
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

        Returns
        -------
        np.ndarray
            perturbation to be added to the original audio time series to protect against voice cloning
        """
        total_perturbation = torch.zeros(l).to(self.DEVICE)

        for segment in src_segments:
            source_se = extract_se(segment["tensor"], self).detach()
            loss_f = self.generate_loss_function(segment, source_se)

            if self.target is None:
                target_se = choose_target(source_se, self.voices)
            else:
                target_se = self.target

            target_segment = convert(segment["tensor"], source_se, target_se, self)
            padding = torch.nn.ZeroPad1d(
                (0, len(segment["tensor"]) - len(target_segment))
            )
            initial_params = padding(target_segment) - segment["tensor"]

            perturbation = self.minimize(loss_f, initial_params, segment["id"])
            padding = torch.nn.ZeroPad1d((segment["start"], max(0, l - segment["end"])))
            padded = padding(perturbation.detach())
            total_perturbation += padded[:l]

        return total_perturbation.cpu().detach().numpy()
