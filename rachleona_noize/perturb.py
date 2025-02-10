import os
import torch

from pathlib import Path
from rachleona_noize.encoders import *
from rachleona_noize.logging import Logger
from rachleona_noize.quality import *
from rachleona_noize.openvoice.models import SynthesizerTrn
from rich.progress import track
from rachleona_noize.utils import (
    get_tgt_embs,
    get_hparams_from_file,
    ConfigError,
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
    CDPAM_WEIGHT : int
        the weight for the cdpam value between original clip and perturbed clip in the loss function
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
        cdpam,
        avc,
        freevc,
        xtts,
        perturbation_level,
        cdpam_weight,
        distance_weight,
        snr_wieght,
        perturbation_norm_weight,
        frequency_weight,
        avc_weight,
        freevc_weight,
        xtts_weight,
        learning_rate,
        iterations,
        logs,
        target,
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
        dirname = os.path.dirname(__file__)
        pths_location = Path(os.path.join(dirname, "misc"))

        # set up OpenVoice model
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

        # set up paths for weights and info needed for other models
        self.avc_ckpt = os.path.join(pths_location, "avc_model.ckpt")
        self.af_points = os.path.join(pths_location, "points.csv")

        # log total loss and openvoice embeddings difference by default
        log_values = ["loss", "dist"]
        # modules to include in loss function
        if cdpam:
            self.quality_func_generator = generate_cdpam_quality_func
            log_values.append("cdpam")
        else:
            self.quality_func_generator = generate_antifake_quality_func
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
        self.CDPAM_WEIGHT = cdpam_weight
        self.DISTANCE_WEIGHT = distance_weight
        self.SNR_WEIGHT = snr_wieght
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
        source_se : torch.Tensor
            OpenVoice tone colour embedding tensor extracted from the audio segment given
            Will be used in calculating starting parameter for loss minimisation
        """

        # choose a method of calculating quality term
        quality_func = self.quality_func_generator(src["tensor"], self)

        # initialise all necessary encoders for calculating distance loss
        loss_modules = []

        for f in self.loss_generators:
            loss_modules.append(f(src["tensor"], self))

        # combine all to create loss function to be used for this segment
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
            loss_f = self.generate_loss_function(segment)
            initial_params = torch.ones(segment["tensor"].shape).to(self.DEVICE)

            perturbation = self.minimize(loss_f, initial_params, segment["id"])
            padding = torch.nn.ZeroPad1d((segment["start"], max(0, l - segment["end"])))
            padded = padding(perturbation.detach())
            total_perturbation += padded[:l]

        return total_perturbation.cpu().detach().numpy()
