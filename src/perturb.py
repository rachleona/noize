import cdpam
import os
import torch

from cli import warn
from glob import glob
from openvoice.models import SynthesizerTrn
from ov_adapted import extract_se, convert
from rich.progress import track
from utils import cdpam_prep, choose_target, get_hparams_from_file, ConfigError

class PerturbationGenerator:
    def __init__(self,
                 config_file,
                 perturbation_level, 
                 cdpam_weight, 
                 distance_weight, 
                 learning_rate, 
                 iterations):
        
        data_params, model_params, pths_location, misc = get_hparams_from_file(config_file)
        self.data_params = data_params
        self.model_params = model_params
        self.pths_location = pths_location
        self.misc_config = misc

        self.CDPAM_WEIGHT = cdpam_weight
        self.DISTANCE_WEIGHT = distance_weight
        self.LEARNING_RATE = learning_rate
        self.PERTURBATION_LEVEL = perturbation_level
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.ITERATIONS = iterations
        
        self.model = SynthesizerTrn(
            len(getattr(misc, 'symbols', [])),
            data_params.filter_length // 2 + 1,
            n_speakers=data_params.n_speakers,
            **model_params,
        ).to(self.DEVICE)
        try: 
            checkpoint_dict = torch.load(os.path.join(pths_location, 'checkpoint.pth'), 
                                     map_location=torch.device(self.DEVICE), 
                                     weights_only=True)
        except FileNotFoundError:
            raise ConfigError('Cannot find checkpoint tensor in directory given in config file')
        
        self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.hann_window = {}

        voice_bank_dir = os.path.join(pths_location, "voices", "*.pth")
        files = glob(voice_bank_dir)

        if len(files) == 0:
            warn('No reference voice tensors found in tensors directory given in config file')
        
        self.voices = torch.zeros(max(1, len(files)), 1, 256, 1).to(self.DEVICE)
        for k,v in enumerate(files):
            se = torch.load(v, map_location=torch.device(self.DEVICE), weights_only=True)
            if not torch.is_tensor(se) or se.shape != torch.Size((1,256,1)):
                warn(f'Data loaded from {v} is not a valid voice tensor')
                continue
            self.voices[k] += se

    def generate_loss_function(self, src):
        cdpam_loss = cdpam.CDPAM(dev=self.DEVICE)
        source_se = extract_se(src['tensor'], self).detach()
        source_cdpam = cdpam_prep(source_se)

        def loss(perturbation):
            scaled = perturbation / torch.max(perturbation) * self.PERTURBATION_LEVEL
            new_srs_tensor = src['tensor'] + scaled
            new_se = extract_se(new_srs_tensor, self)
            new_cdpam = cdpam_prep(new_srs_tensor)
            euc_dist = torch.sum((source_se - new_se)**2)
            cdpam_val= cdpam_loss.forward(source_cdpam, new_cdpam)

            return -self.DISTANCE_WEIGHT * euc_dist + self.CDPAM_WEIGHT * cdpam_val

        return loss, source_se
        
    def minimize(self, function, initial_parameters, segment_num):
        params = initial_parameters
        params.requires_grad_()
        optimizer = torch.optim.Adam([params], lr=self.LEARNING_RATE)

        for _ in track(range(self.ITERATIONS), description=f"Processing segment {segment_num}"):
            optimizer.zero_grad()
            loss = function(params)
            loss.backward()
            optimizer.step()

        return params / torch.max(params) * self.PERTURBATION_LEVEL
        
    def generate_perturbations(self, src_segments, l):
        total_perturbation = torch.zeros(l).to(self.DEVICE)

        for segment in src_segments:
            loss_f, source_se = self.generate_loss_function(segment)
            target_se = choose_target(source_se, self.voices)
            target_segment = convert(segment['tensor'], source_se, target_se, self)
            padding = torch.nn.ZeroPad1d((0, len(segment['tensor']) - len(target_segment)))
            initial_params = padding(target_segment) - segment['tensor']
            
            perturbation = self.minimize(loss_f, initial_params, segment['id'])
            padding = torch.nn.ZeroPad1d((segment['start'], max(0, l - segment['end'])))
            padded = padding(perturbation.detach())
            total_perturbation += padded[:l]

        return total_perturbation.cpu().detach().numpy()

