import cdpam
import os
import torch

from glob import glob
from openvoice.models import SynthesizerTrn
from openvoice.utils import get_hparams_from_file
from ov_adapted import extract_se, convert
from rich.progress import track
from utils import cdpam_prep, choose_target

class PerturbationGenerator():
    def __init__(self, 
                 perturbation_level, 
                 cdpam_weight, 
                 distance_weight, 
                 learning_rate, 
                 iterations):
        
        # todo: dynamic checkpoints folder? or config file?

        ckpt_converter = 'src/openvoice/checkpoints/converter'
        voice_bank_dir = 'checkpoints'
        self.hps = get_hparams_from_file(os.path.join(ckpt_converter, 'config.json'))

        self.CDPAM_WEIGHT = cdpam_weight
        self.DISTANCE_WEIGHT = distance_weight
        self.LEARNING_RATE = learning_rate
        self.PERTURBATION_LEVEL = perturbation_level
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.ITERATIONS = iterations
        
        self.model = SynthesizerTrn(
            len(getattr(self.hps, 'symbols', [])),
            self.hps.data.filter_length // 2 + 1,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        ).to(self.DEVICE)

        checkpoint_dict = torch.load(os.path.join(ckpt_converter, 'checkpoint.pth'), 
                                     map_location=torch.device(self.DEVICE), 
                                     weights_only=True)
        self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.hann_window = {}

        files = glob(voice_bank_dir)
        self.voices = torch.zeros(len(files), 1, 256, 1)
        for k,v in enumerate(files):
            se =torch.load(v, map_location=torch.device(self.DEVICE), weights_only=True)
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

