import cdpam
from openvoice.utils import get_hparams_from_file
from openvoice.models import SynthesizerTrn
import torch
from utils import cdpam_prep
from ov_adapted import extract_se, convert


class PerturbationGenerator():
    def __init__(self, 
                 perturbation_level, 
                 cdpam_weight, 
                 distance_weight, 
                 learning_rate, 
                 iterations):
        
        # CDPAM_WEIGHT = 50
        # DISTANCE_WEIGHT = 2
        # LEARNING_RATE = 0.02
        # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # ITERATIONS = 500
        # PERTURBATION_LEVEL = 0.005

        ckpt_converter = 'src/openvoice/checkpoints/converter'
        self.hps = get_hparams_from_file(f'{ckpt_converter}/config.json')

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
        checkpoint_dict = torch.load(f'{ckpt_converter}/checkpoint.pth', map_location=torch.device(self.DEVICE))
        self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.hann_window = {}

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
        
    def minimize(self, function, initial_parameters):
        params = initial_parameters
        params.requires_grad_()
        optimizer = torch.optim.Adam([params], lr=self.LEARNING_RATE)

        for _ in range(self.ITERATIONS):
            optimizer.zero_grad()
            loss = function(params)
            loss.backward()
            optimizer.step()

        return params / torch.max(params) * self.PERTURBATION_LEVEL
        
    def generate_perturbations(self, src_segments, target_se, l):
        total_perturbation = torch.zeros(l).to(self.DEVICE)

        for segment in src_segments:
            loss_f, source_se = self.generate_loss_function(segment)
            target_segment = convert(segment['tensor'], source_se, target_se, self)
            padding = torch.nn.ZeroPad1d((0, len(segment['tensor']) - len(target_segment)))
            initial_params = padding(target_segment) - segment['tensor']
            # initial_params = 1e-5 * torch.ones(segment['tensor'].shape).to(DEVICE)
            
            perturbation = self.minimize(loss_f, initial_params)
            padding = torch.nn.ZeroPad1d((segment['start'], max(0, l - segment['end'])))
            padded = padding(perturbation.detach())
            total_perturbation += padded[:l]

        return total_perturbation.cpu().detach().numpy()

