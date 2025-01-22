import cdpam
import csv
import os
import torch
import torchaudio

from rachleona_noize.utils import cdpam_prep


def generate_cdpam_quality_func(src, perturber):
    cdpam_loss = cdpam.CDPAM(dev=perturber.DEVICE)
    source_cdpam = cdpam_prep(src["tensor"])

    def f(new_tensor, perturbation):
        cdpam_val = cdpam_loss.forward(source_cdpam, cdpam_prep(new_tensor))

        if perturber.logger is not None:
            perturber.logger("cdpam", cdpam_val)

        return perturber.CDPAM_WEIGHT * cdpam_val

    return f


def generate_antifake_quality_func(perturber):
    xs = []
    ys = []
    with open(os.path.join(perturber.pths_location, "points.csv"), "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            xs.append(float(row[0]))
            ys.append(float(row[1]))

    def f(new_tensor, perturbation):
        # calculate quality norm
        quality_l2_norm = torch.linalg.vector_norm(perturbation)

        # calculate snr
        diff_waveform_squared = torch.square(perturbation)
        signal_power = torch.mean(torch.square(new_tensor))
        noise_power = torch.mean(diff_waveform_squared)
        quality_snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))

        # calculate frequency filter
        spectrogram = torchaudio.transforms.Spectrogram().cuda()
        diff_spec = spectrogram(perturbation.unsqueeze(0))[0]

        # ys is scaled to 0-1 inversely, with originally large values close to 0, vice versa
        ys_scaled = [1 - (item + 20) / 100 for item in ys]
        ys = ys_scaled

        # for each 201 windows, 201 bc fft window is defaulted to 400
        for i in range(0, diff_spec.shape[0]):
            # by the nyquist theorem, signal processing can only reach half of the sampling rate
            bin_freq = perturber.data_params.sampling_rate / 2 / 200

            # middle point at each bin
            probe_freq = (i + 0.5) * bin_freq

            # use linear interpolation
            for j, x in enumerate(xs):
                if xs[j] < probe_freq and xs[j + 1] > probe_freq:
                    weight_freq = ys[j] + (
                        (probe_freq - xs[j]) * (ys[j + 1] - ys[j])
                    ) / (xs[j + 1] - xs[j])

            diff_spec[i] *= weight_freq

        # sum up the loss, divide by the length
        quality_frequency = torch.sum(diff_spec) / len(diff_spec)

        if perturber.logger is not None:
            perturber.logger("snr", quality_snr)
            perturber.logger("p_norm", quality_l2_norm)
            perturber.logger("freq", quality_frequency)

        # aggregate loss
        return (
            perturber.SNR_WEIGHT * quality_snr
            - perturber.PERTURBATION_NORM_WEIGHT * quality_l2_norm
            - perturber.FREQUENCY_WEIGHT * quality_frequency
        )

    return f
