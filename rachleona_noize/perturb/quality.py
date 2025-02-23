import csv
import torch
import torchaudio


def generate_antifake_quality_func(
    points_file, sr, snr_weight, perturbation_norm_weight, frequency_weight, logger, device
):
    """
    generates function to calculate the quality term in perturbation generation loss
    combines magnitude of perturbation, signal-to-noise ratio and frequency filtering
    based on quality term used in AntiFake project

    Parameters
    ----------
    src : torch.Tensor
        source audio tensor to compare to
    perturber : PerturbationGenerator
        perturber to be used with the generated quality function

    Returns
    -------
    function (new_tensor: torch.Tensor, perturbation: torch.Tensor) -> torch.Tensor
        returns calculated quality difference between src and new_tensor
        (given new_tensor = src + perturbation)
    """
    xs = []
    ys = []
    with open(points_file) as file:
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
        spectrogram = torchaudio.transforms.Spectrogram().to(device)
        diff_spec = spectrogram(perturbation.unsqueeze(0))[0]

        # ys is scaled to 0-1 inversely, with originally large values close to 0, vice versa
        ys_scaled = [1 - (item + 20) / 100 for item in ys]

        # for each 201 windows, 201 bc fft window is defaulted to 400
        for i in range(0, diff_spec.shape[0]):
            # by the nyquist theorem, signal processing can only reach half of the sampling rate
            bin_freq = sr / 2 / 200

            # middle point at each bin
            probe_freq = (i + 0.5) * bin_freq

            # use linear interpolation
            for j, x in enumerate(xs):
                if xs[j] < probe_freq and xs[j + 1] > probe_freq:
                    weight_freq = ys_scaled[j] + (
                        (probe_freq - xs[j]) * (ys_scaled[j + 1] - ys_scaled[j])
                    ) / (xs[j + 1] - xs[j])

            diff_spec[i] *= weight_freq

        # sum up the loss, divide by the length
        quality_frequency = torch.sum(diff_spec) / len(diff_spec)

        if logger is not None:
            logger.log("snr", quality_snr)
            logger.log("p_norm", quality_l2_norm)
            logger.log("freq", quality_frequency)

        # aggregate loss
        return (
            snr_weight * quality_snr
            - perturbation_norm_weight * quality_l2_norm
            - frequency_weight * quality_frequency
        )

    return f
