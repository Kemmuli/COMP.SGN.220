import math
import librosa as lb
import numpy as np
from numpy.random import default_rng
from utils import extract_mel_band_energies


def add_white_noise(data, noise_level):
    snr = 10*math.log10(1+noise_level)
    rms_data = np.sqrt(np.mean(data**2))
    rms_noise = math.sqrt(rms_data**2/(pow(10, snr/10)))
    noise = np.random.normal(0, rms_noise, data.shape[0])
    noisy_data = data + noise

    return noisy_data


def pitch_shift(data, sr, n_steps):
    return lb.effects.pitch_shift(y=np.array(data).astype(float), sr=sr, n_steps=float(n_steps))


def add_impulse_response(data, rir):
    return np.convolve(data, rir, 'same')


def spec_augment(data):
    mF = data.shape[0]
    # Randomly select range of mel frequency bins to be zeroed-out.
    rng = default_rng()
    f = rng.integers(low=0, high=40, endpoint=True)
    f0 = rng.integers(low=0, high=mF, endpoint=True)
    end = f0 + f
    # Map to zeroes
    data[f0:end] = 0

    return data


def main():
    pass


if __name__ == "__main__":
    main()
