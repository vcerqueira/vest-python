import numpy as np
from scipy import fft


def fft_amplitude(x: np.ndarray):
    """ Average amplitude of FFT
    :param x: a 1-d numeric vector
    :return: scalar feature
    """
    amplitude = np.abs(fft.fft(x) / len(x))

    average_amplitude = np.mean(amplitude)

    return average_amplitude
