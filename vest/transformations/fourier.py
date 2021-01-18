import numpy as np


def fft_real(x: np.ndarray) -> np.ndarray:
    """ Real values of FFT transform
    :param x: a numeric sequence
    :return: a numeric sequence
    """
    x_fft = np.fft.fft(x)
    xt = np.real(x_fft)

    return xt


def fft_imaginary(x: np.ndarray) -> np.ndarray:
    """ Imaginary values of FFT transform
    :param x: a numeric sequence
    :return: a numeric sequence
    """
    x_fft = np.fft.fft(x)
    xt = np.imag(x_fft)

    return xt
