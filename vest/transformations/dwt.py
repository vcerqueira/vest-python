import numpy as np
import pywt


def DWT(x: np.ndarray):
    """ Discrete Wavelet Transform

    :param x: a numeric sequence
    :return: np.ndarray
    """

    xt = pywt.downcoef(part="a",
                       data=x,
                       wavelet="db1",
                       level=1)

    return xt
