import numpy as np
import pywt


def denoise_signal_wave(x: np.ndarray,
                        wavelet='db4',
                        level=1,
                        thr=10):
    """ Denoising the signal with wavelets
    https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94390

    :param x: A numeric sequence
    :param wavelet: String denoting the wavelet to be used
    :param level: Level of transformation
    :param thr: Threshold for replacement--see reference
    :return: A transformed signal
    """
    coefficient_values = \
        pywt.wavedec(x, wavelet, mode="per", level=level)

    coefficient_values[1:] = \
        (pywt.threshold(i, value=thr, mode='hard')
         for i in coefficient_values[1:])

    denoised_x = pywt.waverec(coefficient_values, wavelet, mode='per')

    return denoised_x
