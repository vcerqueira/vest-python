import numpy as np
import librosa
from typing import Optional


def mf_cepstral_coef(x: np.ndarray,
                     frequency: Optional[int] = None):
    """ Mel-frequency cepstral coefficients
    https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94390
    https://www.kaggle.com/ilu000/1-private-lb-kernel-lanl-lgbm

    :param x: a numeric sequence
    :param frequency: integer representing the frequency
    :return:
    """

    x = x.astype(float)

    if frequency is not None:
        mfcc = librosa.feature.mfcc(x, sr=frequency)
    else:
        mfcc = librosa.feature.mfcc(x)

    average_mfcc = mfcc.mean(axis=1)

    return average_mfcc
