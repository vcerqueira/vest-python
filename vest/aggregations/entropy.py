import numpy as np
import nolds
from tsfel.feature_extraction import wavelet_entropy


class EntropyFeatures:
    """
    Entropy based features
    """

    @staticmethod
    def sample(x: np.ndarray):
        """ Sample Entropy
        :param x: a 1-d numeric vector
        :return: scalar feature
        """

        out = nolds.sampen(x)

        if np.isinf(out):
            out = np.nan

        return out

    @staticmethod
    def wavelet(x: np.ndarray):
        """ Wavelet entropy
        :param x: a 1-d numeric vector
        :return: scalar feature
        """

        out = wavelet_entropy(x)

        return out
