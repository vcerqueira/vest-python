import numpy as np
import nolds


def hurst(x: np.ndarray) -> float:
    """ Hurst exponent
    :param x: a 1-d numeric vector
    :return: numeric scalar
    """
    out = nolds.hurst_rs(x)

    return out
