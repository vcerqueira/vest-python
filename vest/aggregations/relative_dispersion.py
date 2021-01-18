import numpy as np


def relative_dispersion(x: np.ndarray) -> float:
    """ Relative dispersion of vector
    """
    out = np.std(x) / np.std(np.diff(x))

    return out
