import numpy as np
from numpy.linalg import LinAlgError


def norm(x: np.ndarray) -> float:
    """ Norm of vector

    :param x: 1-d numeric vector
    :return: numeric scalar
    """
    try:
        out = np.linalg.norm(x)
    except LinAlgError:
        out = np.nan

    return out
