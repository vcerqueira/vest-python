import numpy as np


def gini_index(x: np.ndarray):
    """ Gini index coefficient
    :param x: a 1-d numeric vector
    :return: numeric scalar
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    diff_sum = 0
    for i, xi in enumerate(x[:-1], 1):
        diff_sum += np.sum(np.abs(xi - x[i:]))

    out = diff_sum / (len(x) ** 2 * np.mean(x))

    return out
