import numpy as np


def diff1(x):
    """ First differences

    :param x: a numeric sequence
    :return: np.ndarray
    """

    xt = np.diff(x, n=1, prepend=[np.nan])

    return xt


def diff2(x):
    """ Second differences

    :param x: a numeric sequence
    :return: np.ndarray
    """

    xt = np.diff(x, n=2, prepend=[np.nan, np.nan])

    return xt
