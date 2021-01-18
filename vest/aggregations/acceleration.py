import numpy as np

from vest.transformations.sma import SMA
from vest.transformations.ema import EMA


def acceleration(x: np.ndarray):
    """ Vector acceleration
    :param x: a 1-d numeric vector
    :return: scalar feature
    """
    n = int(np.sqrt(len(x)))

    div = EMA(x, n) / SMA(x, n)

    return np.nanmean(div)
