from scipy.stats.mstats import winsorize
import numpy as np


def winsorise(x: np.ndarray,
              lower_limit: float = 0.1,
              upper_limit: float = 0.1):
    """ Winsorisation

    :param x: a numeric sequence
    :param lower_limit: float
    :param upper_limit: float
    :return: np.ndarray
    """

    limits = [lower_limit, upper_limit]

    x_wins = winsorize(x, limits=limits)
    x_wins = np.array(x_wins)

    return x_wins
