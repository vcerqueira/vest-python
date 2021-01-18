import pandas as pd
import numpy as np


def SMA(x, n: int):
    """ Simple Moving Average

    :param x: a numeric sequence
    :param n: period for the moving average
    :return: np.ndarray
    """

    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    x_sma = x.rolling(n).mean()
    x_sma = np.array(x_sma)

    return x_sma
