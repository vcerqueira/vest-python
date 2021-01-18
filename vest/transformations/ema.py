import pandas as pd
import numpy as np


def EMA(x, n: int):
    """ Exponential Moving Average

    :param x: a numeric sequence
    :param n: period for the moving average
    :return: np.ndarray
    """

    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    x_ema = x.ewm(span=n).mean()
    x_ema = np.array(x_ema)

    return x_ema
