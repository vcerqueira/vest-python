import numpy as np
import pandas as pd


def SMD(x, n: int):
    """ Simple Moving Deviation

    :param x: a numeric sequence
    :param n: period for the moving average
    :return: np.ndarray
    """

    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    try:
        rolling_sdev_values = x.rolling(n).std().dropna()
        rolling_sdev_values = rolling_sdev_values.values
    except ValueError:
        rolling_sdev_values = np.repeat(np.nan, repeats=len(x) - n + 1)

    return rolling_sdev_values
