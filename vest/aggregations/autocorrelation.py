import numpy as np
from statsmodels.tsa import stattools


def avg_acf(x: np.ndarray, n_lags: int = 10) -> float:
    """ Average ACF
    :param x: a 1-d numeric vector
    :param n_lags: Number of lags to compute feature
    :return: scalar feature
    """

    n_lags = np.min([n_lags, int(len(x) / 2) - 1])

    try:
        out = stattools.acf(x, nlags=n_lags)
        avg_corr = np.mean(out[1:(n_lags + 1)])
    except np.linalg.LinAlgError:
        avg_corr = np.nan

    return float(avg_corr)


def avg_pacf(x: np.ndarray, n_lags: int = 10) -> float:
    """ Average PACF
    :param x: a 1-d numeric vector
    :param n_lags: Number of lags to compute feature
    :return: scalar feature
    """

    n_lags = np.min([n_lags, int(len(x) / 2) - 1])

    try:
        out = stattools.pacf(x, nlags=n_lags)
        avg_corr = np.mean(out[1:(n_lags + 1)])
    except np.linalg.LinAlgError:
        avg_corr = np.nan

    return float(avg_corr)
