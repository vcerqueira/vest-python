import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox


def ljung_box_test(x: np.ndarray, lags: int = 10) -> float:
    """ Ljung-box test

    :param x: 1-d numeric vector
    :param lags: Number of lags for the test
    :return:
    """
    if lags >= len(x):
        lags = int(len(x) / 2)

    test = \
        acorr_ljungbox(x,
                       lags=[lags],
                       return_df=True)

    p_value = test.values[0][1]

    return p_value
