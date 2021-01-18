import numpy as np
import pandas as pd


def good_feats_indicator(x: pd.DataFrame,
                         na_thr: float = 0.67) -> np.ndarray:
    """ Preprocessing features
    Checking infinite values, unique values, and missing values

    :param x: Features as pd.DataFrame
    :param na_thr: threshold of NAs in a column for dropping the column. Otherwise, values are
    imputed
    :return: Boolean representing the features to keep
    """
    x = x.replace([np.inf, -np.inf], np.nan)

    na_percentage = x.isnull().sum(axis=0) / x.shape[0]
    too_many_na = na_percentage > na_thr

    unique_values = x.nunique()
    near_zero_var = unique_values < 2

    zipped_values = \
        zip(too_many_na.values,
            near_zero_var.values)

    good_feats_bool = [not (x or y) for x, y in zipped_values]

    return np.array(good_feats_bool)


def transform_within_embedding_vector(X, fun, **kwargs):
    """ Transforming an embedded time series

    :param X: Embedded time series
    :param fun: transformation function to be applied
    :param kwargs: ...
    :return: transformed embedded time series
    """

    xt = np.apply_along_axis(func1d=fun,
                             axis=1,
                             arr=X, **kwargs)

    return xt


def percentage_difference(x: np.ndarray, y: np.ndarray):
    assert len(x) == len(y)

    xd = (x - y) / np.abs(y)
    xd_percentage = xd * 100

    return xd_percentage
