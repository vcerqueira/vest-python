from typing import List
import pandas as pd
import numpy as np


def filter_correlation(x: pd.DataFrame, thr: float) -> List:
    """ Filter based on correlation

    :param x: pd.DataFrame
    :param thr: correlation threshold
    :return: list with ids to drop
    """
    corr_matrix = x.corr().abs()

    upper_matrix = np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)

    upper = corr_matrix.where(upper_matrix)

    to_drop = [column for column in upper.columns
               if any(upper[column] > thr)]

    return to_drop
