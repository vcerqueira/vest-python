import numpy as np
import nolds


def correlation_dimension_n(x: np.ndarray, n):
    """ Correlation Dimension with Embedding Dimension n
    :param x: a 1-d numeric vector
    :param n: int denoting the embedding dimension
    :return: scalar feature
    """
    try:
        out = nolds.corr_dim(x, emb_dim=n)
    except AssertionError:
        out = -1

    return out


def correlation_dimension1(x: np.ndarray):
    """ Correlation Dimension with Embedding Dimension 1
    :param x: a 1-d numeric vector
    :return: scalar feature
    """
    return correlation_dimension_n(x, 1)


def correlation_dimension2(x: np.ndarray):
    """ Correlation Dimension with Embedding Dimension 2
    :param x: a 1-d numeric vector
    :return: scalar feature
    """
    return correlation_dimension_n(x, 2)
