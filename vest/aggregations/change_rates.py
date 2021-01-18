import numpy as np
from tsfresh.feature_extraction import feature_calculators as fc


class ChangeRate:
    """
    Change rate features from tsfresh
    """

    @staticmethod
    def mean_abs_diff(x: np.ndarray):
        """ Mean absolute change between consecutive values
        :param x: a 1-d numeric vector
        :return: scalar feature
        """

        return fc.mean_abs_change(x)

    @staticmethod
    def mean_diff(x: np.ndarray):
        """ Mean change between consecutive values
        :param x: a 1-d numeric vector
        :return: scalar feature
        """
        return fc.mean_change(x)

    @staticmethod
    def mean_2dc(x: np.ndarray):
        return fc.mean_second_derivative_central(x)
