import numpy as np


class CrossingPoints:
    """
    Number of times the time series crosses a point of interest
    todo add to summarisers
    """

    @staticmethod
    def crossing_points(x: np.ndarray, point_of_interest):
        x_leq = x <= point_of_interest

        p1 = x_leq[:(len(x) - 1)]
        p2 = x_leq[1:]
        cross = (p1 & (~p2)) | (p2 & (~p1))

        out = cross.sum()

        return out

    @staticmethod
    def median(x: np.ndarray):
        out = CrossingPoints().crossing_points(x, np.median(x))

        return out

    @staticmethod
    def zero(x: np.ndarray):
        out = CrossingPoints().crossing_points(x, 0)

        return out

    @staticmethod
    def mean(x: np.ndarray):
        out = CrossingPoints().crossing_points(x, np.mean(x))

        return out
