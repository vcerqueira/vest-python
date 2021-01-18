import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import feature_selection

from vest.selection.correlation import filter_correlation


class VEST(BaseEstimator, TransformerMixin):
    """
    VEST: Vector of Statistics from Time Series
    An approach for systematic feature engineering using time series data sets.

    This is a general class.

    UnivariateVEST is the class object for univariate time series

    BivariateVEST is the class object for multivariate time series, in which feature engineering
    is carried out on pairs of variables

    VectorVEST extracts features from numeric vectors
    """

    def __init__(self):
        self.X = None
        self.aggregators = None
        self.transformers = None
        self.dynamics_names = None
        self.dynamics = None
        self.importance = None
        self.complete_stats = None

    def fit(self,
            X: np.ndarray,
            correlation_thr: float = 0.95,
            filter_by_correlation: bool = True,
            preprocess: bool = True,
            complete_stats: bool = True):

        pass

    def transform(self, X: np.ndarray) -> pd.DataFrame:

        pass

    def filter_dynamics(self, correlation_thr: float):
        """ Filtering bad dynamics

        :param correlation_thr: Correlation threshold
        :return: self
        """

        assert self.dynamics is not None

        ids_to_drop = filter_correlation(self.dynamics, thr=correlation_thr)

        if len(ids_to_drop) > 0:
            self.dynamics = self.dynamics.drop(ids_to_drop, axis=1, inplace=False)

        return self

    def compute_importance(self, y, return_values=False):
        """ Compute importance of dynamics with respect to y according to mutual information


        :param return_values:
        :param y: target variable
        :return: self
        """

        assert self.dynamics is not None

        importance, _ = feature_selection.f_regression(self.dynamics, y)
        self.importance = dict(zip(self.dynamics_names, importance))

        if return_values:
            return importance
