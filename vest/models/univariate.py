from typing import Dict
import numpy as np
import pandas as pd

from vest.models.base import VEST
from vest.preprocess.numeric import NumericPreprocess
from vest.models.operations import Operations
from vest.config.aggregation_functions import SUMMARY_OPERATIONS_ALL
from vest.utils import good_feats_indicator


class UnivariateVEST(VEST):
    """
    VEST: Vector of Statistics from Time Series
    An approach for systematic feature engineering using
    univariate time series data sets.

    -------


    UnivariateVEST is the class used to handle univariate time series.

    The main input X is assumed to be the embedding vectors representing a time series.
    The embedding vector are essentially the lags.

    -------

    1. Each embedding vector is mapped onto different representations,
    for example using moving averages to remove spurious noise;
    2. Each representation is summarised using statistics
    3. These statistics from different representations are concatenated to the auto-regressive
    attributes to improve forecasting performance.

    """

    def __init__(self):

        super().__init__()

        self.models = None
        self.preprocessor = None
        self.t_times = None
        self.a_times = None
        self.summary_operators = None
        self.apply_transform_operators = None

    def fit(self,
            X: np.ndarray,
            correlation_thr: float = 0.95,
            filter_by_correlation: bool = True,
            preprocess: bool = True,
            apply_transform_operators: bool = True,
            summary_operators: Dict = SUMMARY_OPERATIONS_ALL):
        """
        Fitting the feature engineering model.

        :param X: Array-like structure containing the embedding vectors (lags) representing the
        time series
        :param correlation_thr: Float - Correlation threshold for filtering correlated features
        :param filter_by_correlation: Boolean - Whether to filter out features by correlation
        :param preprocess: Boolean - Whether to create an imputation model for features
        :param apply_transform_operators: Boolean -
        :param summary_operators: Dict
        :return: self, with filled self.dynamics
        """

        self.X = X
        self.summary_operators = summary_operators
        self.apply_transform_operators = apply_transform_operators

        op = Operations()

        if self.apply_transform_operators:
            self.models = op.fit_transform_models(self.X)

        self.dynamics, self.transformers, self.aggregators, self.t_times, self.a_times = \
            op.run_summary_operations(X=X,
                                      transformation_models=self.models,
                                      apply_transform_operators=self.apply_transform_operators,
                                      summary_operators=self.summary_operators)

        if self.dynamics.shape[0] > 1:
            bool_to_keep = good_feats_indicator(self.dynamics)
            self.dynamics = self.dynamics.iloc[:, bool_to_keep]

        if filter_by_correlation:
            self.filter_dynamics(correlation_thr=correlation_thr)

        self.dynamics_names = list(self.dynamics.columns)

        if preprocess:
            self.preprocessor = NumericPreprocess()
            self.preprocessor.fit(self.dynamics)

            self.dynamics = self.preprocessor.transform(self.dynamics)
            self.dynamics = pd.DataFrame(self.dynamics, columns=self.dynamics_names)

        return self

    def transform(self, X: np.ndarray) -> pd.DataFrame:
        """
        Apply feature engineering model to new embedding vectors

        -----

        :param X: Array-like structure to retrieve features from
        :return: Feature set as pd.DataFrame
        """
        op = Operations()

        dynamics, _, _, _, _ = \
            op.run_summary_operations(X=X,
                                      transformation_models=self.models,
                                      apply_transform_operators=self.apply_transform_operators,
                                      summary_operators=self.summary_operators)

        dynamics = dynamics[self.dynamics_names]

        dynamics = self.preprocessor.transform(dynamics)
        dynamics = pd.DataFrame(dynamics, columns=self.dynamics_names)

        return dynamics
