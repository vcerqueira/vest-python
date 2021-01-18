from typing import Dict

import numpy as np
import pandas as pd

from vest.models.univariate import UnivariateVEST
from vest.config.aggregation_functions import SUMMARY_OPERATIONS_ALL


class VectorVEST(UnivariateVEST):

    def __init__(self):
        super().__init__()

        self.times = None
        self.a_times = None
        self.t_times = None

    def extract_features(self,
                         x: np.ndarray,
                         apply_transform_operators: bool = True,
                         summary_operators: Dict = SUMMARY_OPERATIONS_ALL):
        """ Get dynamics from a single vector

        :param x: one-dimensional array structure
        :param summary_operators: dict
        :param apply_transform_operators: bool
        :return: unfiltered dynamics of x
        """

        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values

        assert len(x.shape) == 1

        self.fit(x.reshape(1, -1),
                 filter_by_correlation=False,
                 preprocess=False,
                 apply_transform_operators=apply_transform_operators,
                 summary_operators=summary_operators)

        self.t_times = self.t_times
        self.a_times = self.a_times

        return self.dynamics
