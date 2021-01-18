import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, TransformerMixin

from vest.preprocess.embedding import embed2seq, embed


class KDE(BaseEstimator, TransformerMixin):
    """ Transformation based on Kernel Density Estimation
    """

    def __init__(self):
        self.model = None
        self.k = None

    def fit(self, X: np.ndarray):
        """ Fitting the model

        :param X: An embedded time series
        :return: self
        """
        self.k = X.shape[1]
        x = embed2seq(X)

        x = x.reshape(-1, 1)

        model = KernelDensity()
        model.fit(x)

        self.model = model

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """ Transform method

        :param X: An embedded time series
        :return: A transformed embedded time series
        """
        x = embed2seq(X)
        x = x.reshape(-1, 1)
        xt = self.model.score_samples(x)

        xt = embed(xt, self.k)

        return xt
