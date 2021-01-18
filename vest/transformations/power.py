import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

from vest.preprocess.embedding import embed2seq, embed


class PowerTransform(BaseEstimator, TransformerMixin):
    """ Power transformation
    """

    def __init__(self):
        self.model = None
        self.k = None
        self.lambda_ = None

    def fit(self, X: np.ndarray):
        """ Fitting the model

        :param X: An embedded time series
        :return: self
        """
        self.k = X.shape[1]
        x = embed2seq(X)

        x = x.reshape(-1, 1)

        model = PowerTransformer(method='yeo-johnson')
        model.fit(x)

        self.model = model
        self.lambda_ = self.model.lambdas_[0]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """ Transforming with a model

        :param X: An embedded time series
        :return: A transformed embedded time series
        """
        x = embed2seq(X)
        x = x.reshape(-1, 1)
        xt = self.model.transform(x)

        xt = embed(xt.flatten(), self.k)

        return xt
