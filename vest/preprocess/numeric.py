from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


class NumericPreprocess(TransformerMixin, BaseEstimator):
    """ NA Imputation of numeric matrix

    """

    def __init__(self):
        self.X = None
        self.imputation = None

    def fit(self, X):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        self : object
            Returns self.
        """

        numeric_imputation = SimpleImputer(strategy='median')

        X_df = pd.DataFrame(X)
        X_df = X_df.replace([np.inf, -np.inf], np.nan)

        numeric_imputation.fit(X_df.values)

        self.imputation = numeric_imputation

        return self

    def transform(self, X):
        """
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """

        X_df = pd.DataFrame(X)
        X_df = X_df.replace([np.inf, -np.inf], np.nan)

        X = self.imputation.transform(X_df.values)

        return X
