import numpy as np
import pandas as pd
from typing import Dict
import itertools
from collections import ChainMap

from vest.models.base import VEST
from vest.models.vector import VectorVEST
from vest.twod.operations import Operations2D
from vest.config.aggregation_functions import SUMMARY_OPERATIONS_SMALL

op = Operations2D()


class BivariateVEST(VEST):
    """ VEST for bivariate feature extraction.
    Features are extracted based on pairs of vectors (e.g. co-variance)
    todo work in progress
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def extract_features(X: pd.DataFrame,
                         pairwise_transformations: bool,
                         summary_operators: Dict = SUMMARY_OPERATIONS_SMALL):
        """

        :param X: A pd.DataFrame with multiple time series. Features are extracted on all pairs
        of variables
        :param pairwise_transformations: Pairwise transformations
        :param summary_operators: Dict
        available
        :return: A dict with features
        """

        col_comb = itertools.combinations(X.columns, 2)

        extractor = BivariateVEST()

        feature_set = []
        for v1, v2 in col_comb:
            comb_features = extractor.pairwise_extraction(X, v1, v2,
                                                          pairwise_transformations,
                                                          summary_operators)

            feature_set.append(comb_features)

        feature_set = dict(ChainMap(*feature_set))
        for k in feature_set:
            if np.isinf(feature_set[k]):
                feature_set[k] = np.nan

        return feature_set

    @staticmethod
    def pairwise_extraction(X: pd.DataFrame,
                            v1: str,
                            v2: str,
                            pairwise_transformations: bool,
                            summary_operators: Dict = SUMMARY_OPERATIONS_SMALL) -> Dict:
        """
        :param X: A pd.DataFrame with multiple time series.
        :param v1: string denoting a target variable
        :param v2: string denoting another target variable
        :param pairwise_transformations: Whether or not to carry out pairwise transformations.
        :param summary_operators: Dict
        available
        :return: A dict with features
        """
        x = X[v1].values
        y = X[v2].values

        na_return = False
        if all(np.isnan(x)) or all(np.isnan(y)):
            na_return = True
            x = np.random.random(20)
            y = np.random.random(20)

        xy_s = op.run_summary_operations(x, y)
        keys_t = [name + '_' + v1 + v2 for name in xy_s.keys()]
        xy_s = dict(zip(keys_t, xy_s.values()))

        if pairwise_transformations:
            xy_t = op.run_transform_operations(x, y)

            xy_t_feats = dict()
            for t in xy_t:
                xt = xy_t[t]
                for method in summary_operators:
                    feature_name = method + '_' + t + '_' + v1 + v2
                    xy_t_feats[feature_name] = summary_operators[method](xt)

                    if np.isinf(xy_t_feats[feature_name]):
                        xy_t_feats[feature_name] = np.nan

            xy_feature_set = {**xy_t_feats, **xy_s}
        else:
            xy_feature_set = xy_s

        if na_return:
            xy_feature_set = dict.fromkeys(xy_feature_set.keys(), np.nan)

        return xy_feature_set


def multivariate_feature_extraction(X: pd.DataFrame,
                                    apply_transform_operators: bool,
                                    summary_operators: Dict):
    """
    todo docs

    :param X:
    :param apply_transform_operators:
    :param summary_operators:
    :return:
    """
    vector_model = VectorVEST()

    features = []
    for col in X.columns:
        col_features = \
            vector_model.extract_features(x=X[col].values,
                                          apply_transform_operators=apply_transform_operators,
                                          summary_operators=summary_operators)

        col_names = [c.replace('identity', col) for c in col_features]
        col_features.columns = col_names
        features.append(col_features)

    features = pd.concat(features, axis=1)

    bivariate_model = BivariateVEST()
    pairwise_features = \
        bivariate_model.extract_features(X,
                                         pairwise_transformations=apply_transform_operators,
                                         summary_operators=summary_operators)

    pairwise_features = pd.DataFrame(pd.Series(pairwise_features)).T

    features = pd.concat([features, pairwise_features], axis=1)

    return features
