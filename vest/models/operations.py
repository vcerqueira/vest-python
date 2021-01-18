from typing import Dict, Optional
import numpy as np
import pandas as pd
import time
import warnings

from vest.utils import transform_within_embedding_vector
from vest.config.transformation_models import TRANSFORMATION_MODELS, TRANSFORMATION_MODELS_FAST
from vest.config.aggregation_functions \
    import (SUMMARY_OPERATIONS_ALL,
            SUMMARY_OPERATIONS_SMALL)
from vest.config.transformation_functions \
    import (TRANSFORMATIONS_ALL,
            TRANSFORMATIONS_FAST,
            N_PARAMETER)


class Operations:

    @staticmethod
    def run_summary_operations(X: np.ndarray,
                               transformation_models: Optional[Dict],
                               apply_transform_operators: bool = True,
                               summary_operators: Dict = SUMMARY_OPERATIONS_ALL,
                               ignore_warnings: bool = True):
        """ Series transformation and summarization

        :param summary_operators: Dict
        :param X: attribute variables (embedding vectors)
        :param transformation_models: dictionary with transformation models to be applied
        :param apply_transform_operators: Bool
        :param ignore_warnings: Whether or not to ignore warnings. Defaults to True
        :return: complete feature set of dynamics
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        op = Operations()

        if ignore_warnings:
            warnings.simplefilter("ignore")

        if apply_transform_operators:
            X_transformed, t_times = op.run_transform_operations(X)
            X_transformed_models = op.run_transform_models(transformation_models, X)

            X_transformed = {**X_transformed, **X_transformed_models}
        else:
            X_transformed, t_times = dict(identity=X), None

        transformers = list(X_transformed.keys())

        feature_set = []
        column_names = []
        a_times_list = []
        for t in X_transformed:
            xt = X_transformed[t]

            xt = \
                np.apply_along_axis(func1d=lambda z: z[~np.isnan(z)],
                                    arr=xt,
                                    axis=1)

            xt_feats = dict()
            a_times = dict()
            for method in summary_operators:
                print(method)
                start = time.time()
                xt_feats[method] = [summary_operators[method](z) for z in xt]

                delta = time.time() - start
                a_times[method] = delta

            xt_feats = pd.DataFrame(xt_feats)

            xt_feats.columns = str(t) + "_" + xt_feats.columns

            feature_set.append(xt_feats)
            column_names.append(list(xt_feats.columns))
            a_times_list.append(a_times)

        identity_feats = feature_set[0]

        features_all_names = list(identity_feats.columns)
        aggregators = \
            [feat.replace('identity_', '')
             for feat in features_all_names]

        feature_set = pd.concat(feature_set, axis=1, ignore_index=True)
        feature_set.columns = np.array(column_names).flatten()

        return feature_set, transformers, aggregators, t_times, a_times_list[0]

    @staticmethod
    def run_transform_operations(X: np.ndarray, n=None):
        """
        :param X: Attribute variables. In the case of univariate time series, these correspond to
        the embedding vectors after the application of time delay embedding
        :param n: Size of the window for some transformation functions. Defaults to sqrt(len(x))
        :return: Tuple with transformed vectors and respective execution times
        """
        dim = X.shape

        if n is None:
            n = int(np.sqrt(dim[1]))

        output_xt = dict(identity=X)
        times_xt = dict()
        for func in TRANSFORMATIONS_FAST:
            print(func)
            start = time.time()
            if func in N_PARAMETER:
                xt = transform_within_embedding_vector(X, TRANSFORMATIONS_FAST[func], n=n)
            else:
                xt = transform_within_embedding_vector(X, TRANSFORMATIONS_FAST[func])

            delta = time.time() - start

            output_xt[func] = xt
            times_xt[func] = delta

        for t in output_xt:
            assert output_xt[t].shape[0] == X.shape[0]

        return output_xt, times_xt

    @staticmethod
    def fit_transform_models(X):
        """ Fitting transformation models
        """
        models = dict()
        for k in TRANSFORMATION_MODELS_FAST:
            print(k)
            model = TRANSFORMATION_MODELS_FAST[k]()
            model.fit(X)

            models[k] = model

        return models

    @staticmethod
    def run_transform_models(models, X):
        """ Transform method for transformation models
        """
        assert models is not None
        assert len(models) > 0

        xt = dict()
        for k in models:
            xt[k] = models[k].transform(X)

        return xt
