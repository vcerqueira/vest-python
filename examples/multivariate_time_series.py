import numpy as np
import pandas as pd

from vest.models.bivariate import BivariateVEST, multivariate_feature_extraction

from vest.config.aggregation_functions \
    import (SUMMARY_OPERATIONS_ALL,
            SUMMARY_OPERATIONS_FAST,
            SUMMARY_OPERATIONS_SMALL)

series = np.random.random((20, 3))
series = pd.DataFrame(series, columns=['x', 'y', 'z'])

model = BivariateVEST()

features = \
    model.extract_features(series,
                           pairwise_transformations=False,
                           summary_operators=SUMMARY_OPERATIONS_SMALL)

pd.DataFrame(features.items())

multivariate_feature_extraction(series,
                                apply_transform_operators=False,
                                summary_operators=SUMMARY_OPERATIONS_SMALL)
