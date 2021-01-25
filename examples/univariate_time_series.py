import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

from vest.preprocess.embedding import embed_with_target
from vest.models.univariate import UnivariateVEST

from vest.config.aggregation_functions \
    import (SUMMARY_OPERATIONS_ALL,
            SUMMARY_OPERATIONS_FAST,
            SUMMARY_OPERATIONS_SMALL)

series = np.arange(20)

k = 5

X, y = embed_with_target(series, k)

col_names = ["t-" + str(i) for i in list(reversed(range(k)))]
col_names[-1] = "t"
X = pd.DataFrame(X, columns=col_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    shuffle=False,
    random_state=42)

model = UnivariateVEST()

model.fit(X=X_train.values,
          correlation_thr=0.9,
          apply_transform_operators=True,
          summary_operators=SUMMARY_OPERATIONS_SMALL)

training_features = model.dynamics
X_tr_augmented = pd.concat([X_train, training_features], axis=1)

testing_features = model.transform(X_test.values)
testing_features.index = X_test.index
X_ts_augmented = pd.concat([X_test, testing_features], axis=1)

pred_model = Lasso()
pred_model.fit(X_tr_augmented, y_train)
yh_test = pred_model.predict(X_ts_augmented.values)
