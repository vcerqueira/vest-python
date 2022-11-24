import re

import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset

from vest.models.vector import VectorVEST

ds_name = 'nn5_daily_without_missing'

dataset = get_dataset(ds_name, regenerate=False)

train = list(dataset.train)

ds_list = [pd.Series(ds['target'], index=pd.date_range(start=ds['start'], periods=len(ds['target'])))
           for ds in train]

extraction_mod = VectorVEST()

features_list = []
for i, series in enumerate(ds_list):
    print(i)
    features = extraction_mod.extract_features(x=series.values,
                                               apply_transform_operators=False)

    features_list.append(features)

features_df = pd.concat(features_list, axis=0)
features_df.columns = [re.sub('identity_', '', x) for x in features_df.columns]

features_df.describe()

