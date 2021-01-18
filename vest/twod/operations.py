from typing import Dict
import numpy as np
import pandas as pd
from scipy import signal
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy import stats
from statsmodels.tools.sm_exceptions import MissingDataError

from vest.utils import percentage_difference
from vest.twod.aggregations \
    import (co_integration,
            covariance,
            correlation)


class Operations2D:

    @staticmethod
    def run_summary_operations(x: np.ndarray, y: np.ndarray) -> Dict:
        kl_div = stats.entropy(x, y)
        js_div = jensenshannon(x, y, base=2)

        try:
            co_int = co_integration(x, y)
        except MissingDataError:
            co_int = np.nan

        covar = covariance(x, y)

        try:
            corr_p, _ = correlation(x, y, 'pearson')
        except ValueError:
            corr_p = np.nan

        corr_s, _ = correlation(x, y, 'spearman')
        corr_k, _ = correlation(x, y, 'kendall')

        feature_set = \
            dict(kl_div=kl_div,
                 js_div=js_div,
                 co_int=co_int,
                 covar=covar,
                 corr_p=corr_p,
                 corr_s=corr_s,
                 corr_k=corr_k)

        return feature_set

    @staticmethod
    def run_transform_operations(x: np.ndarray, y: np.ndarray) -> Dict:
        """

        :param x:
        :param y:
        :return:
        """

        if isinstance(x, pd.Series):
            x = x.values

        if isinstance(y, pd.Series):
            y = y.values

        p_diff = percentage_difference(x, y)
        cross_corr = signal.correlate(x, y)
        conv_x_y = signal.convolve(x, y)
        relative_entropy = rel_entr(x, y)

        if any(np.isnan(x)):
            x[np.isnan(x)] = np.nanmedian(x)

        if any(np.isnan(y)):
            y[np.isnan(y)] = np.nanmedian(y)

        xy_density, _, _ = np.histogram2d(x, y, normed=True)
        marginal_density = np.apply_along_axis(np.nanmean, axis=1, arr=xy_density)

        xy_transformed = \
            dict(pdiff=p_diff,
                 ccorr=cross_corr,
                 conv=conv_x_y,
                 density=marginal_density,
                 entropy=relative_entropy)

        return xy_transformed
