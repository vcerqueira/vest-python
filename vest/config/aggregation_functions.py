import numpy as np
from scipy import stats

from vest.aggregations.relative_dispersion import relative_dispersion
from vest.aggregations.slope import line_slope
from vest.aggregations.acceleration import acceleration
from vest.aggregations.poincare_variability import st_var, lt_var
from vest.aggregations.change_rates import ChangeRate
from vest.aggregations.correlation import correlation_dimension1, correlation_dimension2
from vest.aggregations.crossing import CrossingPoints
from vest.aggregations.entropy import EntropyFeatures
from vest.aggregations.gini import gini_index
from vest.aggregations.hjorth import HjorthFeatures
from vest.aggregations.fractal_dimension import FractalDimensionFeatures
from vest.aggregations.peaks import PeaksFeatures
from vest.aggregations.ljungbox import ljung_box_test
from vest.aggregations.autocorrelation import avg_acf, avg_pacf
from vest.aggregations.percentiles import p05, p95, p01, p99
from vest.aggregations.norm import norm
from vest.aggregations.lyapunov import mle
from vest.aggregations.hurst import hurst
from vest.aggregations.last_point import last_point
from vest.aggregations.fft import fft_amplitude
from vest.aggregations.stationarity import Stationarity

peaks_features = PeaksFeatures()
fractal_features = FractalDimensionFeatures()
hjorth_features = HjorthFeatures()
crossing_features = CrossingPoints()
changes_features = ChangeRate()
entropy_features = EntropyFeatures()

SUMMARY_OPERATIONS_ALL = \
    dict(
        mean=np.mean,
        median=np.median,
        max=np.max,
        sdev=np.std, # std dev
        amplitude=fft_amplitude,# Average amplitude of FFT
        var=np.var, # variance
        sum=np.sum,
        entropy=stats.entropy, # Shannon entropy
        sample_entropy=entropy_features.sample, # Sample Entropy
        wavelet_entropy=entropy_features.wavelet, # wavelet entropy
        hurst=hurst, # hurst exponent
        mle=mle, # maximum lyapunov exponent
        last_point=last_point, # last known observation
        skewness=stats.skew, # skewness
        kurtosis=stats.kurtosis, # kurtosis
        perc05=p05, # percentile 5
        perc01=p01, # percentile 1
        perc99=p99, # percentile 99
        perc95=p95, # percentile 95
        iqr=stats.iqr, # inter-quartile range
        norm=norm, # vector norm
        slope=line_slope, # slope
        acceleration=acceleration, # acceleration as Exponential MA / Simple MA
        st_var=st_var, # poincare short term variability
        lt_var=lt_var,  # poincare long term variability
        ljungbox=ljung_box_test, # ljung box test for auto-correlation
        aacf=avg_acf, # avg acf
        apacf=avg_pacf, # avg pacf
        rd=relative_dispersion, # relative dispersion as std dev divided by std dev of differenced series
        max_width=peaks_features.max_width, # check scipy docs https://github.com/vcerqueira/vest-python/blob/master/vest/aggregations/peaks.py
        max_prom=peaks_features.max_prominence, # check scipy docs https://github.com/vcerqueira/vest-python/blob/master/vest/aggregations/peaks.py
        peaks_dist=peaks_features.average_dist_between, # check scipy docs https://github.com/vcerqueira/vest-python/blob/master/vest/aggregations/peaks.py
        peaks_dist_alt=peaks_features.cwt_average_dist_between, # check scipy docs https://github.com/vcerqueira/vest-python/blob/master/vest/aggregations/peaks.py
        mean_abs_diff=changes_features.mean_abs_diff, # Mean absolute change between consecutive values
        mean_diff=changes_features.mean_diff, # Mean  change between consecutive values
        mean_2_der=changes_features.mean_2dc, # mean_second_derivative_central
        corr_dim1=correlation_dimension1, # Correlation Dimension with Embedding Dimension 1
        corr_dim2=correlation_dimension2, # Correlation Dimension with Embedding Dimension 2
        cross_median=crossing_features.median, # Number of times the time series crosses the median
        gini=gini_index, # gini index
        higuchi=fractal_features.higuchi, # fractal dimnesion feats. check eeglib https://github.com/vcerqueira/vest-python/blob/master/vest/aggregations/fractal_dimension.py
        hjorth_complexity=hjorth_features.complexity,# hjorth dimnesion feats. check eeglib https://github.com/vcerqueira/vest-python/blob/master/vest/aggregations/fractal_dimension.py
        hjorth_mobility=hjorth_features.mobility,# hjorth dimnesion feats. check eeglib https://github.com/vcerqueira/vest-python/blob/master/vest/aggregations/fractal_dimension.py
        st_adf=Stationarity.adf, # adf stationarity test
        st_level=Stationarity.level_stationarity, # kpss level stationarity test
        st_trend=Stationarity.trend_stationarity, # kpss trend stationarity test
        st_wv=Stationarity.wavelets, # statioanrity test based on wavelets
    )

SUMMARY_OPERATIONS_SMALL = \
    dict(
        mean=np.nanmean,
        median=np.nanmedian,
        max=np.nanmax,
        sdev=np.nanstd,
        var=np.nanvar,
        sum=np.nansum,
        skewness=stats.skew,
        kurtosis=stats.kurtosis,
        perc05=p05,
        perc95=p95,
        iqr=stats.iqr,
        slope=line_slope
    )

SUMMARY_OPERATIONS_FAST = \
    dict(
        mean=np.nanmean,
        median=np.nanmedian,
        max=np.nanmax,
        sdev=np.nanstd,
        amplitude=fft_amplitude,
        var=np.nanvar,
        sum=np.nansum,
        entropy=stats.entropy,
        last_point=last_point,
        skewness=stats.skew,
        kurtosis=stats.kurtosis,
        perc05=p05,
        perc01=p01,
        perc99=p99,
        perc95=p95,
        iqr=stats.iqr,
        norm=norm,
        slope=line_slope,
        st_var=st_var,
        lt_var=lt_var,
        aacf=avg_acf,
        apacf=avg_pacf,
        rd=relative_dispersion,
        max_width=peaks_features.max_width,
        max_prom=peaks_features.max_prominence,
        peaks_dist=peaks_features.average_dist_between,
        mean_abs_diff=changes_features.mean_abs_diff,
        mean_diff=changes_features.mean_diff,
        mean_2_der=changes_features.mean_2dc,
        cross_median=crossing_features.median,
        gini=gini_index,
        st_adf=Stationarity.adf,
        st_level=Stationarity.level_stationarity,
        st_trend=Stationarity.trend_stationarity,
        st_wv=Stationarity.wavelets,
    )
