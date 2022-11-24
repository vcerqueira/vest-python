from math import log, ceil, floor

import numpy as np
import pandas as pd
from pmdarima.arima import ndiffs
import rpy2.robjects as r_objects
from rpy2.robjects import pandas2ri


class Stationarity:
    """
    need R, and the following r packages: urca, forecast, locits

    import numpy as np
    import pandas as pd

    series = pd.Series(np.random.random(256))

    p_val = Stationarity.adf(series)
    # not stationary of p_val < 0.05

    is_not_stationary = Stationarity.wavelets(series)
    # 0 if stationary, 1 otherwise

    """

    @staticmethod
    def adf(series: pd.Series):
        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        pandas2ri.activate()

        series_rpy = pandas2ri.py2rpy_pandasseries(series)

        r_objects.r('''
                    run_urca <- function(x) {
                                    library(forecast)
                                    library(urca)

                                    x <- ts(x)

                                    nd <- ndiffs(x)
                                    if (nd>0) {
                                      x <- diff(x,differences=nd)
                                    }

                                    test <- ur.df(x,type="none",selectlags="AIC",lags=2)

                                    summ <- test@testreg
                                    fstat = summ$fstatistic

                                    p_value = pf(fstat[1L], fstat[2L], fstat[3L], lower.tail = FALSE)

                                    return(p_value)
                            }
                    ''')

        r_adf_func = r_objects.globalenv['run_urca']
        p_value = r_adf_func(series_rpy)
        p_value_ = p_value[0]
        is_not_st = int(p_value_ < 0.05)

        pandas2ri.deactivate()

        return is_not_st

    @staticmethod
    def closest_power_of_two(x):
        possible_results = floor(log(x, 2)), ceil(log(x, 2))

        closest_power = int(min(possible_results, key=lambda z: abs(x - 2 ** z)))

        return closest_power

    @staticmethod
    def goldfeldquant_partition(series: pd.Series,
                                partition_size: float):
        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        assert partition_size < 0.5

        n = series.shape[0]

        p1 = series.head(int(n * partition_size))
        p2 = series.tail(int(n * partition_size))

        return p1, p2

    @classmethod
    def wavelets(cls, series: pd.Series):
        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        pandas2ri.activate()

        p = cls.closest_power_of_two(series.shape[0])

        series_h = series.tail(2 ** p)

        series_rpy = pandas2ri.py2rpy_pandasseries(series_h)

        r_objects.r('''
                     run_wavel <- function(x) {
                            library(forecast)
                            library(locits)

                            x <- ts(x)

                            nd <- ndiffs(x)
                            if (nd>0) {
                              x <- diff(x,differences=nd)
                              x <- c(x[1], x)
                            }

                            capture.output(test <- hwtos2(as.vector(x)))

                            is_non_stationary <- as.integer(test$nreject > 0)

                            return(is_non_stationary)
                        }                    
                    ''')

        r_wavelet_func = r_objects.globalenv['run_wavel']
        is_not_stationary = r_wavelet_func(series_rpy)
        is_not_stationary_ = is_not_stationary[0]
        pandas2ri.deactivate()

        return is_not_stationary_

    @classmethod
    def trend_stationarity(cls,series: pd.Series):
        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        pandas2ri.activate()

        series_rpy = pandas2ri.py2rpy_pandasseries(series)

        r_objects.r('''
                       run_ndiffs <- function(x,test, test_type) {
                                library(forecast)

                                return(ndiffs(x, test=test,type=test_type))
                        }
                        ''')

        r_ndiffs_func = r_objects.globalenv['run_ndiffs']
        n_diffs = r_ndiffs_func(series_rpy, 'kpss', 'trend')
        n_diffs = int(n_diffs[0] > 0 )
        pandas2ri.deactivate()

        return n_diffs

    @classmethod
    def level_stationarity(cls, series: pd.Series):
        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        pandas2ri.activate()

        series_rpy = pandas2ri.py2rpy_pandasseries(series)

        r_objects.r('''
                           run_ndiffs <- function(x,test, test_type) {
                                    library(forecast)

                                    return(ndiffs(x, test=test,type=test_type))
                            }
                            ''')

        r_ndiffs_func = r_objects.globalenv['run_ndiffs']
        n_diffs = r_ndiffs_func(series_rpy, 'kpss', 'level')
        n_diffs = int(n_diffs[0] > 0)
        pandas2ri.deactivate()

        return n_diffs
