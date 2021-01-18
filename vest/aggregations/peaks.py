import numpy as np
from scipy.signal import \
    (find_peaks,
     find_peaks_cwt,
     peak_prominences,
     peak_widths)


class PeaksFeatures:
    """
    A set of features based on peaks
    """

    @staticmethod
    def average_dist_between(x: np.ndarray):
        try:
            peaks, _ = find_peaks(x)

            if len(peaks) > 1:
                out = np.mean(np.diff(peaks))
            else:
                out = 0
        except ValueError:
            out = np.nan

        return out

    @staticmethod
    def max_width(x: np.ndarray):
        try:
            peaks, _ = find_peaks(x)

            if len(peaks) > 1:
                peak_width, _, _, _ = peak_widths(x, peaks)
                out = np.max(peak_width)
            else:
                out = 0
        except ValueError:
            out = np.nan

        return out

    @staticmethod
    def max_prominence(x: np.ndarray):
        try:
            peaks, _ = find_peaks(x)

            if len(peaks) > 1:
                prominence, _, _ = peak_prominences(x, peaks)
                out = np.max(prominence)
            else:
                out = 0
        except ValueError:
            out = np.nan

        return out

    @staticmethod
    def cwt_average_dist_between(x: np.ndarray):
        try:
            peaks, _ = find_peaks(x)

            if len(peaks) > 1:
                peak_width, _, _, _ = peak_widths(x, peaks)
                peaks_cwt_ = find_peaks_cwt(x, peak_width)
                if len(peaks_cwt_) > 1:
                    out = np.mean(np.diff(peaks_cwt_))
                else:
                    out = 0
            else:
                out = 0
        except ValueError:
            out = np.nan

        return out
