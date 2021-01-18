from scipy.stats import linregress
import numpy as np


def line_slope(x: np.ndarray) -> float:
    lm = linregress(x, list(range(len(x))))

    slope = lm[0]

    return slope
