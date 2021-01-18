import numpy as np


def p05(x: np.ndarray) -> float:
    return np.percentile(x, 5)


def p95(x: np.ndarray) -> float:
    return np.percentile(x, 95)


def p01(x: np.ndarray) -> float:
    return np.percentile(x, 1)


def p99(x: np.ndarray) -> float:
    return np.percentile(x, 99)
