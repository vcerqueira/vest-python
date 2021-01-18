import numpy as np


def st_var(x: np.ndarray) -> float:
    """ Short-term Poincare variability
    """
    return np.std(np.diff(x)) / np.sqrt(2)


def lt_var(x: np.ndarray) -> float:
    """ Long-term Poincare variability
    """
    st_var = np.std(np.diff(x)) / np.sqrt(2)
    lt_var = np.sqrt(2 * np.var(x) - st_var**2)

    return lt_var
