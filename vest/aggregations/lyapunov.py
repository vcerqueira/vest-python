import numpy as np
import nolds


def mle(x: np.ndarray) -> float:
    """ Maximum Lyapunov Exponent
    :param x: 1-d numeric vector
    :return: numeric scalar
    """
    k = int(np.sqrt(len(x)))

    try:
        out = nolds.lyap_r(data=x,
                           emb_dim=k,
                           trajectory_len=k,
                           min_neighbors=k)
    except (ValueError, np.linalg.LinAlgError, AssertionError) as e:
        out = np.nan

    return out
