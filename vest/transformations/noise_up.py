import numpy as np

NOISE_SCALE = 0.5


def noise_and_detrend(x: np.ndarray,
                      scale: float = NOISE_SCALE) -> np.ndarray:
    """ Add noise and remove trend
    https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94390

    :param scale: scale of the random normal generator
    :param x: numeric sequence
    :return: transformed numeric sequence
    """

    assert isinstance(x, np.ndarray)

    if len(x.shape) > 1:
        x = x.flatten()

    noise = np.random.normal(0, scale, len(x))

    x = x + noise
    x = x - np.median(x)

    return x
