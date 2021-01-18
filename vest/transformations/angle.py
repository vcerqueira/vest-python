import numpy as np


def vector_angle(x: np.ndarray):
    """ Get the point-wise angle of the vector
    :param x: a numeric sequence
    :return: a numeric sequence of angles
    """

    xt = np.angle(np.fft.rfft(x))

    return xt
