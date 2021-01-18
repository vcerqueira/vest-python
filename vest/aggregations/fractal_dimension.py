import numpy as np
from eeglib.features import PFD, HFD


class FractalDimensionFeatures:
    """ Fractal Dimension Measures
    """
    @staticmethod
    def petrosian(x: np.ndarray):
        return PFD(x)

    @staticmethod
    def higuchi(x: np.ndarray):

        try:
            out = HFD(x)
        except (np.linalg.LinAlgError, ValueError) as e:
            out = np.nan

        return out
