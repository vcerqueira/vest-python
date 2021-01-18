import numpy as np
from eeglib.features \
    import (hjorthActivity,
            hjorthMobility,
            hjorthComplexity)


class HjorthFeatures:
    """ Hjorth based features
    """

    @staticmethod
    def activity(x: np.ndarray):
        return hjorthActivity(x)

    @staticmethod
    def mobility(x: np.ndarray):
        return hjorthMobility(x)

    @staticmethod
    def complexity(x: np.ndarray):
        return hjorthComplexity(x)
