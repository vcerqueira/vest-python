import numpy as np
import pandas as pd


def embed(sequence, k: int, return_df: bool = False):
    """ Time Delay Embedding without X-y partition

    :param return_df: Boolean, whether or not to return series as pd.DataFrame
    :param sequence: A 1-d numeric time series
    :param k: Embedding dimension
    :return: Embedded time series
    """
    X = []

    for i in range(len(sequence)):
        end_ix = i + k - 1
        if end_ix > len(sequence) - 1:
            break

        seq_x = sequence[i:(end_ix + 1)]
        X.append(seq_x)

    X = np.array(X)

    if return_df:
        col_names = ["t-" + str(i) for i in list(reversed(range(k)))]
        col_names[-1] = "t"

        X = pd.DataFrame(X, columns=col_names)

    return X


def embed2seq(sequence_mat: np.ndarray):
    """ Un-embed time series into sequence

    :param sequence_mat: embedded time series
    :return: time series as a sequence
    """

    sequence_part1 = sequence_mat[0][:-1]
    sequence_part2 = np.array([x[-1] for x in sequence_mat])

    sequence = np.concatenate((sequence_part1,
                               sequence_part2))

    return sequence
