import numpy as np


def falpine1(X):
    """ Alpine 1 objective function

    Arguments:

        X: `ndarray((nsamples, ndim))`.

    Returns:

        `ndarray(nsamples)`
    """
    return np.sum(np.abs(X*np.sin(X) + 0.1*X), 1)
