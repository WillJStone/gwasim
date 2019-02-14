import numpy as np
from scipy.stats import multivariate_normal
from cma.transformations import Rotation

def falpine1(X):
    """ Alpine 1 objective function

    Arguments:

        X: `ndarray((nsamples, ndim))`.

    Returns:

        `ndarray(nsamples)`
    """
    return np.sum(np.abs(X*np.sin(X) + 0.1*X), 1)

def falpine2(X):
    """ Alpline 2 objective function

    Arguments:

        X: 'ndarray((nsamples, ndim))'

    returns:

        'ndarray(nsamples)'
    """
    return np.prod(np.sqrt(X) * np.sin(X), 1)

def fDeflectedCorrugatedSpring(X, alpha = 5, K = 5):
    """ Deflected Corrugated Spring objective function

    Arguments:

        X: 'ndarray((nsamples, ndim))'

    returns:

        'ndarray(nsamples)'
    """
    X[X >= 2 * alpha] = 2 * alpha
    X[X <= 0] = 0

    s = np.sum((X - alpha)**2, 1)
    t = np.cos(K * np.sqrt(s))

    return 0.1*np.sum((X - alpha)**2, 1) - t

def fCigar(X):
    """ Cigar objective function

    Arguments:

        X: 'ndarray((nsamples, ndim))'

    returns:

        'ndarray(nsamples)'
    """
    return X[:,0]**2 + np.sum(X[:,1:]**2, 1)

def fRastrigin(X, A = 10):
    """ Rastrigin objective function

    Arguments:

        X: 'ndarray((nsamples, ndim))'

    returns:

        'ndarray(nsamples)'
    """
    n = X.shape[1]

    return A*n + np.sum(X**2 - A * np.cos(2*np.pi*X), 1)

def fYaoLiu9(X):
    """ YaoLiu09 objective function

    Arguments:

        X: 'ndarray((nsamples, ndim))'

    returns:

        'ndarray(nsamples)'
    """
    return np.sum(X**2 - 10*np.cos(2*np.pi*X) + 10, 1)

def fSphere(X):
    """ Sphere objective function

    Arguments:

        X: 'ndarray((nsamples, ndim))'

    returns:

        'ndarray(nsamples)'
    """
    return np.sum(X**2, axis = 1)

def fDoubleGaussian(X):
    """Double Gaussian Objective Function.

    Arguments:

        X: 'ndarray((nsamples, ndim))'

    returns:

        'ndarray(nsamples)'
    """
    mu1 = -1*np.ones(X.shape[1])
    mu2 = np.ones(X.shape[1])
    mu  = np.vstack((mu1, mu2))
    sig = np.eye(X.shape[1])

    mv1 = multivariate_normal(mu[0], sig)
    mv2 = multivariate_normal(mu[1], sig)

    return -(mv1.logpdf(X) + mv2.logpdf(X))

def fRotatedDoubleGaussian(X):
    """Rotated Double Gaussian Objective Function.

    Arguments:

        X: 'ndarray((nsamples, ndim))'

    returns:

        'ndarray(nsamples)'
    """
    mu1 = -1*np.ones(X.shape[1])
    mu2 = np.ones(X.shape[1])
    mu  = np.vstack((mu1, mu2))
    sig = np.eye(X.shape[1])

    R = Rotation(seed = 31415926)
    mu = R(mu)

    mv1 = multivariate_normal(mu[0], sig)
    mv2 = multivariate_normal(mu[1], sig)

    return -(mv1.logpdf(X) + mv2.logpdf(X))
