import numpy as np


def wmean(a, e):
    """Weighted mean of array `a`, with uncertainties given by `e`.
    The weighted mean is calculated using 1/e**2 as weights.

    Args:
        a (array): Array containing data
        e (array): Uncertainties on `a`
    """
    return np.average(a, weights=1 / e**2)


def rms(a):
    """ Root mean square of array `a`

    Args:
        a (array): Array containing data
    """
    return np.sqrt((a**2).mean())


def wrms(a, e):
    """ Weighted root mean square of array `a`, with uncertanty given by `e`.
    The weighted rms is calculated using the weighted mean, where the weights
    are equal to 1/e**2.
    
    Args:
        a (array): Array containing data 
        e (array): Uncertainties on `a`
    """
    w = 1 / e**2
    return np.sqrt(np.sum(w * (a - np.average(a, weights=w))**2) / sum(w))
