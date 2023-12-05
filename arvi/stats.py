import numpy as np
from scipy.stats import median_abs_deviation
from scipy.stats._stats_py import SigmaclipResult


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


def sigmaclip_median(a, low=4.0, high=4.0):
    """
    Same as scipy.stats.sigmaclip but using the median and median absolute
    deviation instead of the mean and standard deviation.
    """
    c = np.asarray(a).ravel()
    delta = 1
    while delta:
        c_mad = median_abs_deviation(c)
        c_median = np.median(c)
        size = c.size
        critlower = c_median - c_mad * low
        critupper = c_median + c_mad * high
        c = c[(c >= critlower) & (c <= critupper)]
        delta = size - c.size

    return SigmaclipResult(c, critlower, critupper)