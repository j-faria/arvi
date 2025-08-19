from functools import partial
import numpy as np
from scipy.stats import norm

def wmean(a, e):
    """Weighted mean of array `a`, with uncertainties given by `e`.
    The weighted mean is calculated using 1/e**2 as weights.

    Args:
        a (array): Array containing data
        e (array): Uncertainties on `a`
    """
    if (e == 0).any():
        raise ZeroDivisionError
    if (e < 0).any():
        raise ValueError
    if (a.shape != e.shape):
        raise ValueError
    return np.average(a, weights=1 / e**2)

def rms(a, ignore_nans=False):
    """ Root mean square of array `a`

    Args:
        a (array): Array containing data
    """
    if ignore_nans:
        a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan
    return np.sqrt((a**2).mean())

def wrms(a, e, ignore_nans=False):
    """ Weighted root mean square of array `a`, with uncertanty given by `e`.
    The weighted rms is calculated using the weighted mean, where the weights
    are equal to 1/e**2.
    
    Args:
        a (array): Array containing data 
        e (array): Uncertainties on `a`
    """
    if ignore_nans:
        nans = np.logical_or(np.isnan(a), np.isnan(e))
        a = a[~nans]
        e = e[~nans]
    if (e == 0).any():
        raise ZeroDivisionError('uncertainty cannot be zero')
    if (e < 0).any():
        raise ValueError('uncertainty cannot be negative')
    if (a.shape != e.shape):
        raise ValueError('arrays must have the same shape')
    w = 1 / e**2
    return np.sqrt(np.sum(w * (a - np.average(a, weights=w))**2) / sum(w))

# from https://stackoverflow.com/questions/20601872/numpy-or-scipy-to-calculate-weighted-median
def weighted_quantiles_interpolate(values, weights, quantiles):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    q = np.searchsorted(c, quantiles * c[-1])
    # Ensure right-end isn't out of bounds. Thanks @Jeromino!
    q_plus1 = np.clip(q + 1, a_min=None, a_max=values.shape[0] - 1)
    return np.where(
        c[q] / c[-1] == quantiles,
        0.5 * (values[i[q]] + values[i[q_plus1]]),
        values[i[q]],
    )

weighted_median = partial(weighted_quantiles_interpolate, quantiles=0.5)



def sigmaclip_median(a, low=4.0, high=4.0, k=1/norm.ppf(3/4)):
    """
    Same as scipy.stats.sigmaclip but using the median and median absolute
    deviation instead of the mean and standard deviation.

    Args:
        a (array):
            Array containing data
        low (float):
            Number of MAD to use for the lower clipping limit
        high (float):
            Number of MAD to use for the upper clipping limit
        k (float):
            Scale factor for the MAD to be an estimator of the standard
            deviation. Depends on the (assumed) distribution of the data.
            Default value is for the normal distribution (=1/norm.ppf(3/4)).
    Returns:
        SigmaclipResult: Object with the following attributes:
            - `clipped`: Masked array of data
            - `lower`: Lower clipping limit
            - `upper`: Upper clipping limit
    """
    from scipy.stats import median_abs_deviation
    from scipy.stats._stats_py import SigmaclipResult
    c = np.asarray(a).ravel()
    delta = 1
    while delta:
        c_mad = median_abs_deviation(c) * k
        c_median = np.median(c)
        size = c.size
        critlower = c_median - c_mad * low
        critupper = c_median + c_mad * high
        c = c[(c >= critlower) & (c <= critupper)]
        delta = size - c.size

    return SigmaclipResult(c, critlower, critupper)