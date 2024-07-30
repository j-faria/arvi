import numpy as np

from .setup_logger import logger

###############################################################################
# the following is mostly a copy of the scipy implementation of
# binned_statistic and binned_statistic_dd
# but allowing for a weights parameter

## careful here!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
xrange = range


def binned_statistic(x, values, statistic='mean', bins=10, range=None,
                     weights=None):
    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1:
        bins = [np.asarray(bins, float)]

    if range is not None:
        if len(range) == 2:
            range = [range]

    medians, edges, binnumbers = binned_statistic_dd(
        [x], values, statistic, bins, range, None, weights)
    return medians, edges[0], binnumbers


def binned_statistic_dd(sample, values, statistic='mean', bins=10, range=None,
                        expand_binnumbers=False, weights=None):
    from numpy.testing import suppress_warnings
    known_stats = [
        'mean', 'median', 'count', 'sum', 'std', 'min', 'max', 'ptp'
    ]
    if not callable(statistic) and statistic not in known_stats:
        raise ValueError('invalid statistic %r' % (statistic, ))

    # `Ndim` is the number of dimensions (e.g. `2` for `binned_statistic_2d`)
    # `Dlen` is the length of elements along each dimension.
    # This code is based on np.histogramdd
    try:
        # `sample` is an ND-array.
        Dlen, Ndim = sample.shape
    except (AttributeError, ValueError):
        # `sample` is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        Dlen, Ndim = sample.shape

    # Store initial shape of `values` to preserve it in the output
    values = np.asarray(values)
    input_shape = list(values.shape)
    # Make sure that `values` is 2D to iterate over rows
    values = np.atleast_2d(values)
    if weights is not None:
        weights = np.atleast_2d(weights)
    Vdim, Vlen = values.shape

    # Make sure `values` match `sample`
    if (statistic != 'count' and Vlen != Dlen):
        raise AttributeError('The number of `values` elements must match the '
                             'length of each `sample` dimension.')

    nbin = np.empty(Ndim, int)  # Number of bins in each dimension
    edges = Ndim * [None]  # Bin edges for each dim (will be 2D array)
    dedges = Ndim * [None]  # Spacing between edges (will be 2D array)

    try:
        M = len(bins)
        if M != Ndim:
            raise AttributeError('The dimension of bins must be equal '
                                 'to the dimension of the sample x.')
    except TypeError:
        bins = Ndim * [bins]

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        smin = np.atleast_1d(np.array(sample.min(axis=0), float))
        smax = np.atleast_1d(np.array(sample.max(axis=0), float))
    else:
        smin = np.zeros(Ndim)
        smax = np.zeros(Ndim)
        for i in xrange(Ndim):
            smin[i], smax[i] = range[i]

    # Make sure the bins have a finite width.
    for i in xrange(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Create edge arrays
    for i in xrange(Ndim):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1)
        else:
            edges[i] = np.asarray(bins[i], float)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i])

    nbin = np.asarray(nbin)

    # Compute the bin number each sample falls into, in each dimension
    sampBin = [np.digitize(sample[:, i], edges[i]) for i in xrange(Ndim)]

    # Using `digitize`, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    for i in xrange(Ndim):
        # Find the rounding precision
        decimal = int(-np.log10(dedges[i].min())) + 6
        # Find which points are on the rightmost edge.
        on_edge = np.where(
            np.around(sample[:, i], decimal) == np.around(
                edges[i][-1], decimal))[0]
        # Shift these points one bin to the left.
        sampBin[i][on_edge] -= 1

    # Compute the sample indices in the flattened statistic matrix.
    binnumbers = np.ravel_multi_index(sampBin, nbin)

    result = np.empty([Vdim, nbin.prod()], float)

    if statistic == 'mean':
        result.fill(np.nan)
        flatcount = np.bincount(binnumbers, None)
        a = flatcount.nonzero()
        for vv in xrange(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            result[vv, a] = flatsum[a] / flatcount[a]
    elif statistic == 'std':
        result.fill(0)
        flatcount = np.bincount(binnumbers, None)
        a = flatcount.nonzero()
        for vv in xrange(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            flatsum2 = np.bincount(binnumbers, values[vv]**2)
            result[vv, a] = np.sqrt(flatsum2[a] / flatcount[a] -
                                    (flatsum[a] / flatcount[a])**2)
    elif statistic == 'count':
        result.fill(0)
        flatcount = np.bincount(binnumbers, None)
        a = np.arange(len(flatcount))
        result[:, a] = flatcount[np.newaxis, :]
    elif statistic == 'sum':
        result.fill(0)
        for vv in xrange(Vdim):
            flatsum = np.bincount(binnumbers, values[vv])
            a = np.arange(len(flatsum))
            result[vv, a] = flatsum
    elif statistic == 'median':
        result.fill(np.nan)
        for i in np.unique(binnumbers):
            for vv in xrange(Vdim):
                result[vv, i] = np.median(values[vv, binnumbers == i])
    elif statistic == 'min':
        result.fill(np.nan)
        for i in np.unique(binnumbers):
            for vv in xrange(Vdim):
                result[vv, i] = np.min(values[vv, binnumbers == i])
    elif statistic == 'max':
        result.fill(np.nan)
        for i in np.unique(binnumbers):
            for vv in xrange(Vdim):
                result[vv, i] = np.max(values[vv, binnumbers == i])
    elif statistic == 'ptp':
        result.fill(np.nan)
        for i in np.unique(binnumbers):
            for vv in xrange(Vdim):
                result[vv, i] = np.ptp(values[vv, binnumbers == i])
    elif callable(statistic):
        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                null = statistic([], weights=[])
            except:
                null = np.nan
        result.fill(null)
        for i in np.unique(binnumbers):
            for vv in xrange(Vdim):
                if weights is None:
                    result[vv, i] = statistic(values[vv, binnumbers == i])
                else:
                    result[vv, i] = statistic(values[vv, binnumbers == i],
                                              weights[vv, binnumbers == i])

    # Shape into a proper matrix
    result = result.reshape(np.append(Vdim, nbin))

    # Remove outliers (indices 0 and -1 for each bin-dimension).
    core = tuple([slice(None)] + Ndim * [slice(1, -1)])
    result = result[core]

    # Unravel binnumbers into an ndarray, each row the bins for each dimension
    if (expand_binnumbers and Ndim > 1):
        binnumbers = np.asarray(np.unravel_index(binnumbers, nbin))

    if np.any(result.shape[1:] != nbin - 2):
        raise RuntimeError('Internal Shape Error')

    # Reshape to have output (`reulst`) match input (`values`) shape
    result = result.reshape(input_shape[:-1] + list(nbin - 2))

    return result, edges, binnumbers


# # put back the documentation
# from scipy.stats import binned_statistic as old_binned_statistic,\
#                         binned_statistic_dd as old_binned_statistic_dd
# doc1 = old_binned_statistic.__doc__
# doc2 = old_binned_statistic_dd.__doc__
# binned_statistic.__doc__ = doc1
# binned_statistic_dd.__doc__ = doc2

###############################################################################


def binRV(time, rv, err=None, stat='wmean', tstat='wmean', estat='addquad',
          remove_nans=True, binning_indices=False, binning_bins=False,
          n_consecutive=None, consecutive_step=None, seed=None):
    """ Bin a dataset of radial-velocity observations.

    Parameters
    ----------
    time : array
        The array of times where the radial velocity is measured. This function
        does nightly binning, based on the integer part of this array.
    rv : array
        The radial-velocity values.
    err : array (optional)
        The uncertainty on the radial-velocity values.
    stat : string or callable (optional)
        The statistic to compute for the radial velocities. By default, this is
        'mean' if the `err` array is not provided and weighted mean ('wmean') 
        if `err` is provided. In that case, the routine does inverse variance
        averaging, using 1/`err`**2 as weights. See other available statistics
        on the `binned_statistic` function.
    tstat : string or callable (optional)
        The statistic to compute for the times. The default is the same as for
        the radial velocities. See other available statistics on the
        `binned_statistic` function.
    estat : string or callable (optional)
        The statistic to compute for the errors. The default is to add them
        quadratically and divide by the number. See other available statistics
        on the `binned_statistic` function.
    remove_nans : bool (optional, default True)
        Whether NaN values in the binned times should be removed
    binning_indices : bool (optional, default False)
        If True, don't actually do any binning, just return the indices that
        will nightly bin the `time` array.
    binning_bins : bool (optional, default=False)
        If True, don't actually do any binning, just return the bins that can be
        passed to binned_statistic to nightly bin the `time` array.
    n_consecutive : int (optional)
        If provided, only `n_consecutive` points within each bin will be used
        for the calculation (if more than that number are available). Which
        points are used is chosen randomly.
    consecutive_step : float (optional)
        If `n_consecutive` is provided, `consecutive_step` defines the maximum
        time step between two points that would be considered consecutive.

    Notes
    -----
    Arguably, the most justified way to perform binning is to use the defaults,
    with times, radial velocities and errors as input. This will lead to a
    weighted average of the times, a weighted average of the radial velocities
    and quadratically added errors.
    """

    # "nightly" binning, based on the integer part of the time array
    intt = time.astype(int)
    _, indices = np.unique(intt, return_index=True)

    if binning_indices:
        return indices

    # include the last time in the array of bins, so we don't miss a point
    # but be careful: if it is alone, perturb the last time by a small ammount
    # so that the difference time[-1] - bins[-1] is not zero
    if indices[-1] == time.size - 1:
        bins = np.r_[time[indices], time[-1] + 1e-10]
    else:
        bins = np.r_[time[indices], time[-1]]

    if binning_bins:
        return bins

    if n_consecutive and not consecutive_step:
        print('Warning: ignoring `n_consecutive` because `consecutive_step` '
              'was not provided')

    elif n_consecutive and consecutive_step:
        # easiest solution is to rebuild the time, rv, and err arrays with
        # only the randomly selected n_consecutive points for each bin
        np.random.seed(seed)

        def consecutive(a, step):
            return (np.ediff1d(a) < step).all()

        newtime = []
        newrv = []
        newerr = []
        digits = np.digitize(time, bins)
        random_choices = []
        for d in np.unique(digits):
            m = np.where(digits == d)[0]
            poss = [
                slice(i, i + n_consecutive)
                for i in range(time[m].size - n_consecutive - 1)
                if consecutive(time[m][slice(i, i + n_consecutive)], consecutive_step)
            ]
            if len(poss) > 0:
                sl = np.random.choice(poss)
                newtime.append(time[m][sl])
                newrv.append(rv[m][sl])
                if err is not None:
                    newerr.append(err[m][sl])
                random_choices.append(
                    slice(m[0] + sl.start, m[0] + sl.stop, sl.step))

        time = np.array(newtime).flatten()
        rv = np.array(newrv).flatten()
        if err is not None:
            err = np.array(newerr).flatten()

    # weighted mean of a
    def wmean(a, e):
        return np.ma.average(a, weights=1 / e**2)

    # bin the RVs
    if (err is not None) and (stat == 'wmean'):
        stat = wmean  # default is weighted mean

    if (err is None) and (stat == 'wmean'):
        stat = 'mean'

    brv = binned_statistic(time, rv, statistic=stat, bins=bins, range=None,
                           weights=err)

    # bin the times
    if (err is not None) and (tstat == 'wmean'):
        tstat = wmean  # default is weighted mean

    if (err is None) and (tstat == 'wmean'):
        tstat = 'mean'

    times = binned_statistic(time, time, statistic=tstat, bins=bins,
                             range=None, weights=err)
    # if there are errors, bin them too
    if err is not None:
        if estat == 'addquad':
            # this function adds the elements of `a` quadratically
            add_quadratically = lambda a: np.sqrt(np.add.reduce(a**2))
            # this function then divides by the number of elements
            mean_quadratically = lambda a: add_quadratically(a) / a.size
            # the two previous functions are only separated for clarity
            estat = mean_quadratically
            errors = binned_statistic(time, err, statistic=estat, bins=bins)
        else:
            errors = binned_statistic(time, err, statistic=estat, bins=bins)
    else:
        errors = [None]

    btime, brv, berr = times[0], brv[0], errors[0]

    # statistic can return nan in some cases
    if remove_nans:
        n = np.isnan(btime)
        btime = btime[~n]
        brv = brv[~n]
        if err is not None:
            berr = berr[~n]

    if err is not None:
        if n_consecutive and consecutive_step:
            return btime, brv, berr, random_choices
        return btime, brv, berr

    else:
        if n_consecutive and consecutive_step:
            return btime, brv, random_choices
        return btime, brv


def bin_ccf_mask(time, ccf_mask):
    indices = binRV(time, None, binning_indices=True)
    indices = np.r_[indices, time.size]
    bmask = []
    for i1, i2 in zip(indices, indices[1:]):
        um = np.unique(ccf_mask[i1:i2]).squeeze()
        if um.size > 1:
            logger.error(f'Non-unique CCF mask within one night (t={time[i1]:.1f}). '
                         'Setting to NaN, but RV should be discarded')
            bmask.append('nan')
        else:
            bmask.append(um)
    return np.array(bmask)