from dataclasses import dataclass, field
from typing import Union
from functools import partial
from datetime import datetime, timezone
import numpy as np

from astropy import units

from .setup_logger import logger
from .config import return_self
from .translations import translate
from .dace_wrapper import get_observations, get_arrays
from .dace_wrapper import do_download_ccf, do_download_s1d, do_download_s2d
from .simbad_wrapper import simbad
from .stats import wmean, wrms


@dataclass
class RV:
    """
    A class holding RV observations

    Attributes:
        star (str):
            The name of the star
        N (int):
            Total number of observations
        verbose (bool):
            Log some operations to the terminal
        instruments (list):
            List of instruments for which there are RVs. Each instrument is also
            stored as an attribute (e.g. `self.CORALIE98` or `self.HARPS`)
    """
    star: str
    N: int = field(init=False, repr=True)
    verbose: bool = field(init=True, repr=False, default=True)
    do_maxerror: Union[bool, float] = field(init=True, repr=False, default=100)
    do_secular_acceleration: bool = field(init=True, repr=False, default=True)
    do_sigma_clip: bool = field(init=True, repr=False, default=True)
    do_adjust_means: bool = field(init=True, repr=False, default=True)
    #
    _child: bool = field(init=True, repr=False, default=False)
    _did_secular_acceleration: bool = field(init=False, repr=False, default=False)
    _did_sigma_clip: bool = field(init=False, repr=False, default=False)
    _did_adjust_means: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self.__star__ = translate(self.star)
        try:
            self.simbad = simbad(self.__star__)
        except ValueError as e:
            logger.error(e.msg)

        if not self._child:
            if self.verbose:
                logger.info('querying DACE...')
            self.dace_result = get_observations(self.__star__,
                                                verbose=self.verbose)
            # store the date of the last DACE query
            time_stamp = datetime.now(timezone.utc)  #.isoformat().split('.')[0]
            self._last_dace_query = time_stamp

        self.units = 'm/s'

        # build children
        if not self._child:
            arrays = get_arrays(self.dace_result)
            for (inst, pipe, mode), data in arrays:
                child = RV.from_dace_data(self.star, inst, pipe, mode, data, _child=True)
                setattr(self, inst, child)

        # build joint arrays
        if not self._child:
            #! sorted?
            self.instruments = sorted(list(self.dace_result.keys()))
            # self.pipelines =
            # "observatory" (or instrument id)
            self.obs = np.concatenate(
                [np.full(getattr(self, inst).N, i+1) for i, inst in enumerate(self.instruments)],
                dtype=int
            )
            # mask
            self.mask = np.full_like(self.obs, True, dtype=bool)
            # all other quantities
            self._build_arrays()

        # do clip_maxerror, secular_acceleration, sigmaclip, adjust_means
        if not self._child:
            if self.do_maxerror:
                self.clip_maxerror(self.do_maxerror)

            if self.do_secular_acceleration:
                self.secular_acceleration()

            if self.do_sigma_clip:
                self.sigmaclip()

            if self.do_adjust_means:
                self.adjust_means()


    def reload(self):
        self.__post_init__()

    @property
    def N(self):
        return self.time.size

    @N.setter
    def N(self, value):
        if not isinstance(value, property):
            logger.error('Cannot set N directly')

    @property
    def NN(self):
        return {inst: getattr(self, inst).N for inst in self.instruments}

    @property
    def mtime(self):
        return self.time[self.mask]

    @property
    def mvrad(self):
        return self.vrad[self.mask]

    @property
    def msvrad(self):
        return self.svrad[self.mask]

    @property
    def instrument_array(self):
        return np.concatenate([[i] * n for i, n in self.NN.items()])

    @property
    def rms(self):
        """ Weighted rms of the (masked) radial velocities """
        if self.mask.sum() == 0:  # only one point
            return np.nan
        else:
            return wrms(self.vrad[self.mask], self.svrad[self.mask])

    @property
    def sigma(self):
        """ Average error bar """
        if self.mask.sum() == 0:  # only one point
            return np.nan
        else:
            return self.svrad[self.mask].mean()

    error = sigma  # alias!

    @classmethod
    def from_dace_data(cls, star, inst, pipe, mode, data, **kwargs):
        s = cls(star, **kwargs)
        #
        ind = np.argsort(data['rjd'])
        # time, RVs, uncertainties
        s.time = data['rjd'][ind]
        s.vrad = data['rv'][ind]
        s.svrad = data['rv_err'][ind]
        # mask
        s.mask = np.full_like(s.time, True, dtype=bool)
        # all other quantities
        for arr in data.keys():
            if arr not in ('rjd', 'rv', 'rv_err'):
                if arr == 'mask':
                    # change name mask -> ccf_mask
                    setattr(s, 'ccf_mask', data[arr][ind])
                else:
                    setattr(s, arr, data[arr][ind])
        #
        s.instruments = [inst]
        s.pipelines = [pipe]
        return s

    def _build_arrays(self):
        if self._child:
            return
        # time
        self.time = np.concatenate(
            [getattr(self, inst).time for inst in self.instruments]
        )
        # RVs
        self.vrad = np.concatenate(
            [getattr(self, inst).vrad for inst in self.instruments]
        )
        # uncertainties
        self.svrad = np.concatenate(
            [getattr(self, inst).svrad for inst in self.instruments]
        )
        arrays = get_arrays(self.dace_result)
        self.quantities = list(arrays[0][-1].keys())
        self.quantities[self.quantities.index('mask')] = 'ccf_mask'
        # all other quantities
        for q in self.quantities:
            if q not in ('rjd', 'rv', 'rv_err'):
                arr = np.concatenate(
                    [getattr(getattr(self, inst), q) for inst in self.instruments]
                )
                setattr(self, q, arr)


    def download_ccf(self, instrument=None):
        directory = f'{self.star}_downloads'
        if instrument is None:
            files = [file for file in self.raw_file if file.endswith('.fits')]
        else:
            if instrument not in self.instruments:
                logger.error(f"No data from instrument '{instrument}'")
                logger.info(f'available: {self.instruments}')
                return
            files = getattr(self, instrument).raw_file

        do_download_ccf(files, directory)

    def download_s1d(self, instrument=None):
        directory = f'{self.star}_downloads'
        if instrument is None:
            files = [file for file in self.raw_file if file.endswith('.fits')]
        else:
            if instrument not in self.instruments:
                logger.error(f"No data from instrument '{instrument}'")
                logger.info(f'available: {self.instruments}')
                return
            files = getattr(self, instrument).raw_file

        do_download_s1d(files, directory)

    def download_s2d(self, instrument=None):
        directory = f'{self.star}_downloads'
        if instrument is None:
            files = [file for file in self.raw_file if file.endswith('.fits')]
        else:
            if instrument not in self.instruments:
                logger.error(f"No data from instrument '{instrument}'")
                logger.info(f'available: {self.instruments}')
                return
            files = getattr(self, instrument).raw_file

        do_download_s2d(files, directory)


    from .plots import plot, plot_fwhm, plot_bis
    from .plots import gls, gls_fwhm, gls_bis
    from .reports import report


    def remove_instrument(self, instrument):
        """ Remove all observations from `instrument` """
        if instrument not in self.instruments:
            logger.error(f"No data from instrument '{instrument}'")
            logger.info(f'available: {self.instruments}')
            return

        ind = self.instruments.index(instrument) + 1
        remove = np.where(self.obs == ind)
        self.obs = np.delete(self.obs, remove)
        self.obs[self.obs > ind] -= 1
        #
        self.time = np.delete(self.time, remove)
        self.vrad = np.delete(self.vrad, remove)
        self.svrad = np.delete(self.svrad, remove)
        #
        self.mask = np.delete(self.mask, remove)
        #
        # all other quantities
        for q in self.quantities:
            if q not in ('rjd', 'rv', 'rv_err'):
                new = np.delete(getattr(self, q), remove)
                setattr(self, q, new)
        #
        self.instruments.remove(instrument)
        #
        delattr(self, instrument)

        if self.verbose:
            logger.info(f"Removed observations from '{instrument}'")

        if return_self:
            return self

    def remove_point(self, index):
        """ Remove individual observations at a given `index` (or indices) """
        index = np.atleast_1d(index)
        try:
            instrument_index = self.obs[index]
            instrument = np.array(self.instruments)[instrument_index - 1]
        except IndexError:
            logger.errors(f'index {index} is out of bounds for N={self.N}')
            return

        self.mask[index] = False
        self._propagate_mask_changes()
        # for i, inst in zip(index, instrument):
        #     index_in_instrument = i - (self.obs < instrument_index).sum()
        #     getattr(self, inst).mask[index_in_instrument] = False
        if return_self:
            return self

    def _propagate_mask_changes(self):
        """ link self.mask with each self.`instrument`.mask """
        masked = np.where(~self.mask)[0]
        for m in masked:
            inst = self.instruments[self.obs[m] - 1]
            n_before = (self.obs < self.obs[m]).sum()
            getattr(self, inst).mask[m - n_before] = False

    def secular_acceleration(self, epoch=55500, plot=False):
        """
        Remove secular acceleration from RVs

        Args:
            epoch (float):
                The reference epoch (DACE uses 55500, 31/10/2010)
            instruments (bool or collection of str):
                Only remove secular acceleration for some instruments, or for all 
                if `instruments=True`
            plot (bool):
                Show a plot of the RVs with the secular acceleration
        """
        if self._did_secular_acceleration:  # don't do it twice
            return

        as_yr = units.arcsec / units.year
        mas_yr = units.milliarcsecond / units.year
        mas = units.milliarcsecond

        π = self.simbad.plx_value * mas
        d = π.to(units.pc, equivalencies=units.parallax())
        μα = self.simbad.pmra * mas_yr
        μδ = self.simbad.pmdec * mas_yr
        μ = μα**2 + μδ**2
        sa = (μ * d).to(units.m / units.second / units.year,
                        equivalencies=units.dimensionless_angles())
        sa = sa.value

        if self.verbose:
            logger.info('removing secular acceleration from RVs')

        if self.units == 'km/s':
            sa /= 1000

        for inst in self.instruments:
            if 'HIRES' in inst:  # never remove it from HIRES...
                continue

            s = getattr(self, inst)
            s.vrad = s.vrad - sa * (s.time - epoch) / 365.25
        
        self._build_arrays()
        self._did_secular_acceleration = True
        if return_self:
            return self

    def sigmaclip(self, sigma=3):
        """ Sigma-clip RVs """
        if self._child or self._did_sigma_clip:
            return
        from scipy.stats import sigmaclip as dosigmaclip
        result = dosigmaclip(self.vrad, low=sigma, high=sigma)
        n = self.vrad.size - result.clipped.size
        if self.verbose and n > 0:
            s = 's' if (n == 0 or n > 1) else ''
            logger.warning(f'sigma-clip RVs removed {n} point' + s)
        ind = (self.vrad > result.lower) & (self.vrad < result.upper)
        self.mask[~ind] = False
        self._propagate_mask_changes()
        if return_self:
            return self

    def clip_maxerror(self, maxerror:float, plot=False):
        """ Mask out points with RV error larger than `maxerror` """
        if self._child:
            return
        self.maxerror = maxerror
        above = self.svrad > maxerror
        n = above.sum()
        self.mask[above] = False

        if self.verbose and above.sum() > 0:
            s = 's' if (n == 0 or n > 1) else ''
            logger.warning(f'clip_maxerror ({maxerror} {self.units}) removed {n} point' + s)

        self._propagate_mask_changes()
        if return_self:
            return self

    def adjust_means(self, just_rv=False):
        if self._child or self._did_adjust_means:
            return

        others = ('fwhm', 'bispan', )
        for inst in self.instruments:
            s = getattr(self, inst)

            if s.N == 1:
                if self.verbose:
                    logger.warning(f'only 1 observation for {inst}, skipping')
                continue

            s.rv_mean = wmean(s.mvrad, s.msvrad)
            s.vrad -= s.rv_mean
            if self.verbose:
                logger.info(f'subtracted weighted average from {inst:10s}: ({s.rv_mean:.3f} {self.units})')
            if just_rv:
                continue
            # log_msg = 'same for '
            for i, other in enumerate(others):
                y, ye = getattr(s, other), getattr(s, other + '_err')
                m = wmean(y[s.mask], ye[s.mask])
                setattr(s, f'{other}_mean', m)
                setattr(s, other, getattr(s, other) - m)
                # log_msg += other
                # if i < len(others) - 1:
                #     log_msg += ', '
            
            # if self.verbose:
            #     logger.info(log_msg)

        self._build_arrays()
        self._did_adjust_means = True
        if return_self:
            return self

def fit_sine(t, y, yerr, period='gls', fix_period=False):
    from scipy.optimize import leastsq
    if period == 'gls':
        from astropy.timeseries import LombScargle
        gls = LombScargle(t, y, yerr)
        freq, power = gls.autopower()
        period = 1 / freq[power.argmax()]

    if fix_period and period is None:
        logger.error('period is fixed but no value provided')
        return

    def sine(t, p):
        return p[0] * np.sin(2 * np.pi * t / p[1] + p[2]) + p[3]

    p0 = [y.ptp(), period, 0.0, 0.0]
    xbest, _ = leastsq(lambda p, t, y, ye: (sine(t, p) - y) / ye,
                       p0, args=(t, y, yerr))
    return xbest, partial(sine, p=xbest)