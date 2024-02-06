import os
from dataclasses import dataclass, field
from typing import Union
from functools import partial
from glob import glob
import warnings
from copy import deepcopy
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
from .binning import binRV


@dataclass
class RV:
    """
    A class holding RV observations

    Examples:
        >>> s = RV('Proxima')

    Attributes:
        star (str):
            The name of the star
        N (int):
            Total number of observations
        instruments (list):
            List of instruments for which there are RVs. Each instrument is also
            stored as an attribute (e.g. `self.CORALIE98` or `self.HARPS`)
        simbad (simbad):
            Information on the target from Simbad
    """
    star: str
    instrument: str = field(init=True, repr=False, default=None)
    N: int = field(init=False, repr=True)
    verbose: bool = field(init=True, repr=False, default=True)
    do_maxerror: Union[bool, float] = field(init=True, repr=False, default=False)
    do_secular_acceleration: bool = field(init=True, repr=False, default=True)
    do_sigma_clip: bool = field(init=True, repr=False, default=False)
    do_adjust_means: bool = field(init=True, repr=False, default=True)
    only_latest_pipeline: bool = field(init=True, repr=False, default=True)
    #
    _child: bool = field(init=True, repr=False, default=False)
    _did_secular_acceleration: bool = field(init=False, repr=False, default=False)
    _did_sigma_clip: bool = field(init=False, repr=False, default=False)
    _did_adjust_means: bool = field(init=False, repr=False, default=False)
    _raise_on_error: bool = field(init=True, repr=False, default=True)

    def __repr__(self):
        if self.N == 0:
            return f"RV(star='{self.star}', N=0)"
        if self.time.size == self.mtime.size:
            return f"RV(star='{self.star}', N={self.N})"
        else:
            nmasked = self.N - self.mtime.size
            return f"RV(star='{self.star}', N={self.N}, masked={nmasked})"

    def __post_init__(self):
        self.__star__ = translate(self.star)

        if not self._child:
            try:
                self.simbad = simbad(self.__star__)
            except ValueError as e:
                logger.error(e)

            if self.verbose:
                logger.info(f'querying DACE for {self.__star__}...')
            try:
                self.dace_result = get_observations(self.__star__, self.instrument, 
                                                    verbose=self.verbose)
            except ValueError as e:
                if self._raise_on_error:
                    raise e
                else:
                    self.time = np.array([])
                    self.instruments = []
                    self.units = ''
                    return


            # store the date of the last DACE query
            time_stamp = datetime.now(timezone.utc)  #.isoformat().split('.')[0]
            self._last_dace_query = time_stamp

        self.units = 'm/s'

        # build children
        if not self._child:
            arrays = get_arrays(self.dace_result,
                                latest_pipeline=self.only_latest_pipeline,
                                verbose=self.verbose)

            for (inst, pipe, mode), data in arrays:
                child = RV.from_dace_data(self.star, inst, pipe, mode, data, _child=True)
                inst = inst.replace('-', '_')
                pipe = pipe.replace('.', '_').replace('__', '_')
                if self.only_latest_pipeline:
                    # save as self.INST
                    setattr(self, inst, child)
                else:
                    # save as self.INST_PIPE
                    setattr(self, f'{inst}_{pipe}', child)

        # build joint arrays
        if not self._child:
            #! sorted?
            if self.only_latest_pipeline:
                self.instruments = [
                    inst.replace('-', '_')
                    for (inst, _, _), _ in arrays
                ]
            else:
                self.instruments = [
                    inst.replace('-', '_') + '_' + pipe.replace('.', '_').replace('__', '_')
                    for (inst, pipe, _), _ in arrays
                ]
            # self.pipelines =

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
        self._did_secular_acceleration = False
        self._did_sigma_clip = False
        self._did_adjust_means = False
        self.__post_init__()

    def snapshot(self):
        import pickle
        from datetime import datetime
        ts = datetime.now().timestamp()
        star_name = self.star.replace(' ', '')
        file = f'{star_name}_{ts}.pkl'
        pickle.dump(self, open(file, 'wb'), protocol=0)
        if self.verbose:
            logger.info(f'Saved snapshot to {file}')

    @property
    def N(self):
        """Total number of observations"""
        return self.time.size

    @N.setter
    def N(self, value):
        if not isinstance(value, property):
            logger.error('Cannot set N directly')

    @property
    def NN(self):
        """ Total number of observations per instrument """
        return {inst: getattr(self, inst).N for inst in self.instruments}

    @property
    def N_nights(self):
        """ Number of individual nights """
        return binRV(self.mtime, None, None, binning_bins=True).size - 1

    @property
    def NN_nights(self):
        return {inst: getattr(self, inst).N_nights for inst in self.instruments}

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

    @property
    def _time_sorter(self):
        return np.argsort(self.time)

    @property
    def _mtime_sorter(self):
        return np.argsort(self.mtime)

    @property
    def _tt(self):
        return np.linspace(self.mtime.min(), self.mtime.max(), 20*self.N)

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
        s.mask[np.isnan(s.svrad)] = False
        ## be careful with bogus values
        s.mask[s.svrad < 0] = False


        # all other quantities
        s._quantities = []
        for arr in data.keys():
            if arr not in ('rjd', 'rv', 'rv_err'):
                if arr == 'mask':
                    # change name mask -> ccf_mask
                    setattr(s, 'ccf_mask', data[arr][ind])
                    s._quantities.append('ccf_mask')
                else:
                    # be careful with bogus values in rhk and rhk_err
                    if arr in ('rhk', 'rhk_err'):
                        mask99999 = (data[arr] == -99999) | (data[arr] == -99)
                        data[arr][mask99999] = np.nan

                    setattr(s, arr, data[arr][ind])
                    s._quantities.append(arr)

        s._quantities = np.array(s._quantities)

        # mask out drs_qc = False
        if not s.drs_qc.all():
            n = (~s.drs_qc).sum()
            logger.warning(f'masking {n} points where DRS QC failed for {inst}')
            s.mask &= s.drs_qc

        s.instruments = [inst]
        s.pipelines = [pipe]
        s.modes = [mode]

        return s

    @classmethod
    def from_arrays(cls, star, time, vrad, svrad, inst, *args, **kwargs):
        s = cls(star, _child=True)
        time, vrad, svrad = map(np.atleast_1d, (time, vrad, svrad))

        if time.size != vrad.size:
            logger.error(f'wrong dimensions: time({time.size}) != vrad({vrad.size})')
            raise ValueError from None
        if time.size != svrad.size:
            logger.error(f'wrong dimensions: time({time.size}) != svrad({svrad.size})')
            raise ValueError from None

        # time, RVs, uncertainties
        s.time = time
        s.vrad = vrad
        s.svrad = svrad
        # mask
        s.mask = kwargs.get('mask', np.full_like(s.time, True, dtype=bool))

        s.instruments = [inst]

        return s

    @classmethod
    def from_snapshot(cls, file=None, star=None):
        import pickle
        from datetime import datetime
        if star is None:
            assert file.endswith('.pkl'), 'expected a .pkl file'
            star, timestamp = file.replace('.pkl', '').split('_')
        else:
            try:
                file = sorted(glob(f'{star}_*.pkl'))[-1]
            except IndexError:
                raise ValueError(f'cannot find any file matching {star}_*.pkl')
            star, timestamp = file.replace('.pkl', '').split('_')

        dt = datetime.fromtimestamp(float(timestamp))
        logger.info(f'Reading snapshot of {star} from {dt}')
        return pickle.load(open(file, 'rb'))

    @classmethod
    def from_rdb(cls, files, star=None, instrument=None, units='ms', **kwargs):
        if isinstance(files, str):
            files = [files]

        if star is None:
            star_ = np.unique([os.path.splitext(f)[0].split('_')[0] for f in files])
            if star_.size == 1:
                logger.info(f'assuming star is {star_[0]}')
                star = star_[0]
        
        
        if instrument is None:
            instruments = np.array([os.path.splitext(f)[0].split('_')[1] for f in files])
            logger.info(f'assuming instruments: {instruments}')
        else:
            instruments = np.atleast_1d(instrument)

        if instruments.size == 1 and len(files) > 1:
            instruments = np.repeat(instruments, len(files))

        factor = 1e3 if units == 'kms' else 1.0

        s = cls(star, _child=True, **kwargs)

        for i, (f, instrument) in enumerate(zip(files, instruments)):
            data = np.loadtxt(f, skiprows=2, usecols=range(3), unpack=True)
            _s = cls(star, _child=True, **kwargs)
            time = data[0]
            _s.time = time
            _s.vrad = data[1] * factor
            _s.svrad = data[2] * factor

            _quantities = []
            #! hack
            data = np.genfromtxt(f, names=True, dtype=None, comments='--', encoding=None)

            if 'fwhm' in data.dtype.fields:
                _s.fwhm = data['fwhm']
                if 'sfwhm' in data.dtype.fields:
                    _s.fwhm_err = data['sfwhm']
                else:
                    _s.fwhm_err = 2 * _s.svrad
            else:
                _s.fwhm = np.zeros_like(time)
                _s.fwhm_err = np.full_like(time, np.nan)

            _quantities.append('fwhm')
            _quantities.append('fwhm_err')

            if 'rhk' in data.dtype.fields:
                _s.rhk = data['rhk']
                if 'srhk' in data.dtype.fields:
                    _s.rhk_err = data['srhk']
            else:
                _s.rhk = np.zeros_like(time)
                _s.rhk_err = np.full_like(time, np.nan)

            _quantities.append('rhk')
            _quantities.append('rhk_err')

            _s.bispan = np.zeros_like(time)
            _s.bispan_err = np.full_like(time, np.nan)
            #! end hack

            _s.mask = np.ones_like(time, dtype=bool)
            _s.obs = np.full_like(time, i + 1)

            _s.instruments = [instrument]
            _s._quantities = np.array(_quantities)
            setattr(s, instrument, _s)

        s._child = False
        s.instruments = list(instruments)
        s._build_arrays()

        if kwargs.get('do_adjust_means', False):
            s.adjust_means()

        return s

    def _check_instrument(self, instrument, strict=False):
        """
        Check if there are observations from `instrument`.

        Args:
            instrument (str, None): Instrument name to check
            strict (bool): Whether to match `instrument` exactly
        Returns:
            instruments (list):
                List of instruments matching `instrument`, or None if there
                are no matches.
        """
        if instrument is None:
            return self.instruments
        if not strict:
            if any([instrument in inst for inst in self.instruments]):
                return [inst for inst in self.instruments if instrument in inst]
        if instrument in self.instruments:
            return [instrument]
        

    def _build_arrays(self):
        """ build all concatenated arrays of `self` from each of the `.inst`s """
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

        # mask
        self.mask = np.concatenate(
            [getattr(self, inst).mask for inst in self.instruments]
        )

        # "observatory" (or instrument id)
        self.obs = np.concatenate(
            [np.full(getattr(self, inst).N, i+1) for i, inst in enumerate(self.instruments)],
            dtype=int
        )


        # all other quantities
        self._quantities = getattr(self, self.instruments[0])._quantities
        if len(self.instruments) > 1:
            for inst in self.instruments[1:]:
                self._quantities = np.intersect1d(self._quantities, getattr(self, inst)._quantities)

        for q in self._quantities:
            if q not in ('rjd', 'rv', 'rv_err'):
                arr = np.concatenate(
                    [getattr(getattr(self, inst), q) for inst in self.instruments]
                )
                setattr(self, q, arr)


    def download_ccf(self, instrument=None, limit=None, directory=None):
        """ Download CCFs from DACE

        Args:
            instrument (str): Specific instrument for which to download data
            limit (int): Maximum number of files to download.
            directory (str): Directory where to store data.
        """
        if directory is None:
            directory = f'{self.star}_downloads'

        if instrument is None:
            files = [file for file in self.raw_file if file.endswith('.fits')]
        else:
            instrument = self._check_instrument(instrument)
            files = []
            for inst in instrument:
                files += list(getattr(self, inst).raw_file)

        do_download_ccf(files[:limit], directory)

    def download_s1d(self, instrument=None, limit=None, directory=None):
        """ Download S1Ds from DACE

        Args:
            instrument (str): Specific instrument for which to download data
            limit (int): Maximum number of files to download.
            directory (str): Directory where to store data.
        """
        if directory is None:
            directory = f'{self.star}_downloads'

        if instrument is None:
            files = [file for file in self.raw_file if file.endswith('.fits')]
        else:
            instrument = self._check_instrument(instrument)
            files = []
            for inst in instrument:
                files += list(getattr(self, inst).raw_file)

        do_download_s1d(files[:limit], directory)

    def download_s2d(self, instrument=None, limit=None, directory=None):
        """ Download S2Ds from DACE

        Args:
            instrument (str): Specific instrument for which to download data
            limit (int): Maximum number of files to download.
            directory (str): Directory where to store data.
        """
        if directory is None:
            directory = f'{self.star}_downloads'

        if instrument is None:
            files = [file for file in self.raw_file if file.endswith('.fits')]
        else:
            instrument = self._check_instrument(instrument)
            files = []
            for inst in instrument:
                files += list(getattr(self, inst).raw_file)

        extracted_files = do_download_s2d(files[:limit], directory)


    from .plots import plot, plot_fwhm, plot_bis, plot_rhk, plot_quantity
    from .plots import gls, gls_fwhm, gls_bis, gls_rhk
    from .reports import report

    from .instrument_specific import known_issues


    def remove_instrument(self, instrument, strict=False):
        """ Remove all observations from one instrument
        
        Args:
            instrument (str): The instrument for which to remove observations.
            strict (bool): Whether to match `instrument` exactly
        
        Note:
            A common name can be used to remove observations for several subsets
            of a given instrument. For example

            ```py
            s.remove_instrument('HARPS')
            ```

            will remove observations from `HARPS03` and `HARPS15`, if they
            exist. But

            ```py
            s.remove_instrument('HARPS03')
            ```

            will remove observations from the specific subset.
        """
        instruments = self._check_instrument(instrument, strict)

        if instruments is None:
            logger.error(f"No data from instrument '{instrument}'")
            logger.info(f'available: {self.instruments}')
            return

        for instrument in instruments:
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
            for q in self._quantities:
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
        """ Remove individual observations at a given index (or indices)
        
        Args:
            index (int, list, ndarray):
                Single index, list, or array of indices to remove.
        """
        index = np.atleast_1d(index)
        try:
            instrument_index = self.obs[index]
            instrument = np.array(self.instruments)[instrument_index - 1]
        except IndexError:
            logger.errors(f'index {index} is out of bounds for N={self.N}')
            return

        if self.verbose:
            logger.info(f'removing points {index}')

        self.mask[index] = False
        self._propagate_mask_changes()
        # for i, inst in zip(index, instrument):
        #     index_in_instrument = i - (self.obs < instrument_index).sum()
        #     getattr(self, inst).mask[index_in_instrument] = False
        if return_self:
            return self

    def remove_non_public(self):
        if self.verbose:
            n = (~self.public).sum()
            logger.info(f'masking non-public observations ({n})')
        self.mask = self.mask & self.public
        self._propagate_mask_changes()

    def remove_single_observations(self):
        """ Remove instruments for which there is a single observation """
        instruments = deepcopy(self.instruments)
        for inst in instruments:
            if getattr(self, inst).mtime.size == 1:
                self.remove_instrument(inst)

    def remove_prog_id(self, prog_id):
        from glob import has_magic
        if has_magic(prog_id):
            from fnmatch import filter
            matching = np.unique(filter(self.prog_id, prog_id))
            mask = np.full_like(self.time, False, dtype=bool)
            for m in matching:
                mask |= np.isin(self.prog_id, m)
            ind = np.where(mask)[0]
            self.remove_point(ind)
        else:
            if prog_id in self.prog_id:
                ind = np.where(self.prog_id == prog_id)[0]
                self.remove_point(ind)
            else:
                if self.verbose:
                    logger.warning(f'no observations for prog_id "{prog_id}"')


    def remove_after_bjd(self, bjd):
        if (self.time > bjd).any():
            ind = np.where(self.time > bjd)[0]
            self.remove_point(ind)


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
        
        try:
            self.simbad
        except AttributeError:
            if self.verbose:
                logger.error('no information from simbad, cannot remove secular acceleration')
            return

        #as_yr = units.arcsec / units.year
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

        if self._child:
            self.vrad = self.vrad - sa * (self.time - epoch) / 365.25
        else:
            for inst in self.instruments:
                if 'HIRES' in inst:  # never remove it from HIRES...
                    continue
                if 'NIRPS' in inst:  # never remove it from NIRPS...
                    continue

                s = getattr(self, inst)
                s.vrad = s.vrad - sa * (s.time - epoch) / 365.25
            
            self._build_arrays()

        self._did_secular_acceleration = True
        if return_self:
            return self

    def sigmaclip(self, sigma=5):
        """ Sigma-clip RVs (per instrument!) """
        #from scipy.stats import sigmaclip as dosigmaclip
        from .stats import sigmaclip_median as dosigmaclip

        if self._child or self._did_sigma_clip:
            return

        for inst in self.instruments:
            m = self.instrument_array == inst
            result = dosigmaclip(self.vrad[m], low=sigma, high=sigma)
            n = self.vrad[m].size - result.clipped.size

            ind = m & ((self.vrad < result.lower) | (self.vrad > result.upper))

            if self.verbose and n > 0:
                s = 's' if (n == 0 or n > 1) else ''
                logger.warning(f'sigma-clip RVs will remove {n} point{s} for {inst}')

            # # check if going to remove all observations from one instrument
            # if n in self.NN.values(): # all observations
            #     # insts = np.unique(self.instrument_array[~ind])
            #     # if insts.size == 1: # of the same instrument?
            #     if self.verbose:
            #         logger.warning(f'would remove all observations from {insts[0]}, skipping')
            #     if return_self:
            #         return self
            #     continue

            self.mask[ind] = False

        self._propagate_mask_changes()

        if self._did_adjust_means:
            self._did_adjust_means = False
            self.adjust_means()

        if return_self:
            return self

    def clip_maxerror(self, maxerror:float, plot=False):
        """ Mask out points with RV error larger than a given value
        
        Args:
            maxerror (float): Maximum error to keep.
            plot (bool): Whether to plot the masked points.
        """
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

    def bin(self):
        """
        Nightly bin the observations.

        WARNING: This creates and returns a new object and does not modify self.
        """
        
        # create copy of self to be returned
        snew = deepcopy(self)
        
        all_bad_quantities = []

        for inst in snew.instruments:
            s = getattr(snew, inst)

            # only one observation?
            if s.N == 1:
                continue
        
            # are all observations masked?
            if s.mtime.size == 0:
                continue

            tb, vb, svb = binRV(s.mtime, s.mvrad, s.msvrad)
            s.vrad = vb
            s.svrad = svb

            bad_quantities = []

            for q in s._quantities:
                Q = getattr(s, q)

                # treat date_night specially, basically doing a group-by
                if q == 'date_night':
                    inds = binRV(s.mtime, None, None, binning_indices=True)
                    setattr(s, q, Q[s.mask][inds])
                    continue

                if Q.dtype != np.float64:
                    bad_quantities.append(q)
                    all_bad_quantities.append(q)
                    continue

                if np.isnan(Q).all():
                    yb = np.full_like(tb, np.nan)
                    setattr(s, q, yb)

                elif q + '_err' in s._quantities:
                    Qerr = getattr(s, q + '_err')
                    if (Qerr == 0.0).all(): # if all errors are NaN, don't use them
                        _, yb = binRV(s.mtime, Q[s.mask], stat='mean', tstat='mean')
                    else:
                        if (Qerr <= 0.0).any(): # if any error is <= 0, set it to NaN
                            Qerr[Qerr <= 0.0] = np.nan

                        _, yb, eb = binRV(s.mtime, Q[s.mask], Qerr[s.mask], remove_nans=False)
                        setattr(s, q + '_err', eb)

                    setattr(s, q, yb)

                elif not q.endswith('_err'):
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        try:
                            _, yb = binRV(s.mtime, Q[s.mask], 
                                        stat=np.nanmean, tstat=np.nanmean)
                            setattr(s, q, yb)
                        except TypeError:
                            pass

            if snew.verbose and len(bad_quantities) > 0:
                logger.warning(f"{inst}, skipping non-float quantities in binning:")
                logger.warning(' ' + str(bad_quantities))
                for bq in bad_quantities:
                    s._quantities = np.delete(s._quantities, s._quantities==bq)
                    delattr(s, bq)  #! careful here

            s.time = tb
            s.mask = np.full(tb.shape, True)
        
        if snew.verbose and len(all_bad_quantities) > 0:
            logger.warning('\nnew object will not have these non-float quantities')

        for q in np.unique(all_bad_quantities):
            delattr(snew, q)

        snew._did_bin = True
        snew._build_arrays()
        return snew

    def nth_day_mean(self, n=1.0):
        mask = np.abs(self.mtime[:, None] - self.mtime[None, :]) < n
        z = np.full((self.mtime.size, self.mtime.size), np.nan)
        z[mask] = np.repeat(self.mvrad[:, None], self.mtime.size, axis=1)[mask]
        return np.nanmean(z, axis=0)

    def subtract_mean(self):
        """ Subtract (single) mean RV from all instruments """
        self._meanRV = meanRV = self.mvrad.mean()
        for inst in self.instruments:
            s = getattr(self, inst)
            s.vrad -= meanRV
        self._build_arrays()

    def _add_back_mean(self):
        """ Add the (single) mean RV removed by self.subtract_mean() """
        if not hasattr(self, '_meanRV'):
            logger.warning("no mean RV stored, run 'self.subtract_mean()'")
            return

        for inst in self.instruments:
            s = getattr(self, inst)
            s.vrad += self._meanRV
        self._build_arrays()

    def adjust_means(self, just_rv=False):
        """ Subtract individual mean RVs from each instrument """
        if self._child or self._did_adjust_means:
            return

        others = ('fwhm', 'bispan', )
        for inst in self.instruments:
            s = getattr(self, inst)

            if s.N == 1:
                if self.verbose:
                    msg = (f'only 1 observation for {inst}, '
                           'adjust_means will set it exactly to zero')
                    logger.warning(msg)
                s.rv_mean = s.vrad[0]
                s.vrad = np.zeros_like(s.time)
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

    def put_at_systemic_velocity(self):
        """
        For instruments in which mean(RV) < ptp(RV), "move" RVs to the systemic
        velocity from simbad. This is useful if some instruments are centered
        at zero while others are not, and instead of calling `.adjust_means()`,
        but it only works when the systemic velocity is smaller than ptp(RV).
        """
        changed = False
        for inst in self.instruments:
            s = getattr(self, inst)
            if s.mask.any():
                if np.abs(s.mvrad.mean()) < s.mvrad.ptp():
                    s.vrad += self.simbad.rvz_radvel * 1e3
                    changed = True
            else:  # all observations are masked, use non-masked arrays
                if np.abs(s.vrad.mean()) < s.vrad.ptp():
                    s.vrad += self.simbad.rvz_radvel * 1e3
                    changed = True
        if changed:
            self._build_arrays()

    def sort_instruments(self, by_first_observation=True, by_last_observation=False):
        if by_last_observation:
            by_first_observation = False
        # if by_first_observation and by_last_observation:
        #     logger.error("'by_first_observation' and 'by_last_observation' can't both be true")
        #     return
        if by_first_observation:
            fun = lambda i: getattr(self, i).time.min()
            self.instruments = sorted(self.instruments, key=fun)
            self._build_arrays()
        if by_last_observation:
            fun = lambda i: getattr(self, i).time.max()
            self.instruments = sorted(self.instruments, key=fun)
            self._build_arrays()

    #

    def save(self, directory=None, instrument=None, full=False,
             save_nans=True):
        """ Save the observations in .rdb files.

        Args:
            directory (str, optional):
                Directory where to save the .rdb files.
            instrument (str, optional):
                Instrument for which to save observations.
            full (bool, optional): 
                Whether to save just RVs and errors (False) or more indicators
                (True).
            save_nans (bool, optional)
                Whether to save NaN values in the indicators, if they exist. If
                False, the full observation is not saved.
        """
        star_name = self.star.replace(' ', '')

        if directory is None:
            directory = '.'
        else:
            os.makedirs(directory, exist_ok=True)

        files = []

        for inst in self.instruments:
            if instrument is not None:
                if instrument not in inst:
                    continue

            _s = getattr(self, inst)

            if not _s.mask.any():  # all observations are masked, don't save
                continue

            if full:
                d = np.c_[
                    _s.mtime, _s.mvrad, _s.msvrad,
                    _s.fwhm[_s.mask], _s.fwhm_err[_s.mask],
                    _s.rhk[_s.mask], _s.rhk_err[_s.mask],
                ]
                if not save_nans:
                    if np.isnan(d).any():
                        # remove observations where any of the indicators are # NaN
                        nan_mask = np.isnan(d[:, 3:]).any(axis=1)
                        d = d[~nan_mask]
                        if self.verbose:
                            logger.warning(f'masking {nan_mask.sum()} observations with NaN in indicators')

                header =  'bjd\tvrad\tsvrad\tfwhm\tsfwhm\trhk\tsrhk\n'
                header += '---\t----\t-----\t----\t-----\t---\t----'
            else:
                d = np.c_[_s.mtime, _s.mvrad, _s.msvrad]
                header = 'bjd\tvrad\tsvrad\n---\t----\t-----'
            
            file = f'{star_name}_{inst}.rdb'
            files.append(file)
            file = os.path.join(directory, file)

            np.savetxt(file, d, fmt='%9.5f', header=header, delimiter='\t', comments='')

            if self.verbose:
                logger.info(f'saving to {file}')
        
        return files

    def checksum(self, write_to=None):
        from hashlib import md5
        d = np.r_[self.time, self.vrad, self.svrad]
        H = md5(d.data.tobytes()).hexdigest()
        if write_to is not None:
            with open(write_to, 'w') as f:
                f.write(H)
        return H


    #
    def run_lbl(self, instrument=None, data_dir=None, 
                skysub=False, tell=False, limit=None, **kwargs):
        from .lbl_wrapper import run_lbl, NIRPS_create_telluric_corrected_S2D

        if instrument is None:
            instruments = self.instruments
        else:
            if instrument not in self.instruments:
                if any([instrument in i for i in self.instruments]):
                    instrument = [i for i in self.instruments if instrument in i]
                else:
                    logger.error(f"No data from instrument '{instrument}'")
                    logger.info(f'available: {self.instruments}')
                    return
            
            if isinstance(instrument, str):
                instruments = [instrument]
            else:
                instruments = instrument

        for instrument in instruments:
            if self.verbose:
                logger.info(f'gathering files for {instrument}')
            files = getattr(self, instrument).raw_file
            files = map(os.path.basename, files)
            if skysub:
                files = [file.replace('.fits', '_S2D_SKYSUB_A.fits') for file in files]
            else:
                files = [file.replace('.fits', '_S2D_A.fits') for file in files]

            if data_dir is None:
                data_dir = f'{self.star}_downloads'

            files = [os.path.join(data_dir, file) for file in files]
            exist = [os.path.exists(file) for file in files]
            if not all(exist):
                logger.error(f"not all required files exist in {data_dir}")
                logger.error(f"missing {np.logical_not(exist).sum()} / {len(files)}")

                from distutils.util import strtobool
                go_on = input('continue? (y/N) ')
                if go_on == '' or not bool(strtobool(go_on)):
                    return

                files = list(np.array(files)[exist])

            # deal with NIRPS telluric correction
            if 'NIRPS' in instrument and tell:
                if self.verbose:
                    logger.info('creating telluric-corrected S2D files')
                files = NIRPS_create_telluric_corrected_S2D(files[:limit])

            run_lbl(self, instrument, files[:limit], **kwargs)

    def load_lbl(self, instrument=None, tell=False, id=None):
        if hasattr(self, '_did_load_lbl') and self._did_load_lbl: # don't do it twice
            return

        from .lbl_wrapper import load_lbl

        if instrument is None:
            instruments = self.instruments
        else:
            if instrument not in self.instruments:
                if any([instrument in i for i in self.instruments]):
                    instrument = [i for i in self.instruments if instrument in i]
                else:
                    logger.error(f"No data from instrument '{instrument}'")
                    logger.info(f'available: {self.instruments}')
                    return
            
            if isinstance(instrument, str):
                instruments = [instrument]
            else:
                instruments = instrument

        for inst in instruments:
            if self.verbose:
                logger.info(f'loading LBL data for {inst}')

            load_lbl(self, inst, tell=tell, id=id)
            # self.instruments.append(f'{inst}_LBL')

        # self._build_arrays()
        self._did_load_lbl = True


    #
    @property
    def planets(self):
        from .nasaexo_wrapper import Planets
        if not hasattr(self, '_planets'):
            self._planets = Planets(self)
        return self._planets


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