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
from . import config
from .translations import translate
from .dace_wrapper import do_download_filetype, do_symlink_filetype, get_observations, get_arrays
from .simbad_wrapper import simbad
from .gaia_wrapper import gaia
from .extra_data import get_extra_data
from .stats import wmean, wrms
from .binning import bin_ccf_mask, binRV
from .HZ import getHZ_period
from .utils import strtobool, there_is_internet, timer


class ExtraFields:
    pass

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
    verbose: bool = field(init=True, repr=False, default=True)
    do_maxerror: Union[bool, float] = field(init=True, repr=False, default=False)
    do_secular_acceleration: bool = field(init=True, repr=False, default=True)
    do_sigma_clip: bool = field(init=True, repr=False, default=False)
    do_adjust_means: bool = field(init=True, repr=False, default=True)
    only_latest_pipeline: bool = field(init=True, repr=False, default=True)
    load_extra_data: Union[bool, str] = field(init=True, repr=False, default=False)
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
            if config.check_internet and not there_is_internet():
                raise ConnectionError('There is no internet connection?')

            # complicated way to query Simbad with self.__star__ or, if that
            # fails, try after removing a trailing 'A'
            for target in (self.__star__, self.__star__.replace('A', '')):
                try:
                    self.simbad = simbad(target)
                    break
                except ValueError:
                    continue
            else:
                if self.verbose:
                    logger.error(f'simbad query for {self.__star__} failed')

            # complicated way to query Gaia with self.__star__ or, if that
            # fails, try after removing a trailing 'A'
            for target in (self.__star__, self.__star__.replace('A', '')):
                try:
                    self.gaia = gaia(target)
                    break
                except ValueError:
                    continue
            else:
                if self.verbose:
                    logger.error(f'Gaia query for {self.__star__} failed')

            # query DACE
            if self.verbose:
                logger.info(f'querying DACE for {self.__star__}...')
            try:
                with timer():
                    mid = self.simbad.main_id if hasattr(self, 'simbad') else None
                    self.dace_result = get_observations(self.__star__, self.instrument,
                                                        main_id=mid, verbose=self.verbose)
            except ValueError as e:
                # querying DACE failed, should we raise an error?
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
                child = RV.from_dace_data(self.star, inst, pipe, mode, data, _child=True,
                                          verbose=self.verbose)
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


            if self.load_extra_data:
                if isinstance(self.load_extra_data, str):
                    path = self.load_extra_data
                else:
                    path = None
                try:
                    self.__add__(get_extra_data(self.star, instrument=self.instrument, path=path),
                                 inplace=True)

                except FileNotFoundError:
                    pass

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

    def __add__(self, other, inplace=False):
        # if not isinstance(other, self.__class__):
        #     raise TypeError('unsupported operand type(s) for +: '
        #                     f"'{self.__class__.__name__}' and '{other.__class__.__name__}'")

        if np.isin(self.instruments, other.instruments).any():
            logger.error('the two objects share instrument(s), cannot add them')
            return

        if inplace:
            #? could it be as simple as this?
            for i in other.instruments:
                self.instruments.append(i)
                setattr(self, i, getattr(other, i))
            self._build_arrays()
        else:
            # make a copy of ourselves
            new_self = deepcopy(self)
            #? could it be as simple as this?
            for i in other.instruments:
                new_self.instruments.append(i)
                setattr(new_self, i, getattr(other, i))
            new_self._build_arrays()
            return new_self


    def reload(self):
        self._did_secular_acceleration = False
        self._did_sigma_clip = False
        self._did_adjust_means = False
        self._did_correct_berv = False
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
    def N(self) -> int:
        """Total number of observations"""
        return self.time.size

    @property
    def NN(self):
        """ Total number of observations per instrument """
        return {inst: getattr(self, inst).N for inst in self.instruments}

    @property
    def N_nights(self) -> int:
        """ Number of individual nights """
        if self.mtime.size == 0:
            return 0
        return binRV(self.mtime, None, None, binning_bins=True).size - 1

    @property
    def NN_nights(self):
        return {inst: getattr(self, inst).N_nights for inst in self.instruments}

    @property
    def _NN_as_table(self) -> str:
        table = ''
        table += ' | '.join(self.instruments) + '\n'
        table += ' | '.join([i*'-' for i in map(len, self.instruments)]) + '\n'
        table += ' | '.join(map(str, self.NN.values())) + '\n'
        return table

    @property
    def point(self):
        return [(t.round(4), v.round(4), sv.round(4)) for t, v, sv in zip(self.time, self.vrad, self.svrad)]

    @property
    def mtime(self) -> np.ndarray:
        """ Masked array of times """
        return self.time[self.mask]

    @property
    def mvrad(self) -> np.ndarray:
        """ Masked array of radial velocities """
        return self.vrad[self.mask]

    @property
    def msvrad(self) -> np.ndarray:
        """ Masked array of radial velocity uncertainties """
        return self.svrad[self.mask]

    @property
    def instrument_array(self):
        return np.concatenate([[i] * n for i, n in self.NN.items()])

    @property
    def rms(self) -> float:
        """ Weighted rms of the (masked) radial velocities """
        if self.mask.sum() == 0:  # only one point
            return np.nan
        else:
            return wrms(self.vrad[self.mask], self.svrad[self.mask])

    @property
    def sigma(self):
        """ Average radial velocity uncertainty """
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
    def _tt(self) -> np.ndarray:
        return np.linspace(self.mtime.min(), self.mtime.max(), 20*self.N)

    @classmethod
    def from_dace_data(cls, star, inst, pipe, mode, data, **kwargs):
        verbose = kwargs.pop('verbose', False)
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
            if verbose:
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
        s._quantities = np.array([])

        return s

    @classmethod
    def from_snapshot(cls, file=None, star=None, verbose=True):
        import pickle
        from datetime import datetime
        if star is None:
            assert file.endswith('.pkl'), 'expected a .pkl file'
            star, timestamp = file.replace('.pkl', '').split('_')
        else:
            try:
                file = sorted(glob(f'{star}_*.*.pkl'))[-1]
            except IndexError:
                raise ValueError(f'cannot find any file matching {star}_*.pkl')
            star, timestamp = file.replace('.pkl', '').split('_')

        dt = datetime.fromtimestamp(float(timestamp))
        if verbose:
            logger.info(f'Reading snapshot of {star} from {dt}')
        return pickle.load(open(file, 'rb'))

    @classmethod
    def from_rdb(cls, files, star=None, instrument=None, units='ms', **kwargs):
        """ Create an RV object from an rdb file or a list of rdb files

        Args:
            files (str, list):
                File name or list of file names
            star (str, optional):
                Name of the star. If None, try to infer it from file name
            instrument (str, list, optional):
                Name of the instrument(s). If None, try to infer it from file name
            units (str, optional):
                Units of the radial velocities. Defaults to 'ms'.

        Examples:
            s = RV.from_rdb('star_HARPS.rdb')
        """
        if isinstance(files, str):
            files = [files]

        if star is None:
            star_ = np.unique([os.path.splitext(os.path.basename(f))[0].split('_')[0] for f in files])
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

        def find_column(data, names):
            has_col = np.array([name in data.dtype.fields for name in names])
            if any(has_col):
                col = np.where(has_col)[0][0]
                return data[names[col]]
            return False

        for i, (f, instrument) in enumerate(zip(files, instruments)):
            data = np.loadtxt(f, skiprows=2, usecols=range(3), unpack=True)
            _s = cls(star, _child=True, **kwargs)
            time = data[0]
            _s.time = time
            _s.vrad = data[1] * factor
            _s.svrad = data[2] * factor

            _quantities = []

            #! hack
            with open(f) as ff:
                header = ff.readline().strip()
                if '\t' in header:
                    names = header.split('\t')
                else:
                    names = header.split()

            if len(names) > 3:
                kw = dict(skip_header=0, comments='--', names=True, dtype=None, encoding=None)
                if '\t' in header:
                    data = np.genfromtxt(f, **kw, delimiter='\t')
                else:
                    data = np.genfromtxt(f, **kw)
                # data.dtype.names = names
            else:
                data = np.array([], dtype=np.dtype([]))

            # try to find FWHM and uncertainty
            if (v := find_column(data, ['fwhm'])) is not False:  # walrus !!
                _s.fwhm = v
                if (sv := find_column(data, ['sfwhm', 'fwhm_err', 'sig_fwhm'])) is not False:
                    _s.fwhm_err = sv
                else:
                    _s.fwhm_err = 2 * _s.svrad
            else:
                _s.fwhm = np.zeros_like(time)
                _s.fwhm_err = np.full_like(time, np.nan)

            _quantities.append('fwhm')
            _quantities.append('fwhm_err')

            if (v := find_column(data, ['rhk'])) is not False:
                _s.rhk = v
                _s.rhk_err = np.full_like(time, np.nan)
                if (sv := find_column(data, ['srhk', 'rhk_err', 'sig_rhk'])) is not False:
                    _s.rhk_err = sv
            else:
                _s.rhk = np.zeros_like(time)
                _s.rhk_err = np.full_like(time, np.nan)

            _quantities.append('rhk')
            _quantities.append('rhk_err')

            _s.bispan = np.zeros_like(time)
            _s.bispan_err = np.full_like(time, np.nan)

            # other quantities, but all NaNs
            for q in ['bispan', 'caindex', 'ccf_asym', 'contrast', 'haindex', 'naindex', 'sindex']:
                setattr(_s, q, np.full_like(time, np.nan))
                setattr(_s, q + '_err', np.full_like(time, np.nan))
                _quantities.append(q)
                _quantities.append(q + '_err')
            for q in ['berv', 'texp']:
                setattr(_s, q, np.full_like(time, np.nan))
                _quantities.append(q)
            for q in ['ccf_mask', 'date_night', 'prog_id', 'raw_file', 'pub_reference']:
                setattr(_s, q, np.full(time.size, ''))
                _quantities.append(q)
            for q in ['drs_qc']:
                setattr(_s, q, np.full(time.size, True))
                _quantities.append(q)

            _s.extra_fields = ExtraFields()
            for field in data.dtype.names:
                if field not in _quantities:
                    setattr(_s.extra_fields, field, data[field])
                    # _quantities.append(field)

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

    @classmethod
    def from_ccf(cls, files, star=None, instrument=None, **kwargs):
        """ Create an RV object from a CCF file or a list of CCF files """
        try:
            import iCCF
        except ImportError:
            logger.error('iCCF is not installed. Please install it with `pip install iCCF`')
            return

        verbose = kwargs.get('verbose', True)

        if isinstance(files, str):
            files = [files]

        CCFs = iCCF.from_file(files)

        if not isinstance(CCFs, list):
            CCFs = [CCFs]

        objects = np.unique([i.HDU[0].header['OBJECT'].replace(' ', '') for i in CCFs])
        if objects.size != 1:
            logger.warning(f'found {objects.size} different stars in the CCF files, '
                           'choosing the first one')
        star = objects[0]

        s = cls(star, _child=True)
        instruments = list(np.unique([i.instrument for i in CCFs]))

        for instrument in instruments:
            # time, RVs, uncertainties
            time = np.array([i.bjd for i in CCFs])
            vrad = np.array([i.RV*1e3 for i in CCFs])
            svrad = np.array([i.RVerror*1e3 for i in CCFs])
            _s = RV.from_arrays(star, time, vrad, svrad, inst=instrument)

            _quantities = []

            _s.fwhm = np.array([i.FWHM*1e3 for i in CCFs])
            _s.fwhm_err = np.array([i.FWHMerror*1e3 for i in CCFs])

            _quantities.append('fwhm')
            _quantities.append('fwhm_err')

            _s.contrast = np.array([i.contrast for i in CCFs])
            _s.contrast_err = np.array([i.contrast_error for i in CCFs])

            _quantities.append('contrast')
            _quantities.append('contrast_err')

            _s.texp = np.array([i.HDU[0].header['EXPTIME'] for i in CCFs])
            _quantities.append('texp')

            _s.date_night = np.array([
                i.HDU[0].header['DATE-OBS'].split('T')[0] for i in CCFs
            ])
            _quantities.append('date_night')

            _s.mask = np.full_like(_s.time, True, dtype=bool)

            _s.drs_qc = np.array([i.HDU[0].header['HIERARCH ESO QC SCIRED CHECK'] for i in CCFs], dtype=bool)
            # mask out drs_qc = False
            if not _s.drs_qc.all():
                n = (~ _s.drs_qc).sum()
                if verbose:
                    logger.warning(f'masking {n} points where DRS QC failed for {instrument}')
                _s.mask &= _s.drs_qc
            print(_s.mask)

            _s._quantities = np.array(_quantities)
            setattr(s, instrument, _s)

        s._child = False
        s.instruments = instruments
        s._build_arrays()

        if instruments == ['ESPRESSO']:
            from .instrument_specific import divide_ESPRESSO
            divide_ESPRESSO(s)

        return s


    def _check_instrument(self, instrument, strict=False, log=False):# -> list | None:
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

        if isinstance(instrument, list):
            if strict:
                return [inst for inst in instrument if inst in self.instruments]
            else:
                r = []
                for i in instrument:
                    if any([i in inst for inst in self.instruments]):
                        r += [inst for inst in self.instruments if i in inst]
                return r

        else:
            if strict:
                if instrument in self.instruments:
                    return [instrument]
            else:
                if any([instrument in inst for inst in self.instruments]):
                    return [inst for inst in self.instruments if instrument in inst]

        if log:
            logger.error(f"No data from instrument '{instrument}'")
            logger.info(f'available: {self.instruments}')
            return

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

    def download_ccf(self, instrument=None, index=None, limit=None,
                     directory=None, symlink=False, **kwargs):
        """ Download CCFs from DACE

        Args:
            instrument (str): Specific instrument for which to download data
            index (int): Specific index of point for which to download data (0-based)
            limit (int): Maximum number of files to download.
            directory (str): Directory where to store data.
        """
        if directory is None:
            directory = f'{self.star}_downloads'

        if instrument is None:
            files = [file for file in self.raw_file if file.endswith('.fits')]
        else:
            strict = kwargs.pop('strict', False)
            instrument = self._check_instrument(instrument, strict=strict)
            files = []
            for inst in instrument:
                files += list(getattr(self, inst).raw_file)

        if index is not None:
            index = np.atleast_1d(index)
            files = list(np.array(files)[index])

        # remove empty strings
        files = list(filter(None, files))

        if symlink:
            if 'top_level' not in kwargs:
                logger.warning('may need to provide `top_level` in kwargs to find file')
            do_symlink_filetype('CCF', files[:limit], directory, **kwargs)
        else:
            do_download_filetype('CCF', files[:limit], directory, verbose=self.verbose, **kwargs)

    def download_s1d(self, instrument=None, index=None, limit=None,
                     directory=None, symlink=False, **kwargs):
        """ Download S1Ds from DACE

        Args:
            instrument (str): Specific instrument for which to download data
            index (int): Specific index of point for which to download data (0-based)
            limit (int): Maximum number of files to download.
            directory (str): Directory where to store data.
        """
        if directory is None:
            directory = f'{self.star}_downloads'

        if instrument is None:
            files = [file for file in self.raw_file if file.endswith('.fits')]
        else:
            strict = kwargs.pop('strict', False)
            instrument = self._check_instrument(instrument, strict=strict)
            files = []
            for inst in instrument:
                files += list(getattr(self, inst).raw_file)

        if index is not None:
            index = np.atleast_1d(index)
            files = list(np.array(files)[index])

        # remove empty strings
        files = list(filter(None, files))

        if symlink:
            if 'top_level' not in kwargs:
                logger.warning('may need to provide `top_level` in kwargs to find file')
            do_symlink_filetype('S1D', files[:limit], directory, **kwargs)
        else:
            do_download_filetype('S1D', files[:limit], directory, verbose=self.verbose, **kwargs)

    def download_s2d(self, instrument=None, index=None, limit=None,
                     directory=None, symlink=False, **kwargs):
        """ Download S2Ds from DACE

        Args:
            instrument (str): Specific instrument for which to download data
            index (int): Specific index of point for which to download data (0-based)
            limit (int): Maximum number of files to download.
            directory (str): Directory where to store data.
        """
        if directory is None:
            directory = f'{self.star}_downloads'

        if instrument is None:
            files = [file for file in self.raw_file if file.endswith('.fits')]
        else:
            strict = kwargs.pop('strict', False)
            instrument = self._check_instrument(instrument, strict=strict)
            files = []
            for inst in instrument:
                files += list(getattr(self, inst).raw_file)

        if index is not None:
            index = np.atleast_1d(index)
            files = list(np.array(files)[index])

        # remove empty strings
        files = list(filter(None, files))

        if symlink:
            if 'top_level' not in kwargs:
                logger.warning('may need to provide `top_level` in kwargs to find file')
            do_symlink_filetype('S2D', files[:limit], directory, **kwargs)
        else:
            do_download_filetype('S2D', files[:limit], directory, verbose=self.verbose, **kwargs)


    from .plots import plot, plot_fwhm, plot_bis, plot_rhk, plot_berv, plot_quantity
    from .plots import gls, gls_fwhm, gls_bis, gls_rhk, window_function
    from .reports import report

    from .instrument_specific import known_issues


    def remove_instrument(self, instrument, strict=False):
        """ Remove all observations from one instrument

        Args:
            instrument (str or list):
                The instrument(s) for which to remove observations.
            strict (bool):
                Whether to match (each) `instrument` exactly

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

            will only remove observations from the specific subset.
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

        if config.return_self:
            return self

    def remove_condition(self, condition):
        """ Remove all observations that satisfy a condition

        Args:
            condition (np.ndarray):
                Boolean array of the same length as the observations
        """
        if self.verbose:
            inst = np.unique(self.instrument_array[condition])
            logger.info(f"Removing {condition.sum()} points from instruments {inst}")
        self.mask = self.mask & ~condition
        self._propagate_mask_changes()

    def remove_point(self, index):
        """
        Remove individual observations at a given index (or indices).
        NOTE: Like Python, the index is 0-based.

        Args:
            index (int, list, ndarray):
                Single index, list, or array of indices to remove.
        """
        index = np.atleast_1d(index)
        try:
            instrument_index = self.obs[index]
            np.array(self.instruments)[instrument_index - 1]
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
        if config.return_self:
            return self

    def remove_non_public(self):
        """ Remove non-public observations """
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
        """ Remove observations from a given program ID """
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
        """ Remove observations after a given BJD """
        if (self.time > bjd).any():
            ind = np.where(self.time > bjd)[0]
            self.remove_point(ind)

    def remove_before_bjd(self, bjd):
        """ Remove observations before a given BJD """
        if (self.time < bjd).any():
            ind = np.where(self.time < bjd)[0]
            self.remove_point(ind)

    def choose_n_points(self, n, seed=None, instrument=None):
        """ Randomly choose `n` observations and mask out the remaining ones

        Args:
            n (int):
                Number of observations to keep.
            seed (int, optional):
                Random seed for reproducibility.
            instrument (str or list, optional):
                For which instrument to choose points (default is all).
        """
        instruments = self._check_instrument(instrument)
        rng = np.random.default_rng(seed=seed)
        for inst in instruments:
            # s = getattr(self, inst)
            mask_for_this_inst = self.obs == self.instruments.index(inst) + 1
            # only choose if there are more than n points
            if self.mask[mask_for_this_inst].sum() > n:
                if self.verbose:
                    logger.info(f'selecting {n} points from {inst}')
                # indices of points for this instrument which are not masked already
                available = np.where(self.mask & mask_for_this_inst)[0]
                # choose n randomly
                i = rng.choice(available, size=n, replace=False)
                # mask the others out
                self.mask[np.setdiff1d(available, i)] = False
        self._propagate_mask_changes()


    def _propagate_mask_changes(self):
        """ link self.mask with each self.`instrument`.mask """
        masked = np.where(~self.mask)[0]
        for m in masked:
            inst = self.instruments[self.obs[m] - 1]
            n_before = (self.obs < self.obs[m]).sum()
            getattr(self, inst).mask[m - n_before] = False

    def secular_acceleration(self, epoch=None, just_compute=False, force_simbad=False):
        """
        Remove secular acceleration from RVs

        Args:
            epoch (float, optional):
                The reference epoch (DACE uses 55500, 31/10/2010)
            instruments (bool or collection of str):
                Only remove secular acceleration for some instruments, or for all
                if `instruments=True`
        """
        if self._did_secular_acceleration and not just_compute:  # don't do it twice
            return

        #as_yr = units.arcsec / units.year
        mas_yr = units.milliarcsecond / units.year
        mas = units.milliarcsecond

        try:
            if force_simbad:
                raise AttributeError

            self.gaia
            self.gaia.plx

            if self.verbose:
                logger.info('using Gaia information to remove secular acceleration')

            if epoch is None:
                # Gaia DR3 epoch (astropy.time.Time('J2016.0', format='jyear_str').jd)
                epoch = 57389.0

            π = self.gaia.plx * mas
            d = π.to(units.pc, equivalencies=units.parallax())
            μα = self.gaia.pmra * mas_yr
            μδ = self.gaia.pmdec * mas_yr
            μ = μα**2 + μδ**2
            sa = (μ * d).to(units.m / units.second / units.year,
                            equivalencies=units.dimensionless_angles())

        except AttributeError:
            try:
                self.simbad
            except AttributeError:
                if self.verbose:
                    logger.error('no information from simbad, cannot remove secular acceleration')
                return

            if self.simbad.plx is None:
                if self.verbose:
                    logger.error('no parallax from simbad, cannot remove secular acceleration')
                return

            if self.verbose:
                logger.info('using Simbad information to remove secular acceleration')

            if epoch is None:
                epoch = 55500

            π = self.simbad.plx * mas
            d = π.to(units.pc, equivalencies=units.parallax())
            μα = self.simbad.pmra * mas_yr
            μδ = self.simbad.pmdec * mas_yr
            μ = μα**2 + μδ**2
            sa = (μ * d).to(units.m / units.second / units.year,
                            equivalencies=units.dimensionless_angles())

        if just_compute:
            return sa

        sa = sa.value

        if self.verbose:
            logger.info('removing secular acceleration from RVs')

        if self.units == 'km/s':
            sa /= 1000

        if self._child:
            self.vrad = self.vrad - sa * (self.time - epoch) / 365.25
        else:
            for inst in self.instruments:
                s = getattr(self, inst)

                # if RVs come from a publication, don't remove the secular
                # acceleration
                if np.all(s.pub_reference != ''):
                    continue

                if 'HIRES' in inst:  # never remove it from HIRES...
                    continue
                if 'NIRPS' in inst:  # never remove it from NIRPS...
                    continue

                if hasattr(s, '_did_secular_acceleration') and s._did_secular_acceleration:
                    continue

                s.vrad = s.vrad - sa * (s.time - epoch) / 365.25

            self._build_arrays()

        self._did_secular_acceleration = True
        self._did_secular_acceleration_epoch = epoch
        self._did_secular_acceleration_simbad = force_simbad

        if config.return_self:
            return self
    
    def _undo_secular_acceleration(self):
        if self._did_secular_acceleration:
            _old_verbose = self.verbose
            self.verbose = False
            sa = self.secular_acceleration(just_compute=True,
                                           force_simbad=self._did_secular_acceleration_simbad)
            self.verbose = _old_verbose
            sa = sa.value

            if self._child:
                self.vrad = self.vrad + sa * (self.time - self._did_secular_acceleration_epoch) / 365.25
            else:
                for inst in self.instruments:
                    if 'HIRES' in inst:  # never remove it from HIRES...
                        continue
                    if 'NIRPS' in inst:  # never remove it from NIRPS...
                        continue

                    s = getattr(self, inst)

                    s.vrad = s.vrad + sa * (s.time - self._did_secular_acceleration_epoch) / 365.25

                self._build_arrays()

            self._did_secular_acceleration = False

    def sigmaclip(self, sigma=5, instrument=None, strict=True):
        """ Sigma-clip RVs (per instrument!) """
        #from scipy.stats import sigmaclip as dosigmaclip
        from .stats import sigmaclip_median as dosigmaclip

        if self._child or self._did_sigma_clip:
            return

        instruments = self._check_instrument(instrument, strict)

        for inst in instruments:
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
            #     if config.return_self:
            #         return self
            #     continue

            self.mask[ind] = False

        self._propagate_mask_changes()

        if self._did_adjust_means:
            self._did_adjust_means = False
            self.adjust_means()

        if config.return_self:
            return self

    def clip_maxerror(self, maxerror:float):
        """ Mask out points with RV error larger than a given value

        Args:
            maxerror (float): Maximum error to keep.
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
        if config.return_self:
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

                # treat ccf_mask specially, doing a 'unique' bin
                if q == 'ccf_mask':
                    setattr(s, q, bin_ccf_mask(s.mtime, getattr(s, q)))
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
        """ Calculate the n-th day rolling mean of the radial velocities """
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

            if s.mtime.size == 0:
                if self.verbose:
                    logger.info(f'all observations of {inst} are masked')
                continue

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

            for i, other in enumerate(others):
                y, ye = getattr(s, other), getattr(s, other + '_err')
                m = wmean(y[s.mask], ye[s.mask])
                setattr(s, f'{other}_mean', m)
                setattr(s, other, getattr(s, other) - m)

        self._build_arrays()
        self._did_adjust_means = True
        if config.return_self:
            return self

    def add_to_vrad(self, values):
        """ Add an array of values to the RVs of all instruments """
        if values.size != self.vrad.size:
            raise ValueError(f"incompatible sizes: len(values) must equal self.N, got {values.size} != {self.vrad.size}")

        for inst in self.instruments:
            s = getattr(self, inst)
            mask = self.instrument_array == inst
            s.vrad += values[mask]

        self._build_arrays()


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
        """ Sort instruments by first or last observation date.

        Args:
            by_first_observation (bool, optional):
                Sort by first observation date.
            by_last_observation (bool, optional):
                Sort by last observation data.
        """
        if by_last_observation:
            by_first_observation = False
        if by_first_observation:
            fun = lambda i: getattr(self, i).time.min()
            self.instruments = sorted(self.instruments, key=fun)
            self._build_arrays()
        if by_last_observation:
            fun = lambda i: getattr(self, i).time.max()
            self.instruments = sorted(self.instruments, key=fun)
            self._build_arrays()


    def save(self, directory=None, instrument=None, full=False, postfix=None,
             save_masked=False, save_nans=True):
        """ Save the observations in .rdb files.

        Args:
            directory (str, optional):
                Directory where to save the .rdb files.
            instrument (str, optional):
                Instrument for which to save observations.
            full (bool, optional):
                Save just RVs and errors (False) or more indicators (True).
            postfix (str, optional):
                Postfix to add to the filenames ([star]_[instrument]_[postfix].rdb).
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
                if save_masked:
                    d = np.c_[
                        _s.time, _s.vrad, _s.svrad,
                        _s.fwhm, _s.fwhm_err,
                        _s.rhk, _s.rhk_err,
                    ]
                else:
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
                if save_masked:
                    d = np.c_[_s.time, _s.vrad, _s.svrad]
                else:
                    d = np.c_[_s.mtime, _s.mvrad, _s.msvrad]
                header = 'bjd\tvrad\tsvrad\n---\t----\t-----'

            file = f'{star_name}_{inst}.rdb'
            if postfix is not None:
                file = f'{star_name}_{inst}_{postfix}.rdb'

            files.append(file)
            file = os.path.join(directory, file)

            np.savetxt(file, d, fmt='%9.5f', header=header, delimiter='\t', comments='')

            if self.verbose:
                logger.info(f'saving to {file}')

        return files

    def checksum(self, write_to=None):
        """ Calculate a hash based on the data """
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
    from .stellar import calc_prot_age

    @property
    def HZ(self):
        if not hasattr(self, 'star_mass'):
            self.star_mass = float(input('stellar mass (Msun): '))
        if not hasattr(self, 'lum'):
            self.lum = float(input('luminosity (Lsun): '))
        return getHZ_period(self.simbad.teff, self.star_mass, 1.0, self.lum)


    @property
    def planets(self):
        """ Query the NASA Exoplanet Archive for any known planets """
        from .nasaexo_wrapper import Planets
        if not hasattr(self, '_planets'):
            self._planets = Planets(self)
        return self._planets


def fit_sine(t, y, yerr=None, period='gls', fix_period=False):
    """ Fit a sine curve of the form y = A * sin(2π * t / P + φ) + c

    Args:
        t (ndarray):
            Time array
        y (ndarray):
            Array of observed values
        yerr (ndarray, optional):
            Array of uncertainties. Defaults to None.
        period (str or float, optional):
            Initial guess for period or 'gls' to get it from Lomb-Scargle
            periodogram. Defaults to 'gls'.
        fix_period (bool, optional):
            Whether to fix the period. Defaults to False.

    Returns:
        p (ndarray):
            Best-fit parameters [A, P, φ, c] or [A, φ, c]
        f (callable):
            Function that returns the best-fit sine curve for input times
    """
    from scipy.optimize import leastsq
    if period == 'gls':
        from astropy.timeseries import LombScargle
        gls = LombScargle(t, y, yerr)
        freq, power = gls.autopower()
        period = 1 / freq[power.argmax()]
    else:
        period = float(period)

    if yerr is None:
        yerr = np.ones_like(y)

    if fix_period:
        def sine(t, p):
            return p[0] * np.sin(2 * np.pi * t / period + p[1]) + p[2]
        f = lambda p, t, y, ye: (sine(t, p) - y) / ye
        p0 = [y.ptp(), 0.0, 0.0]
    else:
        def sine(t, p):
            return p[0] * np.sin(2 * np.pi * t / p[1] + p[2]) + p[3]
        f = lambda p, t, y, ye: (sine(t, p) - y) / ye
        p0 = [y.ptp(), period, 0.0, 0.0]

    xbest, _ = leastsq(f, p0, args=(t, y, yerr))
    return xbest, partial(sine, p=xbest)
