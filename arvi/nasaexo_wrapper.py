from copy import copy

import requests
from io import StringIO

import numpy as np
from astropy.timeseries import LombScargle

from .setup_logger import logger
from kepmodel.rv import RvModel
from spleaf.term import Error


url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?'

STAR_QUERY = [
    'select *',
    'from ps',
    'where',
    'default_flag=1',
    'and',
    "hostname like '{star}'"
]


def run_query(query):
    link = f'{url}query={query}&format=csv'
    r = requests.get(link)
    data = np.genfromtxt(StringIO(r.text), delimiter=',', names=True, 
                         dtype=None, encoding=None)
    return r, data

class Planets:
    def __init__(self, system):
        self.s = system
        self.verbose = system.verbose

        self.star = system.star.replace('GJ', 'GJ ').replace('HD', 'HD ')

        query = ' '.join(STAR_QUERY).replace(' ', '+')
        query = query.format(star=self.star)

        if self.verbose:
            logger.info('querying NASA Exoplanet Archive...')

        self.response, self.data = run_query(query)
        self.np = self.data.size

        # try again with other ids
        if self.np == 0:
            try:
                if hasattr(self.s, 'hd'):
                    hdname = [self.s.hd]
                else:
                    hdname = [i for i in self.s.simbad.ids if 'HD' in i]
                if len(hdname) != 0:
                    hdname = hdname[0]
                    if self.verbose:
                        logger.info(f"trying with the HD name '{hdname}'...")
                    STAR_QUERY_HD = STAR_QUERY[:-1]
                    STAR_QUERY_HD.append(f"hd_name like '{hdname}'")
                    query = ' '.join(STAR_QUERY_HD).replace(' ', '+')
                    self.response, self.data = run_query(query)
                    self.np = self.data.size
                
                hipname = [i for i in self.s.simbad.ids if 'HIP' in i]
                if len(hipname) != 0:
                    hipname = hipname[0]
                    if self.verbose:
                        logger.info(f"trying with the HIP name '{hipname}'...")
                    STAR_QUERY_HIP = STAR_QUERY[:-1]
                    STAR_QUERY_HIP.append(f"hip_name like '{hipname}'")
                    query = ' '.join(STAR_QUERY_HIP).replace(' ', '+')
                    self.response, self.data = run_query(query)
                    self.np = self.data.size
            except AttributeError:
                pass

        if self.verbose:
            logger.info(f'found {self.np} planets')

        self.create_model()

    def create_model(self, instrument=None, strict=False,
                     all_instruments_except=None, **kwargs):
        if instrument is None and all_instruments_except is None:
            ts = self.s._mtime_sorter
            self.model = RvModel(self.s.mtime[ts], self.s.mvrad[ts],
                                err=Error(self.s.msvrad[ts]))
            for inst in self.s.instruments:
                if inst not in self.s.instrument_array[self.s.mask]:
                    continue
                self.model.add_lin(derivative=1.0*(self.s.instrument_array[self.s.mask][ts]==inst),
                                   name=f'offset_inst_{inst}',
                                   value=getattr(self.s, inst).mvrad.mean())
        else:
            if all_instruments_except is not None:
                exception = self.s._check_instrument(all_instruments_except, strict)
                instruments = copy(self.s.instruments)
                for e in exception:
                    instruments.pop(instruments.index(e))
            else:
                instruments = self.s._check_instrument(instrument, strict)
            mask = np.in1d(self.s.instrument_array, instruments)
            ts = self.s._time_sorter[mask]
            self.model = RvModel(self.s.time[ts], self.s.vrad[ts],
                                err=Error(self.s.svrad[ts]))
            for inst in instruments:
                self.model.add_lin(1.0*(self.s.instrument_array[ts]==inst),
                                   f'offset_inst_{inst}')
        self.set_parameters()

    def set_parameters(self):
        self.P = np.atleast_1d(self.data['pl_orbper'])
        self.K = np.atleast_1d(self.data['pl_rvamp'])
        self.e = np.atleast_1d(self.data['pl_orbeccen'])
        self.w = np.atleast_1d(self.data['pl_orblper'])

        if self.P.size == 0:
            return

        if self.model.nkep > 0:
            for i in range(self.np):
                self.model.rm_keplerian(f'{i}')
            self.model.keplerian.uid = 0

        for i in range(self.np):
            if not self.K[i] or np.isnan(self.K[i]):
                self.model.add_keplerian_from_period(self.P[i], fit=True)
            else:
                self.model.add_keplerian([self.P[i], self.K[i], self.e[i], 0.0, self.w[i]],
                                        ['P', 'K', 'e', 'M0', 'omega'], fit=True)

    def add_planet_from_period(self, period):
        self.model.add_keplerian_from_period(period, fit=True)
        self.np += 1
    
    def add_keplerian_from_periodogram(self):
        gls = LombScargle(self.model.t, self.model.y - self.model.model())
        freq, power = gls.autopower()
        self.add_planet_from_period(1/freq[power.argmax()])

    def fit_lin(self, adjust_data=True):
        self.model.show_param()
        self.model.fit_lin()
        self.model.show_param()
        if adjust_data:
            for inst in self.s.instruments:
                _s = getattr(self.s, inst)
                try:
                    _s.vrad -= self.model.get_param(f'lin.offset_inst_{inst}')
                except ValueError:
                    pass
            self.s._build_arrays()

    def fit_angles(self):
        old_param = copy(self.model.fit_param)
        fit_param = [f'kep.{i}.M0' for i in range(self.np)]
        fit_param += [f'kep.{i}.omega' for i in range(self.np)]
        self.model.fit_param = fit_param
        self.model.fit()
        self.model.fit_param = old_param
        self.model.show_param()

    def fit_all(self, adjust_data=False):
        self.model.fit()

        newP = np.array([self.model.get_param(f'kep.{i}.P') for i in range(self.np)])
        if self.verbose and not np.allclose(self.P, newP):
            logger.warning(f'periods changed: {self.P}  -->  {newP}')

        newK = np.array([self.model.get_param(f'kep.{i}.K') for i in range(self.np)])
        if self.verbose and not np.allclose(self.K, newK):
            logger.warning(f'amplitudes changed: {self.K}  -->  {newK}')
        
        newE = np.array([self.model.get_param(f'kep.{i}.e') for i in range(self.np)])
        if self.verbose and not np.allclose(self.e, newE):
            logger.warning(f'eccentricities changed: {self.e}  -->  {newE}')

        if adjust_data:
            for inst in self.s.instruments:
                _s = getattr(self.s, inst)
                try:
                    _s.vrad -= self.model.get_param(f'lin.offset_inst_{inst}')
                except ValueError:
                    pass
            self.s._build_arrays()

    def __repr__(self):
        return f'{self.star}({self.np} planets, '\
               f'P={list(self.P)}, K={list(self.K)}, e={list(self.e)})'
