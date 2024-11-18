import os
import multiprocessing
from functools import partial, lru_cache
from itertools import chain
from collections import namedtuple
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
# import numpy as np

from .setup_logger import logger
from .timeseries import RV

__all__ = ['ESPRESSO_GTO']

path = os.path.join(os.path.dirname(__file__), 'data')


def get_star(star, instrument=None, verbose=False, **kwargs):
    return RV(star, instrument=instrument,
              _raise_on_error=False, verbose=verbose, **kwargs)


class LazyRV:
    def __init__(self, stars: list, instrument: str = None,
                 _parallel_limit=10):
        self.stars = stars
        if isinstance(self.stars, str):
            self.stars = [self.stars]
        self.instrument = instrument
        self._saved = None
        self._parallel_limit = _parallel_limit

    @property
    def N(self):
        return len(self.stars)

    def __repr__(self):
        return f"RV({self.N} stars)"

    def _get(self, **kwargs):
        if self.N > self._parallel_limit:
            # logger.info('Querying DACE...')
            _get_star = partial(get_star, instrument=self.instrument, **kwargs)
            with ThreadPool(8) as pool:
                result = list(tqdm(pool.imap(_get_star, self.stars), 
                                   total=self.N, unit='star',
                                   desc='Querying DACE (can take a while)'))
                print('')
        else:
            result = []
            logger.info('querying DACE...')
            pbar = tqdm(self.stars, total=self.N, unit='star')
            for star in pbar:
                pbar.set_description(star)
                result.append(get_star(star, self.instrument, **kwargs))

        return result

        # # use a with statement to ensure threads are cleaned up promptly
        # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        #     star_to_RV = {
        #         pool.submit(get_star, star, self.instrument): star 
        #         for star in self.stars
        #     }
        #     logger.info('Querying DACE...')
        #     pbar = tqdm(concurrent.futures.as_completed(star_to_RV),
        #                 total=self.N, unit='star')
        #     for future in pbar:
        #         star = star_to_RV[future]
        #         pbar.set_description(star)
        #         try:
        #             result.append(future.result())
        #         except ValueError:
        #             print(f'{star} generated an exception')
        #             result.append(None)
        # return result

    def reload(self, **kwargs):
        self._saved = self._get(**kwargs)
        return self._saved

    def __iter__(self):
        return self._get()

    def __call__(self, **kwargs):
        if not self._saved:
            self._saved = self._get(**kwargs)
        return self._saved

    @lru_cache(maxsize=10)
    def __getitem__(self, index):
        star = self.stars[index]
        if self._saved is not None:
            return self._saved[index]
        return get_star(star, self.instrument, verbose=True)


# sorted by spectral type
WG1_stars = [
    "HIP11533",  # F2
    "HD63077",  # F9
    "HD102365",  # G2
    "HD160691",  # G3
    "HD20794",  # G6
    "HD115617",  # G6
    "HD10700",  # G8
    "HD69830",  # G8
    "HD26965",  # K0
    "HD100623",  # K0
    "HD154088",  # K0
    "HD72673",  # K1
    "HD4628",  # K2
    "HD191408",  # K2
    "HD192310",  # K2
    "HD16160",  # K3
    "HD32147",  # K3
    "HD22496",  # K5
    "HIP93069",  # K5
    "HD209100",  # K5
    "HIP23708",  # K7
    "HD152606",  # K8
    "HD260655",  # M0
    "HIP40239",  # M0
    "HD304636",  # M0
    "HIP85647",  # M0
    "HD165222",  # M0
    "HD191849",  # M0
    "GJ191",  # M1
    "HD42581",  # M1
    "HIP42748",  # M1
    "HIP65859",  # M1
    "HIP86287",  # M1
    "HD176029",  # M1
    "GJ825",  # M1
    "HD225213",  # M2
    "GJ54",  # M2
    "HIP22627",  # M2
    "HIP51317",  # M2
    "HD119850",  # M2
    "GJ832",  # M2
    "HD217987",  # M2
    "GJ273",  # M3
    "GJ388",  # M3
    "HIP62452",  # M3
    "HIP67164",  # M3
    "HIP71253",  # M3
    "GJ628",  # M3
    "HIP85523",  # M3
    "HIP86214",  # M3
    "HIP92403",  # M3
    "HIP113020",  # M3
    "HIP1242",  # M4
    "GJ83.1",  # M4
    "HIP53020",  # M4
    "Ross128",  # M4
    "GJ699",  # M4
    "GJ1002",  # M5
    "GJ1061",  # M5
    "GJ3618",  # M5
    "Proxima",  # M5
    "GJ406",  # M6
]

ESPRESSO_GTO_nt = namedtuple('ESPRESSO_GTO', ['WG1', 'WG2', 'WG3'])
ESPRESSO_GTO = ESPRESSO_GTO_nt(
    WG1=LazyRV(WG1_stars, instrument='ESPRESSO'),
    WG2=LazyRV([], instrument='ESPRESSO'), # TODO
    WG3=LazyRV([], instrument='ESPRESSO'), # TODO
)
ESPRESSO_GTO.WG1.__doc__ = 'RV observations for all WG1 targets. Call ESPRESSO_GTO.WG1() to load them.'
ESPRESSO_GTO.WG2.__doc__ = 'RV observations for all WG2 targets. Call ESPRESSO_GTO.WG2() to load them.'
ESPRESSO_GTO.WG3.__doc__ = 'RV observations for all WG3 targets. Call ESPRESSO_GTO.WG3() to load them.'


import requests

def _get_NIRPS_GTO_stars(WP=1):
    from io import StringIO
    import numpy as np

    url = 'https://www.eso.org/sci/observing/teles-alloc/gto/113/NIRPS/P113_NIRPS-consortium.csv'
    file = StringIO(requests.get(url).content.decode())
    stars_P113 = np.loadtxt(file, delimiter=',', usecols=(0,), dtype=str, skiprows=3)
    
    url = 'https://www.eso.org/sci/observing/teles-alloc/gto/114/NIRPS/P114_NIRPS-consortium.csv'
    file = StringIO(requests.get(url).content.decode())
    stars_P114 = np.loadtxt(file, delimiter=',', usecols=(0,), dtype=str, skiprows=3)

    url = 'https://www.eso.org/sci/observing/teles-alloc/gto/115/NIRPS/P115_NIRPS-consortium.csv'
    file = StringIO(requests.get(url).content.decode())
    stars_P115 = np.loadtxt(file, delimiter=',', usecols=(0,), dtype=str, skiprows=3)

    def _get_stars_period(stars, WP):
        stars = np.delete(stars, stars=='')

        stars = np.char.replace(stars, '_', ' ')
        stars = np.char.replace(stars, "Proxima Centauri", "Proxima")
        stars = np.char.replace(stars, "Barnard's star", "GJ699")
        stars = np.char.replace(stars, "Teegarden's Star", 'Teegarden')

        if WP in (1, 'WP1'):
            wp1_indices = slice(np.where(stars == 'WP1')[0][0] + 1, np.where(stars == 'WP2')[0][0])
            return stars[wp1_indices]
        elif WP == 2:
            wp2_indices = slice(np.where(stars == 'WP2')[0][0] + 1, np.where(stars == 'WP3')[0][0])
            return stars[wp2_indices]
        elif WP == 3:
            wp3_indices = slice(np.where(stars == 'WP3')[0][0] + 1, np.where(stars == 'Other Science 1')[0][0])
            return stars[wp3_indices]
        elif WP == 'OS1':
            os1_indices = slice(np.where(stars == 'Other Science 1')[0][0] + 1, np.where(stars == 'Other Science 2')[0][0])
            return stars[os1_indices]
        elif WP == 'OS2':
            os2_indices = slice(np.where(stars == 'Other Science 2')[0][0] + 1, None)
            stars = np.char.replace(stars, 'MMU', 'No')
            stars = np.char.replace(stars, 'Cl*', '')
            return stars[os2_indices]
    
    stars_P113 = _get_stars_period(stars_P113, WP)
    stars_P114 = _get_stars_period(stars_P114, WP)
    stars_P115 = _get_stars_period(stars_P115, WP)
    return np.union1d(np.union1d(stars_P113, stars_P114), stars_P115)

try:
    NIRPS_GTO_WP1_stars = _get_NIRPS_GTO_stars(WP=1)
    NIRPS_GTO_WP2_stars = _get_NIRPS_GTO_stars(WP=2)
    NIRPS_GTO_WP3_stars = _get_NIRPS_GTO_stars(WP=3)
    NIRPS_GTO_OS1_stars = _get_NIRPS_GTO_stars(WP='OS1')
    NIRPS_GTO_OS2_stars = _get_NIRPS_GTO_stars(WP='OS2')
except requests.ConnectionError:
    from .setup_logger import logger
    logger.error('Cannot download NIRPS GTO protected target list')
else:
    NIRPS_GTO_nt = namedtuple('NIRPS_GTO', ['WP1', 'WP2', 'WP3', 'OS1', 'OS2'])
    NIRPS_GTO_nt.__doc__ = 'RV observations for all NIRPS GTO targets. See NIRPS_GTO.WP1, NIRPS_GTO.WP2, ...'
    NIRPS_GTO = NIRPS_GTO_nt(
        WP1=LazyRV(NIRPS_GTO_WP1_stars, instrument='NIRPS'),
        WP2=LazyRV(NIRPS_GTO_WP2_stars, instrument='NIRPS'),
        WP3=LazyRV(NIRPS_GTO_WP3_stars, instrument='NIRPS'),
        OS1=LazyRV(NIRPS_GTO_OS1_stars, instrument='NIRPS'),
        OS2=LazyRV(NIRPS_GTO_OS2_stars, instrument='NIRPS'),
    )
    NIRPS_GTO.WP1.__doc__ = 'RV observations for all WP1 targets. Call NIRPS_GTO.WP1() to load them.'
    NIRPS_GTO.WP2.__doc__ = 'RV observations for all WP2 targets. Call NIRPS_GTO.WP2() to load them.'
    NIRPS_GTO.WP3.__doc__ = 'RV observations for all WP3 targets. Call NIRPS_GTO.WP3() to load them.'
    NIRPS_GTO.OS1.__doc__ = 'RV observations for all OS1 targets. Call NIRPS_GTO.OS1() to load them.'
    NIRPS_GTO.OS2.__doc__ = 'RV observations for all OS2 targets. Call NIRPS_GTO.OS2() to load them.'
