import os
import multiprocessing
from functools import partial
from itertools import chain
from collections import namedtuple
from tqdm import tqdm
# import numpy as np

from .setup_logger import logger
from .timeseries import RV

__all__ = ['ESPRESSO_GTO']

path = os.path.join(os.path.dirname(__file__), 'data')


def get_star(star, instrument=None):
    return RV(star, instrument=instrument,
              _raise_on_error=False, verbose=False, load_extra_data=False)


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

    def _get(self):
        if self.N > self._parallel_limit:
            # logger.info('Querying DACE...')
            _get_star = partial(get_star, instrument=self.instrument)
            with multiprocessing.Pool() as pool:
                result = list(tqdm(pool.imap(_get_star, self.stars), 
                                   total=self.N, unit='star', desc='Querying DACE'))
                # result = pool.map(get_star, self.stars)
        else:
            result = []
            logger.info('Querying DACE...')
            pbar = tqdm(self.stars, total=self.N, unit='star')
            for star in pbar:
                pbar.set_description(star)
                result.append(get_star(star, self.instrument))

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

    def __iter__(self):
        return self._get()

    def __call__(self):
        if not self._saved:
            self._saved = self._get()
        return self._saved


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
