import os
import sys
import time
from contextlib import contextmanager
from functools import partial
from collections import defaultdict

try:
    from unittest.mock import patch
except ImportError:
    try:
        from mock import patch
    except ImportError as e:
        raise e

import subprocess
import logging
from glob import glob
import numpy as np

try:
    from tqdm import tqdm, trange
except ImportError:
    tqdm = lambda x, *args, **kwargs: x
    trange = lambda *args, **kwargs: range(*args, **kwargs)

from .setup_logger import setup_logger
from .config import config


def create_directory(directory):
    """ Create a directory if it does not exist """
    if not os.path.isdir(directory):
        os.makedirs(directory)

@contextmanager
def chdir(dir):
    """ A simple context manager to switch directories temporarily """
    curdir = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(curdir)

@contextmanager
def stdout_disabled():
    devnull = open(os.devnull, 'w')
    with patch('sys.stdout', devnull):
        yield

@contextmanager
def all_logging_disabled():
    """
    A context manager that will prevent any logging messages triggered during
    the body from being processed.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(logging.CRITICAL)

    try:
        yield
    finally:
        logging.disable(previous_level)


class record_removals:
    def __init__(self, s, storage=None):
        """
        A simple context manager to record removed files

        Args:
            s (RV):
                An `RV` object
            storage (dict): 
                A dictionary to store the removed files, with keys 'raw_file'
                and 'reason' as lists.
        
        Examples:
            >>> with record_removals(s) as rec:
              :    s.remove_instrument('HARPS')
              :    rec.store('removed HARPS')
            >>> rec.storage
        """
        self.s = s
        if storage is None:
            self.storage = defaultdict(list)
        else:
            if 'raw_file' not in storage:
                storage['raw_file'] = []
            if 'reason' not in storage:
                storage['reason'] = []
            self.storage = storage
        self.raw_file_start = self.s.raw_file.copy()

    def store(self, reason):
        missing = ~ np.isin(self.raw_file_start, self.s.raw_file[self.s.mask])
        if missing.any():
            lost = self.raw_file_start[missing]
            self.storage['raw_file'].extend(lost)
            self.storage['reason'].extend(len(lost) * [reason])
            self.raw_file_start = self.s.raw_file[self.s.mask].copy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


@contextmanager
def timer(name=None):
    """ A simple context manager to time a block of code """
    logger = setup_logger()

    if not config.debug:
        yield
        return

    if name is None:
        logger.debug('starting timer')
    else:
        logger.debug(f'{name}: starting timer')

    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        if name is None:
            logger.debug(f'elapsed time {end - start:.2f} seconds')
        else:
            logger.debug(f'{name}: elapsed time {end - start:.2f} seconds')


def sanitize_path(path):
    if os.name == 'nt':  # on Windows, be careful with ':' in filename
        path = path.replace(':', '_')
    path = path.replace('*', '_')
    return path

def pretty_print_table(rows, line_between_rows=True, string=False, 
                       markdown=False, latex=False, logger=None):
    r"""
    Example Output
    ┌──────┬─────────────┬────┬───────┐
    │ True │ short       │ 77 │ catty │
    ├──────┼─────────────┼────┼───────┤
    │ 36   │ long phrase │ 9  │ dog   │
    ├──────┼─────────────┼────┼───────┤
    │ 8    │ medium      │ 3  │ zebra │
    └──────┴─────────────┴────┴───────┘
    """
    _print = logger.info if logger else print
    if string:
        def _print(x, s):
            s += x + '\n'
            return s
    else:
        if logger:
            def _print(x, _):
                logger.info(x)
        else:
            def _print(x, _):
                print(x)

    if latex or markdown:
        line_between_rows = False

    s = ''

    # find the max length of each column
    max_col_lens = list(map(max, zip(*[(len(str(cell)) for cell in row) for row in rows])))

    if markdown:
        bar_char = '|'
    else:
        bar_char = r'│'

    # print the table's top border
    if markdown:
        pass
    elif latex:
        s = _print(r'\begin{table*}', s)
        # s = _print(r'\centering', s)
        s = _print(r'\begin{tabular}' + '{' + ' c ' * len(rows[0]) + '}', s)
    else:
        s = _print(r'┌' + r'┬'.join(r'─' * (n + 2) for n in max_col_lens) + r'┐', s)

    if markdown:
        header_separator = bar_char + bar_char.join('-' * (n + 2) for n in max_col_lens) + bar_char

    rows_separator = r'├' + r'┼'.join(r'─' * (n + 2) for n in max_col_lens) + r'┤'

    if latex:
        row_fstring = ' & '.join("{: <%s}" % n for n in max_col_lens)
    else:
        row_fstring = bar_char.center(3).join("{: <%s}" % n for n in max_col_lens)

    for i, row in enumerate(rows):
        if markdown and i == 1:
            s = _print(header_separator, s)

        if latex:
            s = _print(row_fstring.format(*map(str, row)) + r' \\', s)
        else:
            s = _print(bar_char + ' ' + row_fstring.format(*map(str, row)) + ' ' + bar_char, s)
        

        if line_between_rows and i < len(rows) - 1:
            s = _print(rows_separator, s)


    # print the table's bottom border
    if markdown:
        pass
    elif latex:
        s = _print(r'\end{tabular}', s)
        s = _print(r'\end{table*}', s)
    else:
        s = _print(r'└' + r'┴'.join(r'─' * (n + 2) for n in max_col_lens) + r'┘', s)

    if string:
        return s


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value {!r}".format(val))

def there_is_internet(timeout=1):
    from socket import create_connection
    try:
        create_connection(('8.8.8.8', 53), timeout=timeout)
        return True
    except OSError:
        pass
    return False

def get_data_path():
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, 'data')
    return data_path

def find_data_file(file):
    here = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(here, '..', 'data', file)

    if '*' in data_file:
        files = sorted(glob(data_file))
        if len(files) > 0:
            return files

    if not os.path.exists(data_file):
        data_file = os.path.join(here, 'data', file)
        if '*' in data_file:
            data_file = sorted(glob(data_file))

    return data_file


import importlib.util
import sys
def lazy_import(name):
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def ESPRESSO_ADC_issues():
    adc_file = find_data_file('obs_affected_ADC_issues.dat')
    lines = [line.strip() for line in open(adc_file).readlines()]
    file_roots = [line.split()[1] for line in lines if not line.startswith('#')]
    return np.array(file_roots)

def ESPRESSO_cryostat_issues():
    cryostat_file = find_data_file('obs_affected_blue_cryostat_issues.dat')
    lines = [line.strip() for line in open(cryostat_file).readlines()]
    file_roots = [line.split()[1] for line in lines if not line.startswith('#')]
    return np.array(file_roots)


def get_ra_sexagesimal(ra):
    """ Convert RA in degrees to sexagesimal string representation. """
    from astropy.coordinates import Angle
    from astropy import units as u
    return Angle(ra, unit=u.deg).to(u.hourangle).to_string(sep=':', pad=True)

def get_dec_sexagesimal(dec):
    """ Convert DEC in degrees to sexagesimal string representation. """
    from astropy.coordinates import Angle
    from astropy import units as u
    return Angle(dec, unit=u.deg).to_string(sep=':', pad=True)

def get_max_berv_span(self, n=None):
    """
    Return the indices of the n observations which maximize the BERV span.
    If n is None, return all indices sorted by BERV span.    
    """
    berv_argsort = np.argsort(self.berv)

    # if n is None:
    #     n = self.N // 2
    inds = []
    for b1, b2 in zip(berv_argsort[:self.N // 2], berv_argsort[::-1]):
        inds.append(b1)
        inds.append(b2)
    return np.array(inds[:n])

def get_object_fast(file):
    with open(file, 'rb') as f:
        f.read(800) # read first 10 keywords
        key = f.read(8)
        assert key == b'OBJECT  ', 'Object keyword not found.'
        f.read(2)
        value = f.read(20)
    return value.decode().split("'")[1].strip()


def get_simbad_oid(self):
    import requests
    if isinstance(self, str):
        star = self
    else:
        star = self.star
    oid = requests.post('https://simbad.cds.unistra.fr/simbad/sim-tap/sync', 
                        data=dict(format='text', request='doQuery', lang='adql', phase='run', 
                                  query=f"SELECT basic.OID FROM basic JOIN ident ON oidref = oid WHERE id = '{star}';"))
    oid = oid.text.split()[-1]
    return oid



# from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
