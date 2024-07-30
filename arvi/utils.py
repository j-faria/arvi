import os
import time
from contextlib import contextmanager
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

from .setup_logger import logger
from . import config


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


@contextmanager
def timer():
    """ A simple context manager to time a block of code """
    if not config.debug:
        yield
        return

    logger.debug(f'starting timer')
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        logger.debug(f'elapsed time: {end - start:.2f} seconds')


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


