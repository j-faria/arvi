import os
from contextlib import contextmanager
try:
    from unittest.mock import patch
except ImportError:
    try:
        from mock import patch
    except ImportError as e:
        raise e

import logging
from glob import glob
from numpy import array


def create_directory(directory):
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

def ESPRESSO_ADC_issues():
    adc_file = find_data_file('obs_affected_ADC_issues.dat')
    lines = [line.strip() for line in open(adc_file).readlines()]
    file_roots = [line.split()[1] for line in lines if not line.startswith('#')]
    return array(file_roots)

def ESPRESSO_cryostat_issues():
    cryostat_file = find_data_file('obs_affected_blue_cryostat_issues.dat')
    lines = [line.strip() for line in open(cryostat_file).readlines()]
    file_roots = [line.split()[1] for line in lines if not line.startswith('#')]
    return array(file_roots)
