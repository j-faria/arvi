import os
import tarfile
import collections
import numpy as np
from dace_query import DaceClass
from dace_query.spectroscopy import SpectroscopyClass, Spectroscopy as default_Spectroscopy
from .setup_logger import logger
from .utils import create_directory, all_logging_disabled, stdout_disabled


def load_spectroscopy():
    if 'DACERC' in os.environ:
        dace = DaceClass(dace_rc_config_path=os.environ['DACERC'])
        return SpectroscopyClass(dace_instance=dace)
    # elif os.path.exists(os.path.expanduser('~/.dacerc')):
    return default_Spectroscopy

def get_arrays(result, latest_pipeline=True, ESPRESSO_mode='HR11', NIRPS_mode='HE', verbose=True):
    arrays = []
    instruments = list(result.keys())

    # if verbose:
    #     if latest_pipeline:
    #         logger.info('selecting latest pipeline version')

    for inst in instruments:
        pipelines = list(result[inst].keys())

        # select ESPRESSO mode, which is defined at the level of the pipeline
        if 'ESPRESSO' in inst:
            if any(ESPRESSO_mode in pipe for pipe in pipelines):
                if verbose:
                    logger.info(f'selecting mode {ESPRESSO_mode} for ESPRESSO')
                i = [i for i, pipe in enumerate(pipelines) if ESPRESSO_mode in pipe][0]
                pipelines = [pipelines[i]]
            else:
                if verbose:
                    logger.warning(f'no observations for requested ESPRESSO mode ({ESPRESSO_mode})')

        if latest_pipeline:
            if verbose and len(pipelines) > 1:
                logger.info(f'selecting latest pipeline ({pipelines[0]}) for {inst}')

            pipelines = [pipelines[0]]

        for pipe in pipelines:
            modes = list(result[inst][pipe].keys())

            # select NIRPS mode, which is defined at the level of the mode
            if 'NIRPS' in inst:
                if NIRPS_mode in modes:
                    if verbose:
                        logger.info(f'selecting mode {NIRPS_mode} for NIRPS')
                    i = modes.index(NIRPS_mode)
                    modes = [modes[i]]
                else:
                    if verbose:
                        logger.warning(f'no observations for requested NIRPS mode ({NIRPS_mode})')

            for mode in modes:
                if 'rjd' not in result[inst][pipe][mode]:
                    logger.error(f"No 'rjd' key for {inst} - {pipe}")
                    raise ValueError

                arrays.append(
                    ((inst, pipe, mode), result[inst][pipe][mode])
                )

    return arrays

def get_observations(star, instrument=None, verbose=True):
    Spectroscopy = load_spectroscopy()
    try:
        with stdout_disabled(), all_logging_disabled():
            result = Spectroscopy.get_timeseries(target=star,
                                                sorted_by_instrument=True,
                                                output_format='numpy')
    except TypeError:
        if instrument is None:
            msg = f'no observations for {star}'
        else:
            msg = f'no {instrument} observations for {star}'
        raise ValueError(msg) from None

    # defaultdict --> dict
    if isinstance(result, collections.defaultdict):
        result = dict(result)
    for inst in result.keys():
        for pipe in result[inst].keys():
            for mode in result[inst][pipe].keys():
                if isinstance(result[inst][pipe][mode], collections.defaultdict):
                    result[inst][pipe][mode] = dict(result[inst][pipe][mode])
            if isinstance(result[inst][pipe], collections.defaultdict):
                    result[inst][pipe] = dict(result[inst][pipe])
        if isinstance(result[inst], collections.defaultdict):
            result[inst] = dict(result[inst])
    #

    instruments = list(result.keys())

    if instrument is not None:
        # select only the provided instrument (if it's there)
        instruments = [inst for inst in instruments if instrument in inst]

    if len(instruments) == 0:
        if instrument is None:
            msg = f'no observations for {star}'
        else:
            msg = f'no {instrument} observations for {star}'
        raise ValueError(msg)

    # sort pipelines, being extra careful with HARPS pipeline names
    # (i.e. ensure that 3.0.0 > 3.5)
    def cmp(a, b):
        if a[0] in ('3.5', '3.5 EGGS') and b[0] == '3.0.0':
            return -1
        if b[0] in ('3.5', '3.5 EGGS') and a[0] == '3.0.0':
            return 1

        if a[0] == b[0]:
            return 0
        elif a[0] > b[0]:
            return 1
        else:
            return -1

    from functools import cmp_to_key
    new_result = {}
    for inst in instruments:
        new_result[inst] = dict(sorted(result[inst].items(),
                                       key=cmp_to_key(cmp), reverse=True))

    if verbose:
        logger.info('RVs available from')
        with logger.contextualize(indent='  '):
            _inst = ''
            for inst in instruments:
                pipelines = list(new_result[inst].keys())
                for pipe in pipelines:
                    modes = list(new_result[inst][pipe].keys())
                    for mode in modes:
                        N = len(new_result[inst][pipe][mode]['rjd'])
                        # LOG
                        if inst == _inst:
                            logger.info(f'{" ":>12s} └ {pipe} - {mode} ({N} observations)')
                        else:
                            logger.info(f'{inst:>12s} ├ {pipe} - {mode} ({N} observations)')
                        _inst = inst

    return new_result


def check_existing(output_directory, files, type):
    existing = [
        f.partition('.fits')[0] for f in os.listdir(output_directory)
        if type in f
    ]
    if os.name == 'nt':  # on Windows, be careful with ':' in filename
        import re
        existing = [re.sub(r'T(\d+)_(\d+)_(\d+)', r'T\1:\2:\3', f) for f in existing]
    
    # remove type of file (e.g. _CCF_A)
    existing = [f.partition('_')[0] for f in existing]
    
    missing = []
    for file in files:
        if any(other in file for other in existing):
            continue
        missing.append(file)

    return np.array(missing)

def download(files, type, output_directory):
    
    Spectroscopy = load_spectroscopy()
    # with stdout_disabled(), all_logging_disabled():
    Spectroscopy.download_files(files, file_type=type,
                                output_directory=output_directory)

def extract_fits(output_directory):
    file = os.path.join(output_directory, 'spectroscopy_download.tar.gz')
    with tarfile.open(file, "r") as tar:
        files = []
        for member in tar.getmembers():
            if member.isreg():  # skip if the TarInfo is not a file
                member.name = os.path.basename(member.name)  # remove the path
                if os.name == 'nt':  # on Windows, be careful with ':' in filename
                    member.name = member.name.replace(':', '_')
                tar.extract(member, output_directory)
                files.append(member.name)
    os.remove(file)
    return files


def do_download_ccf(raw_files, output_directory, clobber=False, verbose=True):
    raw_files = np.atleast_1d(raw_files)

    create_directory(output_directory)

    # check existing files to avoid re-downloading
    if not clobber:
        raw_files = check_existing(output_directory, raw_files, 'CCF')

    # any file left to download?
    if raw_files.size == 0:
        if verbose:
            logger.info('no files to download')
        return

    if verbose:
        n = raw_files.size
        logger.info(f"Downloading {n} CCFs into '{output_directory}'...")

    download(raw_files, 'ccf', output_directory)

    if verbose:
        logger.info('Extracting .fits files')

    extract_fits(output_directory)


def do_download_s1d(raw_files, output_directory, clobber=False, verbose=True):
    raw_files = np.atleast_1d(raw_files)

    create_directory(output_directory)

    # check existing files to avoid re-downloading
    if not clobber:
        raw_files = check_existing(output_directory, raw_files, 'S1D')

    # any file left to download?
    if raw_files.size == 0:
        if verbose:
            logger.info('no files to download')
        return

    if verbose:
        n = raw_files.size
        logger.info(f"Downloading {n} S1Ds into '{output_directory}'...")

    download(raw_files, 's1d', output_directory)

    if verbose:
        logger.info('Extracting .fits files')

    extract_fits(output_directory)


def do_download_s2d(raw_files, output_directory, clobber=False, verbose=True):
    raw_files = np.atleast_1d(raw_files)

    create_directory(output_directory)

    # check existing files to avoid re-downloading
    if not clobber:
        raw_files = check_existing(output_directory, raw_files, 'S2D')

    # any file left to download?
    if raw_files.size == 0:
        if verbose:
            logger.info('no files to download')
        return

    if verbose:
        n = raw_files.size
        logger.info(f"Downloading {n} S2Ds into '{output_directory}'...")

    download(raw_files, 's2d', output_directory)

    if verbose:
        logger.info('Extracting .fits files')

    extracted_files = extract_fits(output_directory)
    return extracted_files
