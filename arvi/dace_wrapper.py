import os
import tarfile
import numpy as np
from dace_query import DaceClass
from dace_query.spectroscopy import SpectroscopyClass, Spectroscopy as default_Spectroscopy
from .setup_logger import logger
from .utils import create_directory


def load_spectroscopy():
    if 'DACERC' in os.environ:
        dace = DaceClass(dace_rc_config_path=os.environ['DACERC'])
        return SpectroscopyClass(dace_instance=dace)
    # elif os.path.exists(os.path.expanduser('~/.dacerc')):
    return default_Spectroscopy

def get_arrays(result, latest_pipeline=True):
    arrays = []
    instruments = list(result.keys())
    for inst in instruments:
        pipelines = list(result[inst].keys())
        if latest_pipeline:
            pipelines = [pipelines[0]]
        for pipe in pipelines:
            # print(inst, pipe)
            modes = list(result[inst][pipe].keys())
            for mode in modes:
                if 'rjd' not in result[inst][pipe][mode]:
                    logger.error(f"No 'rjd' key for {inst} - {pipe}")
                    raise ValueError

                arrays.append(
                    ((inst, pipe, mode), result[inst][pipe][mode])
                )

    return arrays

def get_observations(star, save_rdb=False, verbose=True):
    Spectroscopy = load_spectroscopy()
    result = Spectroscopy.get_timeseries(target=star,
                                         sorted_by_instrument=True,
                                         output_format='numpy')
    instruments = list(result.keys())

    # sort pipelines, being extra careful with HARPS pipeline names
    # (i.e. ensure that 3.0.0 > 3.5)
    class sorter:
        def __call__(self, x):
            return ('0.3.5', x[1]) if x[0] == '3.5' else x

    for inst in instruments:
        # print(inst, result[inst].keys())
        result[inst] = dict(sorted(result[inst].items(),
                                   key=sorter(), reverse=True))
        # print(inst, result[inst].keys())

    if verbose:
        logger.info('RVs available from')
        with logger.contextualize(indent='   '):
            for inst in instruments:
                pipelines = list(result[inst].keys())
                for pipe in pipelines:
                    mode = list(result[inst][pipe].keys())[0]
                    N = len(result[inst][pipe][mode]['rjd'])
                    # LOG
                    logger.info(f'{inst:12s} {pipe:10s} ({N} observations)')

    return result


def check_existing(output_directory, files, type):
    existing = [
        f.partition('_')[0] for f in os.listdir(output_directory)
        if type in f
    ]
    missing = []
    for file in files:
        if any(other in file for other in existing):
            continue
        missing.append(file)
    return np.array(missing)

def download(files, type, output_directory):
    from .utils import all_logging_disabled, stdout_disabled
    Spectroscopy = load_spectroscopy()
    with stdout_disabled(), all_logging_disabled():
        Spectroscopy.download_files(files, file_type=type,
                                    output_directory=output_directory)

def extract_fits(output_directory):
    file = os.path.join(output_directory, 'spectroscopy_download.tar.gz')
    tar = tarfile.open(file, "r")
    for member in tar.getmembers():
        if member.isreg():  # skip if the TarInfo is not a file
            member.name = os.path.basename(member.name)  # remove the path
            tar.extract(member, output_directory)
    os.remove(file)

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

    extract_fits(output_directory)
