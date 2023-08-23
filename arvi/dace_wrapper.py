import os
import tarfile
import numpy as np
from dace_query import DaceClass
from dace_query.spectroscopy import SpectroscopyClass, Spectroscopy as default_Spectroscopy
from .setup_logger import logger

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
            pipelines = [pipelines[-1]]
        for pipe in pipelines:
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
            return '0.3.5' if x == '3.5' else x

    for inst in instruments:
        result[inst] = dict(sorted(result[inst].items(), 
                                   key=sorter(), reverse=True))

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


def do_download_ccf(raw_files, output_directory, verbose=True):
    raw_files = np.atleast_1d(raw_files)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    if verbose:
        logger.info(f"Downloading {len(raw_files)} CCFs into '{output_directory}'...")

    Spectroscopy = load_spectroscopy()
    
    from .utils import all_logging_disabled, stdout_disabled
    with stdout_disabled(), all_logging_disabled():
        Spectroscopy.download_files(raw_files[:2],
                                    file_type='ccf',
                                    output_directory=output_directory)

    if verbose:
        logger.info('Extracting .fits files')
    
    file = os.path.join(output_directory, 'spectroscopy_download.tar.gz')
    tar = tarfile.open(file, "r")
    for member in tar.getmembers():
        if member.isreg():  # skip if the TarInfo is not a file
            member.name = os.path.basename(member.name)  # remove the path
            tar.extract(member, output_directory)
    os.remove(file)
