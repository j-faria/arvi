import os
import tarfile
import collections
from functools import lru_cache
import numpy as np
from dace_query import DaceClass
from dace_query.spectroscopy import SpectroscopyClass, Spectroscopy as default_Spectroscopy
from .setup_logger import logger
from .utils import create_directory, all_logging_disabled, stdout_disabled, tqdm


def load_spectroscopy() -> SpectroscopyClass:
    from .config import request_as_public
    if request_as_public:
        with all_logging_disabled():
            dace = DaceClass(dace_rc_config_path='none')
        return SpectroscopyClass(dace_instance=dace)
    if 'DACERC' in os.environ:
        dace = DaceClass(dace_rc_config_path=os.environ['DACERC'])
        return SpectroscopyClass(dace_instance=dace)
    # elif os.path.exists(os.path.expanduser('~/.dacerc')):
    return default_Spectroscopy

@lru_cache()
def get_dace_id(star):
    filters = {"obj_id_catname": {"equal": [star]}}
    try:
        with stdout_disabled(), all_logging_disabled():
            r = load_spectroscopy().query_database(filters=filters, limit=1)
        return r['obj_id_daceid'][0]
    except KeyError:
        logger.error(f"Could not find DACE ID for {star}")
        raise ValueError from None

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
                if len(pipelines) > 1 and verbose:
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

def get_observations_from_instrument(star, instrument, main_id=None):
    """ Query DACE for all observations of a given star and instrument

    Args:
        star (str): name of the star
        instrument (str): instrument name
        main_id (str, optional): Simbad main id of target to query DACE id. Defaults to None.

    Raises:
        ValueError: If query for DACE id fails

    Returns:
        dict: dictionary with data from DACE
    """
    try:
        dace_id = get_dace_id(star)
    except ValueError as e:
        if main_id is not None:
            dace_id = get_dace_id(main_id)
        else:
            raise e

    Spectroscopy = load_spectroscopy()
    filters = {
        "ins_name": {"contains": [instrument]}, 
        "obj_id_daceid": {"contains": [dace_id]}
    }
    with stdout_disabled(), all_logging_disabled():
        result = Spectroscopy.query_database(filters=filters)
    
    if len(result) == 0:
        raise ValueError

    r = {}
    for inst in np.unique(result['ins_name']):
        mask1 = result['ins_name'] == inst
        r[inst] = {}
        for pipe in np.unique(result['ins_drs_version'][mask1]):
            mask2 = mask1 & (result['ins_drs_version'] == pipe)
            ins_mode = np.unique(result['ins_mode'][mask2])[0]
            _nan = np.full(mask2.sum(), np.nan)
            r[inst][pipe] = {
                ins_mode: {
                    'texp': result['texp'][mask2],
                    'bispan': result['spectro_ccf_bispan'][mask2],
                    'bispan_err': result['spectro_ccf_bispan_err'][mask2],
                    'drift_noise': result['spectro_cal_drift_noise'][mask2],
                    'rjd': result['obj_date_bjd'][mask2],
                    'cal_therror': _nan,
                    'fwhm': result['spectro_ccf_fwhm'][mask2],
                    'fwhm_err': result['spectro_ccf_fwhm_err'][mask2],
                    'rv': result['spectro_ccf_rv'][mask2],
                    'rv_err': result['spectro_ccf_rv_err'][mask2],
                    'berv': result['spectro_cal_berv'][mask2],
                    'ccf_noise': _nan,
                    'rhk': result['spectro_analysis_rhk'][mask2],
                    'rhk_err': result['spectro_analysis_rhk_err'][mask2],
                    'contrast': result['spectro_ccf_contrast'][mask2],
                    'contrast_err': result['spectro_ccf_contrast_err'][mask2],
                    'cal_thfile': result['spectro_cal_thfile'][mask2],
                    'spectroFluxSn50': result['spectro_flux_sn50'][mask2],
                    'protm08': result['spectro_analysis_protm08'][mask2],
                    'protm08_err': result['spectro_analysis_protm08_err'][mask2],
                    'caindex': result['spectro_analysis_ca'][mask2],
                    'caindex_err': result['spectro_analysis_ca_err'][mask2],
                    'pub_reference': result['pub_ref'][mask2],
                    'drs_qc': result['spectro_drs_qc'][mask2],
                    'haindex': result['spectro_analysis_halpha'][mask2],
                    'haindex_err': result['spectro_analysis_halpha_err'][mask2],
                    'protn84': result['spectro_analysis_protn84'][mask2],
                    'protn84_err': result['spectro_analysis_protn84_err'][mask2],
                    'naindex': result['spectro_analysis_na'][mask2],
                    'naindex_err': result['spectro_analysis_na_err'][mask2],
                    'snca2': _nan,
                    'mask': result['spectro_ccf_mask'][mask2],
                    'public': result['public'][mask2],
                    'spectroFluxSn20': result['spectro_flux_sn20'][mask2],
                    'sindex': result['spectro_analysis_smw'][mask2],
                    'sindex_err': result['spectro_analysis_smw_err'][mask2],
                    'drift_used': _nan,
                    'ccf_asym': result['spectro_ccf_asym'][mask2],
                    'ccf_asym_err': result['spectro_ccf_asym_err'][mask2],
                    'date_night': result['date_night'][mask2],
                    'raw_file': result['file_rootpath'][mask2],
                    'prog_id': result['prog_id'][mask2],
                    'th_ar': result['th_ar'][mask2],
                    'th_ar1': result['th_ar1'][mask2],
                    'th_ar2': result['th_ar2'][mask2],
                }
            }
    return r

def get_observations(star, instrument=None, main_id=None, verbose=True):
    if instrument is None:
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
    else:
        try:
            result = get_observations_from_instrument(star, instrument, main_id)
        except ValueError:
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
    """ Check how many of `files` exist in `output_directory` """
    existing = [
        f.partition('.fits')[0] for f in os.listdir(output_directory)
        if type in f
    ]

    # also check for lowercase type
    existing += [
        f.partition('.fits')[0] for f in os.listdir(output_directory)
        if type.lower() in f
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
    """ Download files from DACE """
    Spectroscopy = load_spectroscopy()
    with stdout_disabled(), all_logging_disabled():
        Spectroscopy.download_files(files, file_type=type.lower(),
                                    output_directory=output_directory)

def extract_fits(output_directory):
    """ Extract fits files from tar.gz file """
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


def do_symlink_filetype(type, raw_files, output_directory, clobber=False, top_level=None, verbose=True):
    terminations = {
        'CCF': '_CCF_A.fits',
        'S1D': '_S1D_A.fits',
        'S2D': '_S2D_A.fits',
    }

    create_directory(output_directory)

    raw_files = np.atleast_1d(raw_files)

    # check existing files
    if not clobber:
        raw_files = check_existing(output_directory, raw_files, type)

    n = raw_files.size

    # any file left?
    if n == 0:
        if verbose:
            logger.info('no files to symlink')
        return

    if verbose:
        msg = f"symlinking {n} {type}s into '{output_directory}'..."
        logger.info(msg)

    for file in tqdm(raw_files):
        if top_level is not None:
            top = file.split('/')[0] + '/'
            if not top_level.endswith('/'):
                top_level = top_level + '/'
            file = file.replace(top, top_level)

        file = file.replace('.fits', terminations[type])

        if os.path.exists(file):
            os.symlink(file, os.path.join(output_directory, os.path.basename(file)))
            # print(file, os.path.join(output_directory, os.path.basename(file)))
        else:
            logger.warning(f'file not found: {file}')


def do_download_filetype(type, raw_files, output_directory, clobber=False,
                         verbose=True, chunk_size=20):
    """ Download CCFs / S1Ds / S2Ds from DACE """
    raw_files = np.atleast_1d(raw_files)

    create_directory(output_directory)

    # check existing files to avoid re-downloading
    if not clobber:
        raw_files = check_existing(output_directory, raw_files, type)

    n = raw_files.size

    # any file left to download?
    if n == 0:
        if verbose:
            logger.info('no files to download')
        return

    # avoid an empty chunk
    if chunk_size > n:
        chunk_size = n

    if verbose:
        if chunk_size < n:
            msg = f"Downloading {n} {type}s "
            msg += f"(in chunks of {chunk_size}) "
            msg += f"into '{output_directory}'..."
            logger.info(msg)
        else:
            msg = f"Downloading {n} {type}s into '{output_directory}'..."
            logger.info(msg)

    iterator = [raw_files[i:i + chunk_size] for i in range(0, n, chunk_size)]
    for files in tqdm(iterator, total=len(iterator)):
        download(files, type, output_directory)
        extract_fits(output_directory)

    logger.info('Extracted .fits files')


# def do_download_s1d(raw_files, output_directory, clobber=False, verbose=True):
#     """ Download S1Ds from DACE """
#     raw_files = np.atleast_1d(raw_files)

#     create_directory(output_directory)

#     # check existing files to avoid re-downloading
#     if not clobber:
#         raw_files = check_existing(output_directory, raw_files, 'S1D')

#     # any file left to download?
#     if raw_files.size == 0:
#         if verbose:
#             logger.info('no files to download')
#         return

#     if verbose:
#         n = raw_files.size
#         logger.info(f"Downloading {n} S1Ds into '{output_directory}'...")

#     download(raw_files, 's1d', output_directory)

#     if verbose:
#         logger.info('Extracting .fits files')

#     extract_fits(output_directory)


# def do_download_s2d(raw_files, output_directory, clobber=False, verbose=True):
#     """ Download S2Ds from DACE """
#     raw_files = np.atleast_1d(raw_files)

#     create_directory(output_directory)

#     # check existing files to avoid re-downloading
#     if not clobber:
#         raw_files = check_existing(output_directory, raw_files, 'S2D')

#     # any file left to download?
#     if raw_files.size == 0:
#         if verbose:
#             logger.info('no files to download')
#         return

#     if verbose:
#         n = raw_files.size
#         logger.info(f"Downloading {n} S2Ds into '{output_directory}'...")

#     download(raw_files, 's2d', output_directory)

#     if verbose:
#         logger.info('Extracting .fits files')

#     extracted_files = extract_fits(output_directory)
#     return extracted_files
