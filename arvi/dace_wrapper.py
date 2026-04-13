import os
import sys
import tarfile
import collections
from functools import lru_cache, partial
from itertools import islice
import numpy as np

from .setup_logger import setup_logger
from .utils import create_directory, all_logging_disabled, stdout_disabled, timer, tqdm


def load_spectroscopy(user=None, verbose=True):
    logger = setup_logger()
    with all_logging_disabled():
        from dace_query.spectroscopy import SpectroscopyClass, Spectroscopy as default_Spectroscopy
        from dace_query import DaceClass

    from .config import config
    # requesting as public
    if config.request_as_public:
        if verbose:
            logger.warning('requesting DACE data as public')
        with all_logging_disabled():
            dace = DaceClass(dace_rc_config_path='none')
        return SpectroscopyClass(dace_instance=dace)
    # path to DACERC file in config
    if config.dacerc_path != '' and user is None:
        if verbose:
            logger.info(f'using credentials from {config.dacerc_path}')
        dace = DaceClass(dace_rc_config_path=config.dacerc_path)
        return SpectroscopyClass(dace_instance=dace)
    # DACERC environment variable is set, should point to a dacerc file with credentials
    if 'DACERC' in os.environ and user is None:
        dace = DaceClass(dace_rc_config_path=os.environ['DACERC'])
        return SpectroscopyClass(dace_instance=dace)
    # user provided, should be a section in ~/.dacerc
    if user is not None:
        import configparser
        import tempfile
        config = configparser.ConfigParser()
        config.read(os.path.expanduser('~/.dacerc'))
        if user not in config.sections():
            raise ValueError(f'Section for user "{user}" not found in ~/.dacerc')
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            new_config = configparser.ConfigParser()
            new_config['user'] = config[user]
            new_config.write(f)
        dace = DaceClass(dace_rc_config_path=f.name)
        if verbose:
            logger.info(f'using credentials for user {user} in ~/.dacerc')
        return SpectroscopyClass(dace_instance=dace)
    # default
    if not os.path.exists(os.path.expanduser('~/.dacerc')):
        logger.warning('requesting DACE data as public (no .dacerc file found)')
    return default_Spectroscopy


@lru_cache(maxsize=1024)
def get_dace_id(star, verbose=True, raise_error=False):
    logger = setup_logger()
    filters = {"obj_id_catname": {"equal": [star]}}
    try:
        with all_logging_disabled():
            r = load_spectroscopy().query_database(filters=filters, limit=1)
        return str(r['obj_id_daceid'][0])
    except KeyError:
        if verbose:
            logger.error(f"Could not find DACE ID for {star}")
        if not raise_error:
            return None
        raise ValueError from None

def get_arrays(result, only_latest_pipeline=True,
               ESPRESSO_mode='SINGLEHR11', NIRPS_mode='HE', verbose=True):

    logger = setup_logger()
    arrays = []
    instruments = [str(i) for i in result.keys()]

    for inst in instruments:
        pipelines = [str(p) for p in result[inst].keys()]
        npipe = len(pipelines)

        if only_latest_pipeline:
            latest = sorted(pipelines)[-1]
            if verbose and npipe > 1:
                logger.info(f'selecting latest pipeline ({latest}) for {inst}')
            pipelines = [pipelines[pipelines.index(latest)]]

        for pipe in pipelines:
            modes = [m for m in result[inst][pipe].keys()]

            # ESPRESSO and NIRPS have a "preferred" mode
            for m, i in (
                (ESPRESSO_mode, 'ESPRESSO'),
                (NIRPS_mode, 'NIRPS'),
            ):
                if i in inst and len(modes) > 1:
                    if m in modes:
                        if verbose:
                            logger.info(f'selecting mode {m} for {inst} - {pipe}')
                        i = modes.index(m)
                        modes = [modes[i]]
                    elif verbose:
                        logger.warning(f'no observations for requested {i} mode ({m})')


            # HARPS observations should not be separated by 'mode' if some are
            # done together with NIRPS, but should be separated by 'EGGS' mode
            if 'HARPS' in inst:
                m0 = modes[0]
                data = {
                    k: np.concatenate([result[inst][pipe][m][k] for m in modes])
                    for k in result[inst][pipe][m0].keys()
                }
                if 'HARPS+NIRPS' in modes:
                    arrays.append( ((str(inst), str(pipe), str(m0)), data) )
                    continue

                if 'EGGS+NIRPS' in modes or 'EGGS' in modes:
                    arrays.append( ((str(inst + '_EGGS'), str(pipe), str(m0)), data) )
                    continue

            for mode in modes:
                if 'rjd' not in result[inst][pipe][mode]:
                    logger.error(f"No 'rjd' key for {inst} - {pipe}")
                    raise ValueError

                arrays.append(
                    ((str(inst), str(pipe), str(mode)), result[inst][pipe][mode])
                )

    return arrays

# def get_observations_from_instrument(star, instrument, user=None, main_id=None, verbose=True):
#     """ Query DACE for all observations of a given star and instrument

#     Args:
#         star (str):
#             name of the star
#         instrument (str):
#             instrument name
#         user (str, optional):
#             DACERC user name. Defaults to None.
#         main_id (str, optional):
#             Simbad main id of target to query DACE id. Defaults to None.
#         verbose (bool, optional):
#             whether to print warnings. Defaults to True.

#     Raises:
#         ValueError:
#             If query for DACE id fails

#     Returns:
#         dict:
#             dictionary with data from DACE
#     """
#     Spectroscopy = load_spectroscopy(user, verbose)
#     found_dace_id = False
#     with timer('dace_id query'):
#         try:
#             dace_id = get_dace_id(star, verbose=verbose, raise_error=True)
#             found_dace_id = True
#         except ValueError as e:
#             if main_id is not None:
#                 try:
#                     dace_id = get_dace_id(main_id, verbose=verbose, raise_error=True)
#                     found_dace_id = True
#                 except ValueError:
#                     pass

#     if not found_dace_id:
#         try:
#             with all_logging_disabled():
#                 result = Spectroscopy.get_timeseries(target=star,
#                                                      sorted_by_instrument=True,
#                                                      output_format='numpy')
#                 return result
#         except TypeError:
#             msg = f'no {instrument} observations for {star}'
#             raise ValueError(msg) from None

#     if (isinstance(instrument, str)):
#         filters = {
#             "ins_name": {"contains": [instrument]},
#             "obj_id_daceid": {"equal": [dace_id, main_id]}
#         }
#     elif (isinstance(instrument, (list, tuple, np.ndarray))):
#         filters = {
#             "ins_name": {"contains": instrument},
#             "obj_id_daceid": {"equal": [dace_id, main_id]}
#         }
#     with all_logging_disabled():
#         result = Spectroscopy.query_database(filters=filters)
    
#     if len(result) == 0:
#         raise ValueError

#     r = {}

#     for inst in np.unique(result['ins_name']):
#         mask1 = result['ins_name'] == inst
#         r[str(inst)] = {}

#         key2 = 'ins_drs_version'
#         n_key2 = len(np.unique(result[key2][mask1]))
#         if len(np.unique(result['pub_bibcode'][mask1])) >= n_key2:
#             key2 = 'pub_bibcode'

#         for pipe in np.unique(result[key2][mask1]):
#             mask2 = mask1 & (result[key2] == pipe)
#             r[str(inst)][str(pipe)] = {}

#             for ins_mode in np.unique(result['ins_mode'][mask2]):
#                 mask3 = mask2 & (result['ins_mode'] == ins_mode)
#                 _nan = np.full(mask3.sum(), np.nan)

#                 translations = {
#                     'obj_date_bjd': 'rjd',
#                     'spectro_drs_qc': 'drs_qc',
#                     'spectro_cal_berv_mx': 'bervmax',
#                     'pub_ref': 'pub_reference',
#                     'file_rootpath': 'raw_file',
#                     'spectro_ccf_asym': 'ccf_asym',
#                     'spectro_ccf_asym_err': 'ccf_asym_err',
#                 }
#                 new_result = {}
#                 for key in result.keys():
#                     if key in translations:
#                         new_key = translations[key]
#                     else:
#                         new_key = key
#                         new_key = new_key.replace('spectro_ccf_', '')
#                         new_key = new_key.replace('spectro_cal_', '')
#                         new_key = new_key.replace('spectro_analysis_', '')
#                     new_result[new_key] = result[key][mask3]

#                 new_result['ccf_noise'] = np.sqrt(
#                     np.square(result['spectro_ccf_rv_err'][mask3]) - np.square(result['spectro_cal_drift_noise'][mask3])
#                 )

#                 r[str(inst)][str(pipe)][str(ins_mode)] = new_result

#                 # r[str(inst)][str(pipe)][str(ins_mode)] = {
#                 #     'texp': result['texp'][mask3],
#                 #     'bispan': result['spectro_ccf_bispan'][mask3],
#                 #     'bispan_err': result['spectro_ccf_bispan_err'][mask3],
#                 #     'drift_noise': result['spectro_cal_drift_noise'][mask3],
#                 #     'rjd': result['obj_date_bjd'][mask3],
#                 #     'cal_therror': _nan,
#                 #     'fwhm': result['spectro_ccf_fwhm'][mask3],
#                 #     'fwhm_err': result['spectro_ccf_fwhm_err'][mask3],
#                 #     'rv': result['spectro_ccf_rv'][mask3],
#                 #     'rv_err': result['spectro_ccf_rv_err'][mask3],
#                 #     'berv': result['spectro_cal_berv'][mask3],
#                 #     'ccf_noise': np.sqrt(
#                 #         np.square(result['spectro_ccf_rv_err'][mask3]) - np.square(result['spectro_cal_drift_noise'][mask3])
#                 #     ),
#                 #     'rhk': result['spectro_analysis_rhk'][mask3],
#                 #     'rhk_err': result['spectro_analysis_rhk_err'][mask3],
#                 #     'contrast': result['spectro_ccf_contrast'][mask3],
#                 #     'contrast_err': result['spectro_ccf_contrast_err'][mask3],
#                 #     'cal_thfile': result['spectro_cal_thfile'][mask3],
#                 #     'spectroFluxSn50': result['spectro_flux_sn50'][mask3],
#                 #     'protm08': result['spectro_analysis_protm08'][mask3],
#                 #     'protm08_err': result['spectro_analysis_protm08_err'][mask3],
#                 #     'caindex': result['spectro_analysis_ca'][mask3],
#                 #     'caindex_err': result['spectro_analysis_ca_err'][mask3],
#                 #     'pub_reference': result['pub_ref'][mask3],
#                 #     'drs_qc': result['spectro_drs_qc'][mask3],
#                 #     'haindex': result['spectro_analysis_halpha'][mask3],
#                 #     'haindex_err': result['spectro_analysis_halpha_err'][mask3],
#                 #     'protn84': result['spectro_analysis_protn84'][mask3],
#                 #     'protn84_err': result['spectro_analysis_protn84_err'][mask3],
#                 #     'naindex': result['spectro_analysis_na'][mask3],
#                 #     'naindex_err': result['spectro_analysis_na_err'][mask3],
#                 #     'snca2': _nan,
#                 #     'mask': result['spectro_ccf_mask'][mask3],
#                 #     'public': result['public'][mask3],
#                 #     'spectroFluxSn20': result['spectro_flux_sn20'][mask3],
#                 #     'sindex': result['spectro_analysis_smw'][mask3],
#                 #     'sindex_err': result['spectro_analysis_smw_err'][mask3],
#                 #     'drift_used': _nan,
#                 #     'ccf_asym': result['spectro_ccf_asym'][mask3],
#                 #     'ccf_asym_err': result['spectro_ccf_asym_err'][mask3],
#                 #     'date_night': result['date_night'][mask3],
#                 #     'raw_file': result['file_rootpath'][mask3],
#                 #     'prog_id': result['prog_id'][mask3],
#                 #     'th_ar': result['th_ar'][mask3],
#                 #     'th_ar1': result['th_ar1'][mask3],
#                 #     'th_ar2': result['th_ar2'][mask3],
#                 # }
    
#     # print(r.keys())    
#     # print([r[k].keys() for k in r.keys()])
#     # print([r[k1][k2].keys() for k1 in r.keys() for k2 in r[k1].keys()])
#     return r

def _warn_harpsn(instrument):
    if 'HARPSN' in instrument or 'HARPS-N' in instrument:
        logger = setup_logger()
        logger.warning(f'Did you mean "HARPN" instead of "{instrument}"?')
        return True
    return False


def get_observations(star, instrument=None, user=None,
                     only_latest_pipeline=False, verbose=True):
    logger = setup_logger()
    Spectroscopy = load_spectroscopy(user, verbose)

    if instrument is None:
        filters = None
    else:
        if isinstance(instrument, str):
            instrument = [instrument]

        filters = {"instrument_name": {"contains": instrument}}

    with stdout_disabled(), all_logging_disabled():
        result = Spectroscopy.get_timeseries(
            target=star,
            filters=filters,
            drs_version='latest' if only_latest_pipeline else None,
        )
    
    if len(result) == 0:
        if instrument is None:
            msg = f'no observations for {star}'
        else:
            if len(instrument) == 1:
                instrument = instrument[0]
            msg = f'no {instrument} observations for {star}'
        
            _warn_harpsn(instrument)

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

    instruments = list(map(str, result.keys()))

    if instrument is not None:
        # select only the provided instrument (if it's there)
        if (isinstance(instrument, str)):
            instruments = [inst for inst in instruments if instrument in inst]
        elif (isinstance(instrument, list)):
            instruments = [inst for inst in instruments if any(i in inst for i in instrument)]
    if len(instruments) == 0:
        if instrument is None:
            msg = f'no observations for {star}'
        else:
            msg = f'no {instrument} observations for {star}'
        raise ValueError(msg)

    if verbose:
        logger.info('RVs available from')
        with logger.contextualize(indent='  '):
            _inst = ''
            max_len_inst = max([len(inst) for inst in instruments])
            for inst in sorted(instruments):
                pipelines = list(result[inst].keys())
                max_len_pipe = max([len(pipe) for pipe in pipelines])
                for pipe in pipelines:
                    last_pipe = pipe == pipelines[-1]
                    modes = list(result[inst][pipe].keys())
                    for mode in modes:
                        N = len(result[inst][pipe][mode]['rjd'])
                        # LOG
                        if inst == _inst and last_pipe:
                            logger.info(f'{" ":>{max_len_inst}s} └ {pipe:{max_len_pipe}s} - {mode} ({N} observations)')
                        elif inst == _inst:
                            logger.info(f'{" ":>{max_len_inst}s} ├ {pipe:{max_len_pipe}s} - {mode} ({N} observations)')
                        else:
                            logger.info(f'{inst:>{max_len_inst}s} ├ {pipe:{max_len_pipe}s} - {mode} ({N} observations)')
                            _inst = inst

    return result


def check_existing(output_directory, files, type):
    """ Check how many of `files` exist in `output_directory` """
    existing = [
        f.partition('.fits')[0] for f in os.listdir(output_directory)
        if type in f
    ]

    if type == 'S2D':
        existing += [
            f.partition('.fits')[0] for f in os.listdir(output_directory)
            if 'e2ds' in f
        ]   

    # also check for lowercase type
    existing += [
        f.partition('.fits')[0] for f in os.listdir(output_directory)
        if type.lower() in f
    ]

    if os.name == 'nt':  # on Windows, be careful with ':' in filename
        import re
        existing = [re.sub(r'T(\d+)_(\d+)_(\d+)', r'T\1:\2:\3', f) for f in existing]
    
    # remove type of file (e.g. _CCF_A) and 'r.' prefix
    existing = [f.partition('_')[0] for f in existing]
    existing = [f.partition('r.')[2] for f in existing]
    existing = np.unique(existing)

    missing = []
    for file in files:
        if any(other in file for other in existing):
            continue
        missing.append(file)

    return np.array(missing)

def download(files, type, output_directory, output_filename=None, user=None,
             quiet=True, pbar=None):
    """ Download files from DACE """
    Spectroscopy = load_spectroscopy(user)
    if isinstance(files, str):
        files = [files]
    if isinstance(files, np.ndarray):
        files = files.tolist()

    file_type = type.lower()
    if file_type == 's2d':
        file_type = ['S2D_A', 'S2D_BLAZE_A']

    kw = {
        "filters": {"file_rootname": {"equal": files}},
        "file_type": file_type,
        "compressed": True,
        "drs_version": "latest",
        "output_directory": output_directory,
        "output_filename": output_filename,
    }

    if quiet:
        with all_logging_disabled(), stdout_disabled():
            Spectroscopy.download(**kw)
    else:
        Spectroscopy.download(**kw)
    if pbar is not None:
        pbar.update()


def extract_fits(output_directory, filename=None):
    """ Extract fits files from tar.gz file """
    if filename is None:
        filename = 'spectroscopy_download.tar.gz'
    file = os.path.join(output_directory, filename)
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
    logger = setup_logger()
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


def do_download_filetype(type, raw_files, output_directory, clobber=False, user=None,
                         verbose=True, chunk_size=20, parallel_limit=30):
    """ Download CCFs / S1Ds / S2Ds from DACE """
    logger = setup_logger()
    raw_files = np.atleast_1d(raw_files)
    raw_files_original = raw_files.copy()

    create_directory(output_directory)

    # check existing files to avoid re-downloading
    if not clobber:
        raw_files = check_existing(output_directory, raw_files, type)
    
    n = raw_files.size

    # any file left to download?
    if n == 0:
        if verbose:
            logger.info('no files to download')
        return list(map(os.path.basename, raw_files_original))

    # avoid an empty chunk
    if chunk_size > n:
        chunk_size = n  

    if verbose:
        if chunk_size < n:
            msg = f"downloading {n} {type}s "
            msg += f"(in chunks of {chunk_size}) "
            msg += f"into '{output_directory}'..."
            logger.info(msg)
        else:
            msg = f"downloading {n} {type}s into '{output_directory}'..."
            logger.info(msg)

    if n < parallel_limit:
        iterator = [raw_files[i:i + chunk_size] for i in range(0, n, chunk_size)]
        if len(iterator) > 1:
            iterator = tqdm(iterator, total=len(iterator))
        for files in iterator:
            download(files, type, output_directory, quiet=False, user=user)
            extract_fits(output_directory)

    else:
        def chunker(it, size):
            iterator = iter(it)
            while chunk := list(islice(iterator, size)):
                yield chunk

        chunks = list(chunker(raw_files, chunk_size))
        pbar = tqdm(total=len(chunks))
        it1 = [
            (files, type, output_directory, f'spectroscopy_download{i+1}.tar.gz', user, True, pbar)
            for i, files in enumerate(chunks)
        ]
        it2 = [(output_directory, f'spectroscopy_download{i+1}.tar.gz') for i in range(len(chunks))]

        # import multiprocessing as mp
        # with mp.Pool(4) as pool:
        from multiprocessing.pool import ThreadPool

        with ThreadPool(4) as pool:
            pool.starmap(download, it1)
            pool.starmap(extract_fits, it2)
            print('')

    sys.stdout.flush()
    logger.info('extracted .fits files')
    return list(map(os.path.basename, raw_files_original))

