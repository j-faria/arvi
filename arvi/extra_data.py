import os
from glob import glob
import json

from numpy import full
from .setup_logger import logger
from . import timeseries

refs = {
    'HD86226': 'Teske et al. 2020 (AJ, 160, 2)'
}

def get_extra_data(star, instrument=None, path=None, verbose=True):
    if path is None:
        path = os.path.dirname(__file__)
        path = os.path.join(path, 'data', 'extra')

    metadata = json.load(open(os.path.join(path, 'metadata.json'), 'r'))
    # print(metadata)

    files = glob(os.path.join(path, star + '*'))
    files = [f for f in files if os.path.isfile(f)]
    files = [f for f in files if not os.path.basename(f).endswith('.zip')]

    if len(files) == 0:
        raise FileNotFoundError

    def get_instruments(files):
        instruments = [os.path.basename(f).split('.')[0] for f in files]
        instruments = [i.split('_', maxsplit=1)[1] for i in instruments]
        return instruments
    
    instruments = get_instruments(files)

    if instrument is not None:
        if not any([instrument in i for i in instruments]):
            raise FileNotFoundError
        files = [f for f in files if instrument in f]
        instruments = get_instruments(files)

    if verbose:
        logger.info(f'loading extra data for {star}')

    units = len(files) * ['ms']
    reference = len(files) * [None]
    did_sa = len(files) * [False]

    for i, file in enumerate(files):
        file_basename = os.path.basename(file)
        if file_basename in metadata:
            if 'instrument' in metadata[file_basename]:
                instruments[i] = metadata[file_basename]['instrument']
            if 'units' in metadata[file_basename]:
                units[i] = metadata[file_basename]['units']
            if 'reference' in metadata[file_basename]:
                reference[i] = metadata[file_basename]['reference']
            if 'corrected_for_secular_acceleration' in metadata[file_basename]:
                did_sa[i] = metadata[file_basename]['corrected_for_secular_acceleration']

    s = timeseries.RV.from_rdb(files[0], star=star, instrument=instruments[0], units=units[0])
    for file, instrument, unit in zip(files[1:], instruments[1:], units[1:]):
        s = s + timeseries.RV.from_rdb(file, star=star, instrument=instrument, units=unit)

    for i, (inst, ref, inst_did_sa) in enumerate(zip(s.instruments, reference, did_sa)):
        _s = getattr(s, inst)
        if ref is not None:
            _s.pub_reference = full(_s.N, ref)
        if inst_did_sa:
            _s._did_secular_acceleration = True

    return s
