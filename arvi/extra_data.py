import os
from glob import glob
from numpy import full

from .setup_logger import logger
from . import timeseries

refs = {
    'HD86226': 'Teske et al. 2020 (AJ, 160, 2)'
}

def get_extra_data(star, path=None, verbose=True):
    if path is None:
        path = os.path.dirname(__file__)
        path = os.path.join(path, 'data', 'extra')
    files = glob(os.path.join(path, star + '*'))
    if len(files) == 0:
        raise FileNotFoundError
    
    if verbose:
        logger.info(f'Loading extra data for {star}')

    s = timeseries.RV.from_rdb(files, star=star)

    if star in refs:
        for i in s.instruments:
            _s = getattr(s, i)
            _s.pub_reference = full(_s.N, refs[star])
    
    return s
