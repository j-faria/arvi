from multiprocessing.pool import ThreadPool
import re
import requests
from io import StringIO

import numpy as np

from .setup_logger import setup_logger

URL_CCF = "http://atlas.obs-hp.fr/sophie/sophie.cgi?n=sophiecc&ob=date&a=t&o={target}"
URL_HEADER = "http://atlas.obs-hp.fr/sophie/sophie.cgi?n=sophiecc&c=i&z=fd&a=t&o=sophie:[ccf,{seq},{mask},0]"

def extract_keyword(keyword, text, raise_error=True):
    for line in text.splitlines():
        if keyword in line:
            value = re.findall(fr'{keyword}\s+([\'\w\d.]+)', line)[0]
            value = value.replace("'", "")
            try:
                return float(value)
            except ValueError:
                return value
    if raise_error:
        raise KeyError(f'Keyword {keyword} not found')

def query_sophie_archive(star: str, verbose=True):
    from .timeseries import RV
    logger = setup_logger()

    resp = requests.get(URL_CCF.format(target=star))
    if 'leda did not return a position for the name' in resp.text:
        raise ValueError(f'no SOPHIE observations for {star}')

    data = np.genfromtxt(StringIO(resp.text), dtype=None, usecols=(0, 4),
                         names=("seq", "mask"))
    
    if verbose:
        logger.info(f'found {len(data)} SOPHIE observations for {star}')

    urls = [URL_HEADER.format(seq=seq, mask=mask) for seq, mask in data]
    with ThreadPool(8) as pool:
        responses = pool.map(requests.get, urls)

    bjd, vrad, svrad = [], [], []
    fwhm, contrast = [], []
    ccf_mask = []
    _quantities = []
    errors = []

    for i, resp in enumerate(responses):
        if resp.text == '':
            errors.append(i)
            continue
        
        try:
            t, v = map(lambda k: extract_keyword(k, resp.text), 
                       ("OHP DRS BJD", "OHP DRS CCF RV"))
        except KeyError:
            errors.append(i)
            continue
        else:
            bjd.append(t)
            vrad.append(v)
        
        try:
            svrad.append(extract_keyword("OHP DRS CCF ERR", resp.text))
        except KeyError:
            try:
                svrad.append(1e-3 * extract_keyword("OHP DRS DVRMS", resp.text))
            except KeyError:
                bjd.pop(-1)
                vrad.pop(-1)
                errors.append(i)
                continue

        fwhm.append(extract_keyword('OHP DRS CCF FWHM', resp.text))
        _quantities.append('fwhm')

        contrast.append(extract_keyword('OHP DRS CCF CONTRAST', resp.text))
        _quantities.append('contrast')

        ccf_mask.append(extract_keyword('OHP DRS CCF MASK', resp.text))
        _quantities.append('ccf_mask')

    if len(errors) > 0:
        logger.warning(f'Could not retrieve {len(errors)} observation'
                       f'{"s" if len(errors) > 1 else ""}')

    bjd = np.array(bjd) - 2400000.5

    s = RV.from_arrays(star, bjd, vrad, svrad, 'SOPHIE',
                       fwhm=fwhm, fwhm_err=2*np.array(svrad),
                       contrast=contrast, 
                       ccf_mask=ccf_mask)
    s.units = 'km/s'

    # strings
    for q in ['date_night', 'prog_id', 'raw_file', 'pub_reference']:
        setattr(s, q, np.full(bjd.size, ''))
        _quantities.append(q)

    s._quantities = np.array(_quantities)

    setattr(s, 'SOPHIE', s)
    s._child = False
    s.verbose = False
    s._build_arrays()
    s.change_units('m/s')
    s.verbose = verbose

    return s
    
