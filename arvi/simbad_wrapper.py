import os
import requests
from dataclasses import dataclass
from functools import partial

import numpy as np

try:
    from uncertainties import ufloat
except ImportError:
    ufloat = lambda x, y: x

from .stellar import EFFECTIVE_TEMPERATURES, teff_to_sptype
from .translations import translate

DATA_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(DATA_PATH, 'data')

QUERY = """
SELECT basic.OID,
       RA,
       DEC,
       main_id,
       pmra,
       pmdec,
       plx_value, plx_err,
       rvz_radvel,
       sp_type
FROM basic JOIN ident ON oidref = oid
WHERE id = '{star}';
"""

# SELECT filter, flux, flux_err
# FROM basic JOIN ident ON oid = ident.oidref JOIN flux ON oid = flux.oidref
# WHERE id = 'HD23079';

BV_QUERY = """
SELECT B, V FROM allfluxes
JOIN ident USING(oidref)
WHERE id = '{star}';
"""

FILTERS_QUERY = """
SELECT filter, flux, flux_err, bibcode FROM flux 
JOIN ident USING(oidref)
WHERE id = '{star}';
"""

MEAS_QUERY = """
SELECT teff, log_g, log_g_prec, fe_h, fe_h_prec, bibcode FROM mesFe_H 
JOIN ident USING(oidref)
WHERE id = '{star}';
"""

IDS_QUERY = """
SELECT ids FROM ids
JOIN ident USING(oidref)
WHERE id = '{star}';
"""

OID_QUERY = """
SELECT basic.OID FROM basic 
JOIN ident ON oidref = oid 
WHERE id = '{star}';
"""

HD_GJ_HIP_QUERY = """
SELECT id2.id
FROM ident AS id1 JOIN ident AS id2 USING(oidref)
WHERE id1.id = '{star}' AND id2.id LIKE '{name}%';
"""

RA_DEC_QUERY = """
SELECT DISTINCT basic.OID,
       RA,
       DEC,
       main_id,
       DISTANCE(POINT('ICRS', RA, DEC), POINT('ICRS', {ra}, {dec})) as "dist"
FROM basic JOIN flux ON oidref = oid
WHERE CONTAINS(POINT('ICRS', RA, DEC), CIRCLE('ICRS', {ra}, {dec}, 0.02)) = 1
ORDER BY "dist";
"""

def find_identifier(identifier, star):
    response = run_query(HD_GJ_HIP_QUERY.format(name=identifier, star=star))
    if identifier in response:
        return response.split('"')[1]
    raise ValueError(f'no {identifier} identifier found for "{star}"')

find_HD = partial(find_identifier, 'HD')
find_GJ = partial(find_identifier, 'GJ')
find_HIP = partial(find_identifier, 'HIP')


@dataclass
class Measurements:
    teff: list
    logg: list
    feh: list
    bibcode: list


def run_query(query, SIMBAD_URL='http://simbad.u-strasbg.fr'):
    url = f'{SIMBAD_URL}/simbad/sim-tap/sync'
    data = dict(query=query, request='doQuery', lang='ADQL', format='text/plain', phase='run')
    try:
        response = requests.post(url, data=data, timeout=2)
    except requests.ReadTimeout as err:
        raise IndexError(err) from None
    except requests.ConnectionError as err:
        raise IndexError(err) from None
    return response.content.decode()

def parse_table1(table, cols=None, values=None):
    header = table.splitlines()[0].split('|')
    if cols is None:
        cols = list(map(str.strip, header))
    else:
        cols = cols + list(map(str.strip, header))
    if values is None:
        values = table.splitlines()[2].split('|')
    else:
        values = values + table.splitlines()[2].split('|')
    values = list(map(str.strip, values))
    values = [value.replace('"', '') for value in values]
    return cols, values

def parse_tablen(table, cols=None, values=None):
    header = table.splitlines()[0].split('|')
    cols = list(map(str.strip, header))
    values = [list(map(str.strip, row.split('|'))) for row in table.splitlines()[2:]]
    return cols, values

def parse_value(value, err=None, prec=None):
    try:
        v = float(value)
        if err:
            try:
                v = ufloat(float(value), float(err))
            except ValueError:
                pass
        if prec:
            try:
                v = ufloat(float(value), 10**-int(prec))
            except ValueError:
                pass
    except ValueError:
        v = np.nan
    return v


class simbad:
    """
    A very simple wrapper around a TAP query to simbad for a given target. This
    class simply runs a few TAP queries and stores the result as attributes.

    Attributes:
        ra (float): right ascension
        dec (float): declination
        coords (SkyCoord): coordinates as a SkyCoord object
        main_id (str): main identifier
        gaia_id (int): Gaia DR3 identifier
        plx (float): parallax
        rvz_radvel (float): radial velocity
        sp_type (str): spectral type
        B (float): B magnitude 
        V (float): V magnitude
        ids (list): list of identifiers
    """
    def __init__(self, star:str, _debug=False):
        """
        Args:
            star (str): The name of the star to query simbad
        """
        from astropy.coordinates import SkyCoord
        import pysweetcat

        self.star = translate(star, ngc=True, ic=True)

        if 'kobe' in self.star.lower():
            fname = os.path.join(DATA_PATH, 'KOBE-translate.csv')
            kobe_translate = {}
            if os.path.exists(fname):
                with open(fname) as f:
                    for line in f.readlines():
                        kobe_id, catname = line.strip().split(',')
                        kobe_translate[kobe_id] = catname
                try:
                    self.star = star = kobe_translate[self.star]
                except KeyError:
                    raise ValueError(f'simbad query for {star} failed')

        # oid = run_query(query=OID_QUERY.format(star=star))
        # self.oid = str(oid.split()[-1])

        try:
            table1 = run_query(query=QUERY.format(star=self.star))
            if _debug:
                print('table1:', table1)
            cols, values = parse_table1(table1)
        except IndexError:
            raise ValueError(f'simbad query for {star} failed') from None

        try:
            table2 = run_query(query=BV_QUERY.format(star=self.star))
            if _debug:
                print('table2:', table2)
            cols, values = parse_table1(table2, cols, values)
        except IndexError:
            self.B = self.V = np.nan

        try:
            table3 = run_query(query=IDS_QUERY.format(star=self.star))
            if _debug:
                print('table3:', table3)
            line = table3.splitlines()[2]
            self.ids = line.replace('"', '').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').split('|')
        except IndexError:
            self.ids = []

        table4 = run_query(query=FILTERS_QUERY.format(star=self.star))
        if _debug:
            print('table4:\n', table4)
        for row in table4.splitlines()[2:]:
            filter_name, mag, mag_err, bibcode = row.replace('"', '').split('|')
            filter_name = filter_name.strip()
            try:
                setattr(self, '_' + filter_name, ufloat(float(mag), float(mag_err)))
            except ValueError:
                setattr(self, '_' + filter_name, float(mag))

        # measurements table
        table5 = run_query(query=MEAS_QUERY.format(star=self.star))
        if _debug:
            print('table5:\n', table5)

        _teff, _logg, _feh, _bibcode = [], [], [], []
        for row in table5.splitlines()[2:]:
            teff, log_g, log_g_prec, fe_h, fe_h_prec, bibcode = row.replace('"', '').split('|')
            _bibcode.append(bibcode)
            _teff.append(parse_value(teff))
            _logg.append(parse_value(log_g, prec=log_g_prec))
            _feh.append(parse_value(fe_h, prec=fe_h_prec))

        self.measurements = Measurements(_teff, _logg, _feh, _bibcode)


        try:
            self.gaia_id = int([i for i in self.ids if 'Gaia DR3' in i][0]
                               .split('Gaia DR3')[-1])
        except IndexError:
            self.gaia_id = None

        for col, val in zip(cols, values):
            if col == 'oid':
                setattr(self, col, str(val))
                continue
            try:
                setattr(self, col, float(val))
            except ValueError:
                setattr(self, col, val)

        self.coords = SkyCoord(self.ra, self.dec, unit='deg')

        if self.plx_value == '':
            self.plx_value = None
        
        self.plx = self._plx_value = self.plx_value
        del self.plx_value

        try:
            swc_data = pysweetcat.get_data()
            data = swc_data.find(star)
            if data is None:
                for id in self.ids:
                    data = swc_data.find(id)
                    if data is not None:
                        break
            if data is None:
                raise IndexError
            else:
                self.teff = data['teff']
                self.sweetcat = data

        except IndexError:
            if self.sp_type == '':
                if len(self.measurements.teff) > 0:
                    try:
                        self.teff = int(np.mean(self.measurements.teff))
                        self.sp_type = teff_to_sptype(self.teff)
                    except ValueError:
                        self.teff = np.nan
                        self.sp_type = ''
            elif self.sp_type[:2] in EFFECTIVE_TEMPERATURES:
                self.teff = EFFECTIVE_TEMPERATURES[self.sp_type[:2]]

    def __repr__(self):
        V = self.V
        sp_type = self.sp_type
        return f'{self.star} ({V=}, {sp_type=})'

    @property
    def bmv(self):
        return self.B - self.V

    @classmethod
    def from_ra_dec(cls, ra, dec, **kwargs):
        table1 = run_query(query=RA_DEC_QUERY.format(ra=ra, dec=dec))
        cols, values = parse_tablen(table1)
        if len(values) == 0:
            raise ValueError(f'no Simbad results for ra={ra}, dec={dec}')
        assert cols == ['oid', 'ra', 'dec', 'main_id', 'dist']
        star = values[0][cols.index('main_id')].replace('"', '')
        return cls(star, **kwargs)

def argsort_by_spectral_type(sptypes):
    STs = [f'{letter}{n}' for letter in ('F', 'G', 'K', 'M') for n in range(10)]
    order = {st: i for i, st in enumerate(STs)}
    indices = {i:st for i, st in zip(range(len(sptypes)), sptypes)}
    return [i[0] for i in sorted(indices.items(), key=lambda item: order[item[1]])]
