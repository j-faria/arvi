from collections import defaultdict
import os
from urllib.error import URLError
import requests
from dataclasses import dataclass
from functools import lru_cache, partial

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
SELECT b.main_id, m.teff, m.log_g, m.log_g_prec, m.fe_h, m.fe_h_prec, m.bibcode FROM mesFe_H AS m
JOIN ident USING(oidref)
JOIN basic as b ON oidref = oid
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

OTYPE_QUERY = """
SELECT otype FROM basic 
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


@lru_cache
def run_query(query, SIMBAD_URL='http://simbad.u-strasbg.fr', timeout=5):
    url = f'{SIMBAD_URL}/simbad/sim-tap/sync'
    data = dict(query=query, request='doQuery', lang='ADQL', format='text/plain', phase='run')
    try:
        response = requests.post(url, data=data, timeout=timeout)
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

    @classmethod
    def from_values(
        cls,
        star: str,
        ra: float = None,
        dec: float = None,
        main_id: str = None,
        gaia_id: int = None,
        plx: float = None,
        rvz_radvel: float = None,
        sp_type: str = None,
        otype: str = None,
        B: float = None,
        V: float = None,
        ids: list[str] = [],
        **kwargs
    ):
        s = cls.__new__(cls)
        s.star = star
        s.ra = ra
        s.dec = dec
        s.main_id = main_id
        s.gaia_id = gaia_id
        s.plx = plx
        s.rvz_radvel = rvz_radvel
        s.sp_type = sp_type
        s.otype = otype
        s.B = B
        s.V = V
        s.ids = ids
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    def __init__(self, star:str, _debug=False, _timeout=5):
        """
        Args:
            star (str): The name of the star to query simbad
            _debug (bool): Whether to print debug information
            _timeout (int): Timeout for the TAP query
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
                        kobe_id, catname, _dr3 = line.strip().split(',')
                        kobe_translate[kobe_id] = catname
                try:
                    self.star = star = kobe_translate[self.star]
                except KeyError:
                    raise ValueError(f'simbad query for {star} failed')

        # oid = run_query(query=OID_QUERY.format(star=star))
        # self.oid = str(oid.split()[-1])

        try:
            table1 = run_query(query=QUERY.format(star=self.star), timeout=_timeout)
            if _debug:
                print('table1:', table1)
            cols, values = parse_table1(table1)
        except IndexError:
            raise ValueError(f'simbad query for {star} failed') from None

        try:
            table2 = run_query(query=BV_QUERY.format(star=self.star), timeout=_timeout)
            if _debug:
                print('table2:\n', table2)
            cols, values = parse_table1(table2, cols, values)
        except IndexError:
            self.B = self.V = np.nan

        try:
            table21 = run_query(query=OTYPE_QUERY.format(star=self.star), timeout=_timeout)
            if _debug:
                print('table21:\n', table21)
            cols, values = parse_table1(table21, cols, values)
        except IndexError:
            self.otype = None

        try:
            table3 = run_query(query=IDS_QUERY.format(star=self.star), timeout=_timeout)
            if _debug:
                print('table3:', table3)
            line = table3.splitlines()[2]
            self.ids = line.replace('"', '').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').split('|')
        except IndexError:
            self.ids = []

        table4 = run_query(query=FILTERS_QUERY.format(star=self.star), timeout=_timeout)
        if _debug:
            print('table4:\n', table4)
        for row in table4.splitlines()[2:]:
            filter_name, mag, mag_err, bibcode = row.replace('"', '').split('|')
            filter_name = filter_name.strip()
            try:
                setattr(self, '_' + filter_name, ufloat(float(mag), float(mag_err)))
            except ValueError:
                setattr(self, '_' + filter_name, float(mag))
            
            # substitute missing V magnitude
            if filter_name == 'V' and values[cols.index('V')] == '':
                values[cols.index('V')] = mag
            # substitute missing B magnitude
            if filter_name == 'B' and values[cols.index('B')] == '':
                values[cols.index('B')] = mag


        # measurements table
        table5 = run_query(query=MEAS_QUERY.format(star=self.star), timeout=_timeout)
        if _debug:
            print('table5:\n', table5)

        _teff, _logg, _feh, _bibcode = [], [], [], []
        for row in table5.splitlines()[2:]:
            _, teff, log_g, log_g_prec, fe_h, fe_h_prec, bibcode = row.replace('"', '').split('|')
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
                if val == '':
                    val = None
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
        except URLError:
            pass
        except IndexError:
            if self.sp_type in (None, ''):
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


def pluralize(query):
    return query.replace("id = '{star}'", "id in {star_tuple}")


class list_simbad:
    def __init__(self, stars, _timeout=15):
        from astropy.coordinates import SkyCoord

        if isinstance(stars, str):
            stars = [stars]
        if isinstance(stars, tuple):
            stars = list(stars)
        self.stars = stars
        self.starst = tuple(stars)

        Q = pluralize(QUERY)
        table = run_query(query=Q.format(star_tuple=self.starst), timeout=_timeout)
        cols, values = parse_tablen(table)
        for col, vals in zip(cols, list(zip(*values))):
            try:
                vals = list(map(float, vals))
            except ValueError:
                pass
            try:
                vals = [v.replace('"', '') for v in vals]
            except AttributeError:
                pass
            setattr(self, col, vals)


        Q = pluralize(BV_QUERY)
        table = run_query(query=Q.format(star_tuple=self.starst), timeout=_timeout)
        cols, values = parse_tablen(table)
        values = list(zip(*values))
        for col, vals in zip(cols, values):
            vals = ['nan' if v == '' else v for v in vals]
            setattr(self, col, list(map(float, vals)))

        Q = pluralize(OTYPE_QUERY)
        table = run_query(query=Q.format(star_tuple=self.starst), timeout=_timeout)
        cols, values = parse_tablen(table)
        self.otype = [v[0].replace('"', '') for v in values]

        Q = pluralize(IDS_QUERY)
        table = run_query(query=Q.format(star_tuple=self.starst), timeout=_timeout)
        cols, values = parse_tablen(table)
        self.ids = [
            [' '.join(vv.replace('"', '').split()) for vv in v] for v in values
        ]
        self.gaia_id = [
            int(i.split("Gaia DR3")[1]) for ids in self.ids
            for i in ids if "Gaia DR3" in i
        ]

        Q = pluralize(FILTERS_QUERY)
        table = run_query(query=Q.format(star_tuple=self.starst), timeout=_timeout)
        cols, values = parse_tablen(table)
        filters = defaultdict(list)
        for row in values:
            filter_name, mag, mag_err, bibcode = row
            filter_name = filter_name.replace('"', '').strip()
            filters[filter_name].append(float(mag))
        for key in filters:
            setattr(self, '_' + key, filters[key])

        # measurements table
        Q = pluralize(MEAS_QUERY)
        table = run_query(query=Q.format(star_tuple=self.starst), timeout=_timeout)
        _teff, _logg, _feh, _bibcode = [], [], [], []
        measures = {}
        for row in table.splitlines()[2:]:
            main_id, teff, log_g, log_g_prec, fe_h, fe_h_prec, bibcode = row.replace('"', '').split('|')
            if main_id not in measures:
                measures[main_id] = {'teff': [], 'log_g': [], 'fe_h': [], 'bibcode': []}
            measures[main_id]['teff'].append(parse_value(teff))
            measures[main_id]['log_g'].append(parse_value(log_g, prec=log_g_prec))
            measures[main_id]['fe_h'].append(parse_value(fe_h, prec=fe_h_prec))
            measures[main_id]['bibcode'].append(bibcode)
        
        self.measurements = []
        for main_id in measures:
            self.measurements.append(
                Measurements(measures[main_id]['teff'], 
                             measures[main_id]['log_g'], 
                             measures[main_id]['fe_h'], 
                             measures[main_id]['bibcode'])
        )

        self.plx = self.plx_value
        del self.plx_value

        self.coords = [
            SkyCoord(ra, dec, unit='deg') for ra, dec in zip(self.ra, self.dec)
        ]

    @property
    def individual(self):
        all_vals = zip(
            self.stars, self.ra, self.dec, self.main_id, self.gaia_id, self.plx,
            self.rvz_radvel, self.sp_type, self.otype, self.B, self.V, self.ids
        )
        S = [simbad.from_values(*args) for args in all_vals]
        for i, s in enumerate(S):
            s.plx_err = self.plx_err[i]
            s.measurements = self.measurements[i]
            s._V = self._V[i]
            s._B = self._B[i]
            s._G = self._G[i]
            s._H = self._H[i]
            s._J = self._J[i]
            s._K = self._K[i]

        return S


def argsort_by_spectral_type(sptypes):
    STs = [f'{letter}{n}' for letter in ('F', 'G', 'K', 'M') for n in range(10)]
    order = {st: i for i, st in enumerate(STs)}
    indices = {i:st for i, st in zip(range(len(sptypes)), sptypes)}
    return [i[0] for i in sorted(indices.items(), key=lambda item: order[item[1]])]
