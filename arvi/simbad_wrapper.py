from dataclasses import dataclass, field
import requests

from astropy.coordinates import SkyCoord

QUERY = """
SELECT basic.OID,
       RA,
       DEC,
       main_id,
       pmra,
       pmdec,
       plx_value,
       rvz_radvel,
       sp_type
FROM basic JOIN ident ON oidref = oid
WHERE id = '{star}';
"""

BV_QUERY = """
SELECT B, V from allfluxes
JOIN ident USING(oidref)
WHERE id = '{star}';
"""

IDS_QUERY = """
SELECT ids from ids
JOIN ident USING(oidref)
WHERE id = '{star}';
"""

def run_query(query):
    url = 'http://simbad.u-strasbg.fr/simbad/sim-tap/sync'
    response = requests.post(url,
                             data=dict(query=query,
                                       request='doQuery',
                                       lang='ADQL',
                                       format='text/plain',
                                       phase='run'))
    return response.content.decode()

def parse_table(table, cols=None, values=None):
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


effective_temperatures = {
    'F0': 7350, 'F2': 7050, 'F3': 6850, 'F5': 6700, 'F6': 6550, 'F7': 6400, 'F8': 6300, 
    'G0': 6050, 'G1': 5930, 'G2': 5800, 'G5': 5660, 'G8': 5440,
    'K0': 5240, 'K1': 5110, 'K2': 4960, 'K3': 4800, 'K4': 4600, 'K5': 4400, 'K7': 4000,
    'M0': 3750, 'M1': 3700, 'M2': 3600, 'M3': 3500, 'M4': 3400, 'M5': 3200, 'M6': 3100, 'M7': 2900, 'M8': 2700,
}


class simbad:
    """
    A very simple wrapper around a TAP query to simbad for a given target. This
    class simply runs a few TAP queries and stores the result as attributes.

    Attributes:
        ra (float): right ascension
        dec (float): declination
        coords (SkyCoord): coordinates as a SkyCoord object
        main_id (str): main identifier
        plx_value (float): parallax
        rvz_radvel (float): radial velocity
        sp_type (str): spectral type
        B (float): B magnitude 
        V (float): V magnitude
        ids (list): list of identifiers
    """
    def __init__(self, star:str):
        """
        Args:
            star (str): The name of the star to query simbad
        """
        self.star = star
        try:
            table1 = run_query(query=QUERY.format(star=star))
            cols, values = parse_table(table1)

            table2 = run_query(query=BV_QUERY.format(star=star))
            cols, values = parse_table(table2, cols, values)

            table3 = run_query(query=IDS_QUERY.format(star=star))
            line = table3.splitlines()[2]
            self.ids = line.replace('"', '').replace('    ', ' ').split('|')
        except IndexError:
            raise ValueError(f'simbad query for {star} failed')

        for col, val in zip(cols, values):
            try:
                setattr(self, col, float(val))
            except ValueError:
                setattr(self, col, val)

        self.coords = SkyCoord(self.ra, self.dec, unit='deg')

        if self.sp_type[:2] in effective_temperatures:
            self.teff = effective_temperatures[self.sp_type[:2]]

    def __repr__(self):
        V = self.V
        sp_type = self.sp_type
        return f'{self.star} ({V=}, {sp_type=})'