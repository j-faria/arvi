import csv
import requests
import time
import importlib.resources as resources
import numpy as np

from .setup_logger import setup_logger

def get_toi_list(verbose=True):
    logger = setup_logger()
    toi_list = resources.files('arvi') / 'data' / 'exofop_toi_list.csv'
    now = time.time()
    download = not toi_list.exists() or toi_list.stat().st_mtime < now - 48 * 60 * 60
    if download:
        if verbose:
            logger.info('Downloading exofop TOI list (can take a while)...')
        r = requests.get('https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv')
        with open(toi_list, 'wb') as f:
            f.write(r.content)
    return toi_list

class exofop:
    def __init__(self, star: str, verbose=True, _debug=False):
        self.star = star
        self.verbose = verbose
        
        toi_list = get_toi_list(verbose=verbose)
        tsv = ('|'.join(i) for i in csv.reader(open(toi_list, encoding='utf8')))
        self.data = np.genfromtxt(tsv, delimiter='|', 
                                  names=True, encoding='utf8', dtype=None)


        try:
            if self.star.startswith('TIC'):
                self.tic = self.star
                w = self.data['TIC_ID'] == int(self.star[3:])
                self.toi = 'TOI-' + str(int(self.data['TOI'][w][0]))
            else:
                toi = self.star.replace('TOI-', '')
                toi = toi if toi.endswith('.01') else toi + '.01'
                toi_float = float(toi)
                if toi_float not in self.data['TOI']:
                    raise ValueError
                w = self.data['TOI'] == toi_float
                self.tic = 'TIC' + str(int(self.data['TIC_ID'][w][0]))
                self.toi = self.star
        except ValueError:
            raise ValueError(f'{self.star} not found in exofop TOI list')
        else:
            self.ra = str(self.data['RA'][w][0])
            self.dec = str(self.data['Dec'][w][0])

            self.epoch = float(self.data['Epoch_BJD'][w][0])
            self.period = float(self.data['Period_days'][w][0])
            if self.period == 0.0:
                self.period = np.nan
            self.duration = float(self.data['Duration_hours'][w][0])
            self.depth = float(self.data['Depth_ppm'][w][0])

        
    def __repr__(self):
        return f'{self.star} (TIC={self.tic}, epoch={self.epoch:.3f}, period={self.period:.3f})'
