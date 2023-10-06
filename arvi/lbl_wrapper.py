import os
import numpy as np
import matplotlib.pyplot as plt

from .setup_logger import logger
from .timeseries import RV
from .stats import wmean

from scipy.stats import norm
from scipy.stats import sigmaclip
from astropy.io import fits
from lbl import lbl_wrap
from tqdm import tqdm


def run_lbl(self, instrument, files):
    rparams = dict()

    #   Currently supported instruments are SPIROU, HARPS, ESPRESSO, CARMENES
    #                                       NIRPS_HE, NIRPS_HA, MAROONX
    if 'HARPS' in instrument:
        rparams['INSTRUMENT'] = 'HARPS'
        rparams['DATA_SOURCE'] = 'ESO'
    elif 'ESPRESSO' in instrument:
        rparams['INSTRUMENT'] = 'ESPRESSO'
        rparams['DATA_SOURCE'] = 'ESO'
    elif 'NIRPS' in instrument:
        mode = getattr(self, 'NIRPS').modes[0]
        if mode == 'HE':
            rparams['INSTRUMENT'] = 'NIRPS_HE'
            rparams['DATA_SOURCE'] = 'ESO'
        if mode == 'HA':
            rparams['INSTRUMENT'] = 'NIRPS_HA'
            rparams['DATA_SOURCE'] = 'ESO'

    #       SPIROU: APERO or CADC
    #       NIRPS_HA: APERO or ESO
    #       NIRPS_HE: APERO or ESO
    #       CARMENES: None
    #       ESPRESSO: None
    #       HARPS: None
    #       MAROONX: RED or BLUE

    lbl_run_dir = 'LBL_run_dir'
    science_dir = os.path.join(
        lbl_run_dir, 'science', f'{self.star}_{instrument}')
    os.makedirs(science_dir, exist_ok=True)

    for file in files:
        link_from = os.path.abspath(file)
        link_to = os.path.join(science_dir, os.path.basename(file))
        try:
            os.symlink(link_from, link_to)
        except FileExistsError:
            pass

        if 'NIRPS' in instrument:
            calib_dir = os.path.join(lbl_run_dir, 'calib')
            os.makedirs(calib_dir, exist_ok=True)
            # put blaze files into /calib
            blaze_file = link_from.replace('_S2D_A', '_S2D_BLAZE_A')
            H = fits.getheader(link_from)
            for i in range(1, 50):
                if H[f'HIERARCH ESO PRO REC1 CAL{i} CATG'] == 'BLAZE_A':
                    calib_name = H[f'HIERARCH ESO PRO REC1 CAL{i} NAME']
                    break
            calib_name = os.path.join(calib_dir, calib_name)
            try:
                os.symlink(blaze_file, calib_name)
            except FileExistsError:
                pass

    rparams['DATA_DIR'] = lbl_run_dir
    # rparams['INPUT_FILE'] = '*S2D_A.fits'

    # The data type (either SCIENCE or FP or LFC)
    rparams['DATA_TYPES'] = ['SCIENCE']
    # The object name (this is the directory name under the /science/
    #    sub-directory and thus does not have to be the name in the header
    rparams['OBJECT_SCIENCE'] = [f'{self.star}_{instrument}']
    # This is the template that will be used or created
    rparams['OBJECT_TEMPLATE'] = [f'{self.star}_{instrument}']
    # This is the object temperature in K - used for getting a stellar model
    #   for the masks it only has to be good to a few 100 K
    rparams['OBJECT_TEFF'] = [self.simbad.teff]

    # run the telluric cleaning process
    rparams['RUN_LBL_TELLUCLEAN'] = False
    # create templates from the data in the science directory
    rparams['RUN_LBL_TEMPLATE'] = True
    # create a mask using the template created or supplied
    rparams['RUN_LBL_MASK'] = False
    # run the LBL compute step - which computes the line by line for each observation
    rparams['RUN_LBL_COMPUTE'] = False
    # run the LBL compile step - which compiles the rdb file and deals with outlier rejection
    rparams['RUN_LBL_COMPILE'] = False
    # skip observations if a file is already on disk
    # (useful when adding a few new files) there is one for each RUN_XXX step
    rparams['SKIP_LBL_TELLUCLEAN'] = False
    rparams['SKIP_LBL_TEMPLATE'] = False
    rparams['SKIP_LBL_MASK'] = False
    rparams['SKIP_LBL_COMPUTE'] = False
    rparams['SKIP_LBL_COMPILE'] = False

    # turn on/off plots
    # rparams['PLOTS'] = False

    # RUN!
    lbl_wrap(rparams)


def load_lbl(self, instrument=None):
    lbl_run_dir = 'LBL_run_dir'
    fits_file = os.path.join(lbl_run_dir, 'lblrdb',
                             f'lbl_{self.star}_{self.star}.fits')
    if not os.path.exists(fits_file):
        if instrument is None:
            logger.error(
                f'File "{fits_file}" does not exist, and instrument not provided')
            return
        else:
            fits_file = os.path.join(lbl_run_dir, 'lblrdb',
                                     f'lbl_{self.star}_{instrument}_{self.star}_{instrument}.fits')

    hdu = fits.open(fits_file)
    RDB = hdu[9].data

    s = RV.from_arrays(self.star,
                       RDB['rjd'], RDB['vrad'], RDB['svrad'],
                       instrument, mask=getattr(self, instrument).mask)

    s.fwhm = RDB['fwhm']
    s.fwhm_err = RDB['sig_fwhm']

    s.secular_acceleration()

    if self._did_adjust_means:
        s.vrad -= wmean(s.vrad, s.svrad)
        s.fwhm -= wmean(s.fwhm, s.fwhm_err)

    # store other columns
    columns = (
        'dW', 'sdW',
        'contrast', 'sig_contrast>contrast_err',
        'vrad_achromatic', 'svrad_achromatic',
        'vrad_chromatic_slope', 'svrad_chromatic_slope',
        'vrad_g', 'svrad_g',
        'vrad_r', 'svrad_r',
        'vrad_457nm', 'svrad_457nm',
        'vrad_473nm', 'svrad_473nm',
        'vrad_490nm', 'svrad_490nm',
        'vrad_507nm', 'svrad_507nm',
        'vrad_524nm', 'svrad_524nm',
        'vrad_542nm', 'svrad_542nm',
        'vrad_561nm', 'svrad_561nm',
        'vrad_581nm', 'svrad_581nm',
        'vrad_601nm', 'svrad_601nm',
        'vrad_621nm', 'svrad_621nm',
        'vrad_643nm', 'svrad_643nm',
        'vrad_665nm', 'svrad_665nm',
        'vrad_688nm', 'svrad_688nm',
        'vrad_712nm', 'svrad_712nm',
        'vrad_737nm', 'svrad_737nm'
    )
    for col in columns:
        if '>' in col:  # store with a different name
            setattr(s, col.split('>')[1], RDB[col.split('>')[0]])
        else:
            setattr(s, col, RDB[col])

    setattr(self, f'{instrument}_LBL', s)

