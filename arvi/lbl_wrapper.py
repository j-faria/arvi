import os
import re
import io
from glob import glob
import requests
from requests.auth import HTTPBasicAuth
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


def NIRPS_create_telluric_corrected_S2D(files):
    new_files = []
    for file in files:
        telluric_file = file.replace('_S2D_A', '_S2D_TELL_A')
        telluric_corrected_file = file.replace('_S2D_A', '_S2D_TELL_CORR_A')
        
        if os.path.exists(telluric_file):
            telluric_model = fits.open(telluric_file)[6].data
        else:
            telluric_file = telluric_file.replace('_S2D_TELL_A', '_TELL_SPECTRUM_A')
            telluric_model = fits.open(telluric_file)[1].data

        HDU = fits.open(file)
        with np.errstate(over='ignore'):
            HDU[1].data /= telluric_model
        HDU[1].data[telluric_model < 0.1] = 0.0
        HDU.writeto(telluric_corrected_file, overwrite=True)
        new_files.append(telluric_corrected_file)

    return new_files


def run_lbl(self, instrument, files, id=None,
            RUN_LBL_TEMPLATE=False, RUN_LBL_MASK=False, RUN_LBL_COMPUTE=False, RUN_LBL_COMPILE=False,
            SKIP_LBL_TEMPLATE=False, SKIP_LBL_MASK=False, SKIP_LBL_COMPUTE=False, SKIP_LBL_COMPILE=False):

    rparams = dict()

    #   Currently supported instruments are SPIROU, HARPS, ESPRESSO, CARMENES
    #                                       NIRPS_HE, NIRPS_HA, MAROONX
    if 'HARPS' in instrument:
        rparams['INSTRUMENT'] = 'HARPS'
        rparams['DATA_SOURCE'] = 'ESO'
    elif 'ESPRESSO' in instrument:
        rparams['INSTRUMENT'] = 'ESPRESSO'
        rparams['DATA_SOURCE'] = 'ESO'
        rparams['COMPIL_WAVE_MIN'] = 370
        rparams['COMPIL_WAVE_MAX'] = 800
    elif 'NIRPS' in instrument:
        mode = getattr(self, 'NIRPS').modes[0]
        if mode == 'HE':
            rparams['INSTRUMENT'] = 'NIRPS_HE'
            rparams['DATA_SOURCE'] = 'ESO'
        if mode == 'HA':
            rparams['INSTRUMENT'] = 'NIRPS_HA'
            rparams['DATA_SOURCE'] = 'ESO'
    elif 'CARMENES' in instrument:
        rparams['INSTRUMENT'] = 'CARMENES'
        rparams['DATA_SOURCE'] = None

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
    
    if id is not None:
        science_dir = f'{science_dir}_{id}'

    os.makedirs(science_dir, exist_ok=True)
    
    # science dir should have only symlinks, and they can change every time
    # so we delete them before proceeding
    previous_symlinks = glob(os.path.join(science_dir, '*'))
    for f in previous_symlinks:
        os.remove(f)

    # create symlinks for files in science dir
    for file in files:
        link_from = os.path.abspath(file)
        link_to = os.path.join(science_dir, os.path.basename(file))
        try:
            os.symlink(link_from, link_to)
        except FileExistsError:
            pass

        # if 'NIRPS' in instrument:
        #     calib_dir = os.path.join(lbl_run_dir, 'calib')
        #     os.makedirs(calib_dir, exist_ok=True)
        #     # put blaze files into /calib
        #     blaze_file = link_from.replace('_S2D_A', '_S2D_BLAZE_A')
        #     H = fits.getheader(link_from)
        #     for i in range(1, 50):
        #         if H[f'HIERARCH ESO PRO REC1 CAL{i} CATG'] == 'BLAZE_A':
        #             calib_name = H[f'HIERARCH ESO PRO REC1 CAL{i} NAME']
        #             break
        #     calib_name = os.path.join(calib_dir, calib_name)
        #     try:
        #         os.symlink(blaze_file, calib_name)
        #     except FileExistsError:
        #         pass

    rparams['DATA_DIR'] = lbl_run_dir
    # rparams['INPUT_FILE'] = '*S2D_A.fits'

    # The data type (either SCIENCE or FP or LFC)
    rparams['DATA_TYPES'] = ['SCIENCE']
    # The object name (this is the directory name under the /science/
    #    sub-directory and thus does not have to be the name in the header
    rparams['OBJECT_SCIENCE'] = [f'{self.star}_{instrument}']
    # This is the template that will be used or created
    rparams['OBJECT_TEMPLATE'] = [f'{self.star}_{instrument}']

    if id is not None:
        rparams['OBJECT_SCIENCE'][0] = rparams['OBJECT_SCIENCE'][0] + f'_{id}'
        rparams['OBJECT_TEMPLATE'][0] = rparams['OBJECT_TEMPLATE'][0] + f'_{id}'

    # This is the object temperature in K - used for getting a stellar model
    #   for the masks it only has to be good to a few 100 K
    try:
        rparams['OBJECT_TEFF'] = [self.simbad.teff]
    except AttributeError:
        rparams['OBJECT_TEFF'] = [int(input('teff: '))]

    # run the telluric cleaning process
    rparams['RUN_LBL_TELLUCLEAN'] = False
    if rparams['RUN_LBL_TELLUCLEAN']:
        rparams['DO_TELLUCLEAN'] = True
        rparams['TELLUCLEAN_DV0'] = 0

    # create templates from the data in the science directory
    rparams['RUN_LBL_TEMPLATE'] = RUN_LBL_TEMPLATE
    # create a mask using the template created or supplied
    rparams['RUN_LBL_MASK'] = RUN_LBL_MASK
    # run the LBL compute step - which computes the line by line for each observation
    rparams['RUN_LBL_COMPUTE'] = RUN_LBL_COMPUTE
    # run the LBL compile step - which compiles the rdb file and deals with outlier rejection
    rparams['RUN_LBL_COMPILE'] = RUN_LBL_COMPILE
    # skip observations if a file is already on disk
    # (useful when adding a few new files) there is one for each RUN_XXX step
    rparams['SKIP_LBL_TELLUCLEAN'] = False
    rparams['SKIP_LBL_TEMPLATE'] = SKIP_LBL_TEMPLATE
    rparams['SKIP_LBL_MASK'] = SKIP_LBL_MASK
    rparams['SKIP_LBL_COMPUTE'] = SKIP_LBL_COMPUTE
    rparams['SKIP_LBL_COMPILE'] = SKIP_LBL_COMPILE

    # turn on/off plots
    # rparams['PLOTS'] = False

    # RUN!
    lbl_wrap(rparams)


def get_lbl_apero(self):
    main_url = 'http://apero.exoplanets.ca/ari/nirps/nirps_he_online/objects/'

    translate = {
        'LP804-27': 'LP_804M27', 'HIP79431': 'LP_804M27',
    }

    if self.star in translate:
        star = translate[self.star]
    else:
        star = self.star

    # special case
    if star == 'Proxima':
        star = star.upper()


    got_rdb = False
    url = main_url + f'{star}.html'
    password = input('password: ')
    resp = requests.get(url, auth=HTTPBasicAuth('nirps', password))
    rdb = re.findall('href="([\w.\/]+lbl_\w+_\w+.rdb)', resp.text)
    if len(rdb) != 0:
        got_rdb = True

    if not got_rdb:
        url = main_url + f'{star.replace("GJ", "GL")}.html'
        resp = requests.get(url, auth=HTTPBasicAuth('nirps', password))
        rdb = re.findall('href="([\w.\/]+lbl_\w+_\w+.rdb)', resp.text)
        if len(rdb) != 0:
            got_rdb = True

    if not got_rdb:
        logger.error(f'Cannot find APERO rdb file for {star}')
        raise ValueError

    # (arbitrarily) choose first file
    rdb = rdb[0]
    print(main_url + rdb)
    resp = requests.get(main_url + rdb, auth=HTTPBasicAuth('nirps', password))
    with open(os.path.basename(rdb), 'w') as f:
        f.write(resp.text)

    with io.StringIO(resp.text) as f:
        RDB = np.genfromtxt(f, delimiter='\t', names=True, invalid_raise=False,
                            comments='N\tN', dtype=None, encoding=None)

    s = RV.from_arrays(self.star, RDB['rjd'], RDB['vrad'], RDB['svrad'], 
                       'NIRPS_LBL')#, mask=self.NIRPS.mask)

    s._quantities = []

    s.fwhm = RDB['fwhm']
    s.fwhm_err = RDB['sig_fwhm']
    s._quantities.append('fwhm')
    s._quantities.append('fwhm_err')

    # s.secular_acceleration()

    if self._did_adjust_means:
        s.vrad -= wmean(s.vrad, s.svrad)
        s.fwhm -= wmean(s.fwhm, s.fwhm_err)

    # store other columns
    columns = (
        'dW', 'sdW',
        'contrast', 'sig_contrast>contrast_err',
        'vrad_achromatic', 'svrad_achromatic',
        'vrad_chromatic_slope', 'svrad_chromatic_slope',
        # 'vrad_g', 'svrad_g',
        # 'vrad_r', 'svrad_r',
        # 'vrad_457nm', 'svrad_457nm',
        # 'vrad_473nm', 'svrad_473nm',
        # 'vrad_490nm', 'svrad_490nm',
        # 'vrad_507nm', 'svrad_507nm',
        # 'vrad_524nm', 'svrad_524nm',
        # 'vrad_542nm', 'svrad_542nm',
        # 'vrad_561nm', 'svrad_561nm',
        # 'vrad_581nm', 'svrad_581nm',
        # 'vrad_601nm', 'svrad_601nm',
        # 'vrad_621nm', 'svrad_621nm',
        # 'vrad_643nm', 'svrad_643nm',
        # 'vrad_665nm', 'svrad_665nm',
        # 'vrad_688nm', 'svrad_688nm',
        # 'vrad_712nm', 'svrad_712nm',
        # 'vrad_737nm', 'svrad_737nm'
    )
    for col in columns:
        if '>' in col:  # store with a different name
            setattr(s, col.split('>')[1], RDB[col.split('>')[0]])
            s._quantities.append(col.split('>')[1])
        else:
            setattr(s, col, RDB[col])
            s._quantities.append(col)

    setattr(self, 'NIRPS_LBL', s)
    self.instruments.append('NIRPS_LBL')


def load_lbl(self, instrument=None, filename=None, tell=False, id=None):
    lbl_run_dir = 'LBL_run_dir'

    if id is None:
        slug = f'{self.star}_{instrument}_{self.star}_{instrument}'
    else:
        slug = f'{self.star}_{instrument}_{id}_{self.star}_{instrument}_{id}'

    f = f'lbl_{slug}'

    if tell:
        f = f + f'_TELL'

    f = f + '.fits'

    fits_file = os.path.join(lbl_run_dir, 'lblrdb', f)

    # print(fits_file, os.path.exists(fits_file))

    if not os.path.exists(fits_file):
        raise FileNotFoundError(fits_file)
        # if instrument is None:
        #     logger.error(
        #         f'File "{fits_file}" does not exist, and instrument not provided')
        #     return
        # else:
        #     fits_file = os.path.join(lbl_run_dir, 'lblrdb',
        #                              f'lbl_{self.star}_{instrument}_{self.star}_{instrument}.fits')

    hdu = fits.open(fits_file)
    RDB = hdu[9].data
    s = RV.from_arrays(self.star,
                       RDB['rjd'], RDB['vrad'], RDB['svrad'],
                       instrument)#, mask=getattr(self, instrument).mask)

    s.fwhm = RDB['fwhm']
    s.fwhm_err = RDB['sig_fwhm']

    # s.secular_acceleration()

    if self._did_adjust_means:
        s.vrad -= wmean(s.vrad, s.svrad)
        s.fwhm -= wmean(s.fwhm, s.fwhm_err)

    # store other columns
    columns = (
        'dW', 'sdW',
        'contrast', 'sig_contrast>contrast_err',
        'vrad_achromatic', 'svrad_achromatic',
        'vrad_chromatic_slope', 'svrad_chromatic_slope',
        'vrad_h', 'svrad_h',
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
        'vrad_737nm', 'svrad_737nm',
        # 
        'HIERARCH ESO QC BERV>berv'
    )
    for col in columns:
        try:
            if '>' in col:  # store with a different name
                setattr(s, col.split('>')[1], RDB[col.split('>')[0]])
            else:
                setattr(s, col, RDB[col])
        except KeyError:
            pass

    setattr(self, f'{instrument}_LBL', s)

