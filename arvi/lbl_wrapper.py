import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from astropy.io import fits
from lbl import lbl_wrap


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
    science_dir = os.path.join(lbl_run_dir, 'science', self.star)
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
    rparams['OBJECT_SCIENCE'] = [self.star]
    # This is the template that will be used or created
    rparams['OBJECT_TEMPLATE'] = [self.star]
    # This is the object temperature in K - used for getting a stellar model
    #   for the masks it only has to be good to a few 100 K
    rparams['OBJECT_TEFF'] = [self.teff]

    # run the telluric cleaning process
    rparams['RUN_LBL_TELLUCLEAN'] = False
    # create templates from the data in the science directory
    rparams['RUN_LBL_TEMPLATE'] = True
    # create a mask using the template created or supplied
    rparams['RUN_LBL_MASK'] = False
    # run the LBL compute step - which computes the line by line for each observation
    rparams['RUN_LBL_COMPUTE'] = True
    # run the LBL compile step - which compiles the rdb file and deals with outlier rejection
    rparams['RUN_LBL_COMPILE'] = True
    # skip observations if a file is already on disk 
    # (useful when adding a few new files) there is one for each RUN_XXX step
    rparams['SKIP_LBL_TELLUCLEAN'] = False
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True

    # turn on/off plots
    # rparams['PLOTS'] = False

    # RUN!
    lbl_wrap(rparams)


def plot_lbl(self):
    lbl_run_dir = 'LBL_run_dir'
    fits_file = os.path.join(lbl_run_dir, 'lblrdb', f'lbl_{self.star}_{self.star}.fits')
    hdu = fits.open(fits_file)
    wave, dv, sdv = hdu[1].data, hdu[2].data, hdu[3].data

    if wave.shape[0] < 10:
        fig, axs = plt.subplots(wave.shape[0], 2, constrained_layout=True, figsize=(7, 10))
        for w, v, sv, ax in zip(wave, dv, sdv, axs):
            ax[0].errorbar(10 * w, v - np.nanmean(v), sv, errorevery=100, fmt='.', ms=1, alpha=0.4)
            ax[0].sharey(axs[0, 0])
            ax[0].sharex(axs[0, 0])
            ax[0].set(xlabel='$\lambda$ [$\AA$]', ylabel='RV [m/s]')
            d = (v - np.nanmean(v)) / sv
            ax[1].hist(d, density=True, bins='doane')
            _x = np.linspace(-5, 5, 100)
            ax[1].plot(_x, norm(0, 1).pdf(_x), 'k', label='N(0,1)')
            ax[1].plot(_x, norm(*norm.fit(d[~np.isnan(d)])).pdf(_x), 'r', label='fit')
            ax[1].legend()
            ax[1].set_yticks([])
            ax[1].sharex(axs[0, 1])
            ax[1].set_xlim(-5, 5)
            ax[1].set(xlabel='RV / $\sigma$', ylabel='')
    plt.show()