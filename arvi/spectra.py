import os
from glob import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .setup_logger import logger

from tqdm import tqdm
import astropy.units as u, astropy.constants as const
from astropy.io import fits

def doppler_shift(wave: np.ndarray, flux: np.ndarray, velocity: float):
    """ Doppler shift a spectrum by a given velocity

    Args:
        wave (np.ndarray): wavelength array
        flux (np.ndarray): flux array
        velocity (float): velocity in km/s
    """
    c = const.c.to(u.km/u.second).value
    doppler_factor = np.sqrt((1 + velocity/c) / (1 - velocity/c))
    new_wavelength = wave * doppler_factor
    new_flux = np.interp(new_wavelength, wave, flux)
    return new_wavelength, new_flux

def fit_gaussian_to_line(wave, flux, center_wavelength, around=0.15 * u.angstrom,
                         careful_continuum=False, plot=True, ax=None):
    from scipy.optimize import curve_fit
    if center_wavelength < wave.min() or center_wavelength > wave.max():
        raise ValueError('`center_wavelength` is outside the wavelength range')
    if center_wavelength < wave[np.nonzero(flux)].min() or center_wavelength > wave[np.nonzero(flux)].max():
        raise ValueError('`center_wavelength` is outside the wavelength range where flux is not zero')

    try:
        wave <<= u.angstrom
    except u.UnitConversionError as e:
        raise ValueError(f'could not convert `wave` to Angstroms: {e}') from None

    try:
        center_wavelength <<= u.angstrom
    except u.UnitConversionError as e:
        raise ValueError(f'could not convert `center_wavelength` to Angstroms: {e}') from None

    try:
        around <<= u.angstrom
    except u.UnitConversionError as e:
        raise ValueError(f'could not convert `around` to Angstroms: {e}') from None


    def gaussian(x, amp, cen, wid, off):
        return amp * np.exp(-(x-cen)**2 / (2*wid**2)) + off

    wave_around = (wave > center_wavelength - around) & (wave < center_wavelength + around)
    w, f = wave[wave_around].value, flux[wave_around]

    if careful_continuum:
        wave_around_continuum = (wave > center_wavelength - 10*around) & (wave < center_wavelength + 10*around)
        wc, fc = wave[wave_around_continuum].value, flux[wave_around_continuum]
        lim = np.percentile(fc, 80)
        wc = wc[fc > lim]
        fc = fc[fc > lim]
        w, f = np.r_[wc, w], np.r_[fc, f]
        f = f[np.argsort(w)]
        w = np.sort(w)

    lower, upper = np.array([
        [-np.inf, 0], 
        [-np.inf, np.inf], 
        [0.0, 0.11], 
        [0.9*f.max(), 1.1*f.max()]
    ]).T

    try:
        popt, pcov = curve_fit(gaussian, w, f, p0=[-f.ptp(), center_wavelength.value, 0.1, f.max()],
                               bounds=(lower, upper))
    except RuntimeError as e:
        logger.warning(f'fit_gaussian_to_line: {e}')
        return None, np.nan, np.nan

    EW = A = (np.sqrt(2) * np.abs(popt[0]) * np.abs(popt[2]) * np.sqrt(np.pi)) / popt[3]
    perr = np.sqrt(np.diag(pcov))
    EW_err = (np.sqrt(2) * np.abs(perr[0]) * np.abs(perr[2]) * np.sqrt(np.pi)) / perr[3]

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        if careful_continuum:
            ax.plot(w, f, 'ko', ms=4, zorder=1)
            wave_around_plot = wave_around_continuum
        else:
            ax.plot(wave[wave_around], flux[wave_around], 'ko', ms=4, zorder=1)
            ax.plot(wave[wave_around], flux[wave_around] - gaussian(w, *popt), 'o', ms=2)
            wave_around_plot = (wave > center_wavelength - 2*around) & (wave < center_wavelength + 2*around)
        # ax.plot(wave[wave_around_plot], flux[wave_around_plot], 'o', ms=2)
        w = wave[wave_around_plot].value
        ax.plot(w, gaussian(w, *popt), 'r-')
        ax.fill_between([popt[1]-A, popt[1]+A], popt[3]+popt[0], popt[3],
                         color='C2', alpha=0.1, lw=0)

    return popt, EW*1e3, EW_err*1e3

def detrend(w, f):
    if w.shape[0] > w.shape[1]:
        w = np.copy(w).T
        f = np.copy(f).T

    f_detrended = np.zeros_like(f)
    for i, (ww, ff) in enumerate(zip(w, f)):
        m = np.nonzero(ff)
        fit = np.polyval(np.polyfit(ww[m] - np.median(ww[m]), ff[m], 1), ww - np.median(ww[m]))
        f_detrended[i] = ff - fit + np.median(ff[m])
    return w, f_detrended

def build_master(self, limit=None, plot=True):
    files = sorted(glob(f'{self.star}_downloads/*S1D_A.fits'))
    if self.verbose:
        logger.info(f'Found {len(files)} S1D files')

    files = files[:limit]

    if len(files) == 0:
        if self.verbose:
            logger.warning('Should probably run `download_s1d` first')
        return

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
        for ax in axs:
            ax.set(xlabel=r'wavelength air [$\AA$]', ylabel='flux')
        axs[0].set_title(self.star, loc='right', fontsize=10)

    w0 = fits.getdata(files[0])['wavelength_air']
    master_flux = np.zeros_like(w0)
    for file in files:
        rv = fits.getval(file, '*CCF RV')[0]
        flux = fits.getdata(file)['flux']
        _, new_flux = doppler_shift(w0, flux, rv)
        master_flux += new_flux
        if plot:
            axs[0].plot(w0, new_flux, alpha=0.5)

    master_flux /= len(files)
    if plot:
        axs[1].plot(w0, master_flux, 'k', label='master')
        axs[1].legend()
        axs[0].legend([], [], title=f'{len(files)} S1D spectra')

    return w0, master_flux


def determine_stellar_parameters(self, linelist: str, plot=True, **kwargs):
    try:
        from juliacall import Main as jl
        jl.seval("using Korg")
        Korg = jl.Korg
    except ModuleNotFoundError:
        msg = 'this function requires juliacall and Korg.jl, please `pip install juliacall`'
        logger.error(msg)
        return

    w, f = build_master(self, plot=plot)

    linelist = np.genfromtxt(linelist, dtype=None, encoding=None, names=True)
    lines = [
        Korg.Line(line['wl'], line['loggf'], Korg.Species(line['elem'].replace('Fe', 'Fe ')), line['EP'])
        for line in linelist
    ]

    if self.verbose:
        logger.info(f'Found {len(lines)} lines in linelist')
        logger.info('Measuring EWs...')

    EW = []
    pbar = tqdm(linelist)
    for line in pbar:
        pbar.set_description(f'{line["elem"]} {line["wl"]}')
        _, ew, _ = fit_gaussian_to_line(w, f, line['wl'], plot=plot,
                                        careful_continuum=kwargs.pop('careful_continuum', False))
        EW.append(ew)

    lines = list(np.array(lines)[~np.isnan(EW)])
    EW = np.array(EW)[~np.isnan(EW)]

    if self.verbose:
        logger.info('Determining stellar parameters (can take a few minutes)...')

    callback = lambda p, r, A: print('current parameters:', p)
    result = Korg.Fit.ews_to_stellar_parameters(lines, EW, callback=callback)
    par, stat_err, sys_err = result

    if self.verbose:
        logger.info(f'Best fit stellar parameters:')
        logger.info(f'  Teff: {par[0]:.0f} ± {sys_err[0]:.0f} K')
        logger.info(f'  logg: {par[1]:.2f} ± {sys_err[1]:.2f} dex')
        logger.info(f'  m/H :  {par[3]:.2f} ± {sys_err[3]:.2f} dex')

    r = {
        'teff': (par[0], sys_err[0]),
        'logg': (par[1], sys_err[1]),
        'vmic': (par[2], sys_err[2]),
        'moh': (par[3], sys_err[3]),
    }

    with open(f'{self.star}_stellar_parameters.pkl', 'wb') as f:
        pickle.dump(r, f)
    
    return r
