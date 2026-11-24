import os
from glob import glob
import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .setup_logger import setup_logger

from tqdm import tqdm
import astropy.units as u, astropy.constants as const
from astropy.io import fits

try:
    from SNT import normalize_spectra
    from SNT.utils.SNT_configs import construct_SNT_configs
    SNT_available = True
except ImportError:
    SNT_available = False


def get_KEYWORD(paths: list[str | Path], keyword: str) -> list[str]:
    def get(file):
        return fits.getval(file.as_posix(), keyword)
    paths = [Path(p) for p in paths]
    with ThreadPool(8) as pool:
        keywords = pool.map(get, paths)
    return keywords
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
        popt, pcov = curve_fit(gaussian, w, f, p0=[-np.ptp(f), center_wavelength.value, 0.1, f.max()],
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

def build_master(self, limit=None, rv_shift=True, normalize=False, 
                 instrument=None, highest_snr=None,
                 plot=True, show_all_spectra=True):
    
    logger = setup_logger()

    files = self.get_s1d_paths(instrument=instrument)
    # files = sorted(glob(f'{self.star}_downloads/*S1D_A.fits'))
    if self.verbose:
        logger.info(f'found {len(files)} S1D files')

    files = files[:limit]

    if highest_snr is not None:
        snr = get_KEYWORD(files, 'HIERARCH ESO QC ORDER50 SNR')
        snr = np.argsort(snr)[-highest_snr:]
        files = np.array(files)[snr].tolist()
        if self.verbose:
            logger.info(f'using the {highest_snr} highest SNR spectra')

    if len(files) == 0:
        if self.verbose:
            logger.warning('Should probably run `download_s1d` first')
        return

    if plot:
        nrows = 2 if show_all_spectra else 1
        nrows += 1 if normalize else 0
        fig, axs = plt.subplots(nrows, 1, sharex=True, constrained_layout=True,
                                # figsize=(9, 6)
                                )
        axs = np.atleast_1d(axs)
        for ax in axs:
            ax.set(xlabel=r'wavelength vaccum [$\AA$]', ylabel='flux')
        axs[0].set_title(self.star, loc='right', fontsize=10)

    data = [fits.getdata(f) for f in files]
    # index of spectra with the lowest wavelength range
    i = np.argmin([np.ptp(d["wavelength"][d["quality"] == 0]) for d in data])
    print(i, files[i])
    w0 = data[i]["wavelength"]
    qual = data[i]["quality"] == 0
    # w0 = fits.getdata(files[0])['wavelength']
    print(f'wavelength range: {w0[qual].min():.2f} - {w0[qual].max():.2f} Å')

    master_flux = []
    master_flux_err = []

    for i, file in enumerate(files):
        if rv_shift:
            if isinstance(rv_shift, float):
                rv = rv_shift
            else:
                rv = fits.getval(file, '*CCF RV')[0]

            print(f'{file}: shifting by {rv} km/s')
            flux = fits.getdata(file)['flux']
            _, new_flux = doppler_shift(w0, flux, rv)
        else:
            new_flux = fits.getdata(file)['flux']

        master_flux.append(new_flux)
        master_flux_err.append(fits.getdata(file)['error'])

        if plot and show_all_spectra:
            axs[0].plot(w0, new_flux, alpha=0.5)

    master_flux = np.array(master_flux)
    master_flux = np.nansum(master_flux, axis=0)

    master_flux_err = np.array(master_flux_err)
    master_flux_err = np.nansum(master_flux_err**2, axis=0)**0.5

    continuum = np.ones_like(w0)

    if normalize:
        H = fits.getheader(files[0])
        n = w0.size
        # config = SNT_Configs(
        #     remove_n_first=int((master_flux[:n//4] == 0).sum()),
        #     remove_n_last=int((master_flux[-n//4:] == 0).sum())
        # )
        continuum = normalize_spectra(w0, master_flux, 
                                      H, '.', 
                                    #   config, 
                                      store_to_disk=False)
        continuum = continuum[0]

    if plot:
        ax_m = axs[1] if show_all_spectra else axs[0]
        ax_m.plot(w0, master_flux, 'k', label='master')
        ax_m.legend()
        if normalize:
            ax_m.plot(w0, continuum, 'r', label='continuum')
            # axs[-1].plot(w0, (master_flux / continuum) * np.nanmedian(continuum), 'k', label='normalized master')
            axs[-1].plot(w0, (master_flux / continuum), 'k', label='normalized master')
        # axs[2].plot(w0, master_flux / np.nanmedian(master_flux),
        #             'k', label='normalized master')
        # axs[2].legend()
        if show_all_spectra:
            axs[0].legend([], [], title=f'{len(files)} S1D spectra')

    if normalize:
        master_flux = (master_flux / continuum) #* np.nanmedian(continuum)
        master_flux_err = (master_flux_err / continuum) #* np.nanmedian(continuum)

    if plot:
        return w0, master_flux, master_flux_err, continuum, (fig, axs)
    else:
        return w0, master_flux, master_flux_err, continuum, None


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
        logger.info(f'found {len(lines)} lines in linelist')
        logger.info('measuring EWs...')

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
        logger.info('determining stellar parameters (can take a few minutes)...')

    callback = lambda p, r, A: print('current parameters:', p)
    result = Korg.Fit.ews_to_stellar_parameters(lines, EW, callback=callback)
    par, stat_err, sys_err = result

    if self.verbose:
        logger.info(f'best-fit stellar parameters:')
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
