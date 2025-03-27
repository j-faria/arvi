import numpy as np

# from Table 5 of Pecaut & Mamajek (2013, ApJS, 208, 9; http://adsabs.harvard.edu/abs/2013ApJS..208....9P)
# https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
EFFECTIVE_TEMPERATURES = {
    'F0': 7220, 'F1': 7020, 'F2': 6820, 'F3': 6750, 'F4': 6670, 'F5': 6550, 'F6': 6350, 'F7': 6280, 'F8': 6180, 'F9': 6050,
    'G0': 5930, 'G1': 5860, 'G2': 5770, 'G3': 5720, 'G4': 5680, 'G5': 5660, 'G6': 5600, 'G7': 5550, 'G8': 5480, 'G9': 5380,
    'K0': 5270, 'K1': 5170, 'K2': 5100, 'K3': 4830, 'K4': 4600, 'K5': 4440, 'K6': 4300, 'K7': 4100, 'K8': 3990, 'K9': 3930,
    'M0': 3850, 'M1': 3660, 'M2': 3560, 'M3': 3430, 'M4': 3210, 'M5': 3060, 'M6': 2810, 'M7': 2680, 'M8': 2570, 'M9': 2380,
}

def teff_to_sptype(teff):
    """
    Estimate the spectral type from the effective temperature, using the
    Pecaut & Mamajek (2013) table.

    Args:
        teff (float): Effective temperature

    Returns:
        str: Spectral type
    """
    teffs = list(EFFECTIVE_TEMPERATURES.values())
    sptypes = list(EFFECTIVE_TEMPERATURES.keys())
    i = np.argmin(np.abs(np.array(teffs) - teff))
    return sptypes[i]


class prot_age_result:
    prot_n84 = None #: float | np.ndarray
    prot_n84_err = None #: float | np.ndarray
    prot_m08 = None #: float | np.ndarray
    prot_m08_err = None #: float | np.ndarray
    age_m08 = None #: float | np.ndarray
    age_m08_err = None #: float | np.ndarray
    def __init__(self):
        pass
    def __repr__(self):
        if isinstance(self.prot_n84, np.ndarray):
            s = f'{self.prot_n84.mean()=:.2f} ± {self.prot_n84_err.mean():.2f}, '
        else:
            s = f'{self.prot_n84=:.2f} ± {self.prot_n84_err:.2f}, '
        if isinstance(self.prot_m08, np.ndarray):
            s += f'{self.prot_m08.mean()=:.2f} ± {self.prot_m08_err.mean():.2f}, '
        else:
            s += f'{self.prot_m08=:.2f} ± {self.prot_m08_err:.2f}, '
        if isinstance(self.age_m08, np.ndarray):
            s += f'{self.age_m08.mean()=:.2f} ± {self.age_m08_err.mean():.2f}'
        else:
            s += f'{self.age_m08=:.2f} ± {self.age_m08_err:.2f}'
        return s.replace('self.', '')


def calc_prot_age(self, bv=None, array=False):
    """
    Calculate rotation period and age from logR'HK activity level, based on the
    empirical relations of Noyes et al. (1984) and Mamajek & Hillenbrand (2008).

    Args:
        self (`arvi.RV`):
            RV object
        bv (float, optional):
            B-V colour. If None, use value from Simbad

    Returns:
        An object with the following attributes:

        prot_n84 (float, array):
            Chromospheric rotational period via Noyes et al. (1984)
        prot_n84_err (float, array):
            Error on 'prot_n84'
        prot_m08 (float, array):
            Chromospheric rotational period via Mamajek & Hillenbrand (2008)
        prot_m08_err (float, array):
            Error on 'prot_m08'
        age_m08 (float, array):
            Gyrochronology age via Mamajek & Hillenbrand (2008)
        age_m08_err (float, array):
            Error on 'age_m08'

    Range of logR'HK-Prot relation: -5.5 < logR'HK < -4.3
    Range of Mamajek & Hillenbrand (2008) relation for ages: 0.5 < B-V < 0.9
    """

    if array:
        log_rhk = self.rhk[~np.isnan(self.rhk)]
    else:
        log_rhk = np.nanmean(self.rhk[self.mask])

    if bv is None:
        bv = self.simbad.B - self.simbad.V

    # Calculate chromospheric Prot:
    if np.any(log_rhk < -4.3) & np.any(log_rhk > -5.5):
        if bv < 1:
            tau = 1.362 - 0.166*(1-bv) + 0.025*(1-bv)**2 - 5.323*(1-bv)**3
        else:
            tau = 1.362 - 0.14*(1-bv)

        prot_n84 = 0.324 - 0.400*(5 + log_rhk) - 0.283*(5 + log_rhk)**2 - 1.325*(5 + log_rhk)**3 + tau
        prot_n84 = 10**prot_n84
        prot_n84_err = np.log(10)*0.08*prot_n84
        if array:
            prot_n84_err = np.full_like(log_rhk, prot_n84_err)

        prot_m08 = (0.808 - 2.966*(log_rhk + 4.52))*10**tau
        prot_m08_err = 4.4*bv*1.7 - 1.7
        if array:
            prot_m08_err = np.full_like(log_rhk, prot_m08_err)
    else:
        prot_n84 = np.nan
        prot_n84_err = np.nan
        prot_m08 = np.nan
        prot_m08_err = np.nan

    # Calculate gyrochronology age:
    if np.any(prot_m08 > 0.0) & (bv > 0.50) & (bv < 0.9):
        age_m08 = 1e-3*(prot_m08/0.407/(bv - 0.495)**0.325)**(1./0.566)
        #age_m08_err = 0.05*np.log(10)*age_m08
        age_m08_err  = 0.2 * age_m08 * np.log(10) # using 0.2 dex typical error from paper
    else:
        age_m08 = np.nan
        age_m08_err = np.nan

    r = prot_age_result()
    r.prot_n84 = prot_n84
    r.prot_n84_err = prot_n84_err
    r.prot_m08 = prot_m08
    r.prot_m08_err = prot_m08_err
    r.age_m08 = age_m08
    r.age_m08_err = age_m08_err
    return r
    # return prot_n84, prot_n84_err, prot_m08, prot_m08_err, age_m08, age_m08_err