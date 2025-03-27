import os, sys
import numpy as np

from .setup_logger import logger
from .utils import ESPRESSO_ADC_issues, ESPRESSO_cryostat_issues


# HARPS started operations in October 1st, 2003
# https://www.eso.org/sci/facilities/lasilla/instruments/harps/news.html
HARPS_start = 52913

# HARPS fiber upgrade (28 May 2015)
# https://www.eso.org/sci/facilities/lasilla/instruments/harps/news/harps_upgrade_2015.html
HARPS_technical_intervention = 57170

# From Lo Curto et al. (2015), The Messenger, vol. 162, p. 9-15  
# On **19 May 2015** HARPS stopped operations and the instrument was opened.
# Installation and alignment of the fibre link lasted roughly one week. On 29
# May, the vacuum vessel was closed and evacuated for the last time. Finally, a
# formal commissioning of the new fibre took place, finishing on **3 June**,
# when the instrument was handed back to Science Operations.
HARPS_technical_intervention_range = (57161, 57176)

# ESPRESSO fiber link upgrade (1 July 2019)
ESPRESSO_technical_intervention = 58665


def divide_ESPRESSO(self):
    """ Split ESPRESSO data into separate sub ESP18 and ESP19 subsets """
    if self._check_instrument('ESPRESSO', strict=False) is None:
        return
    if 'ESPRESSO18' in self.instruments and 'ESPRESSO19' in self.instruments:
        if self.verbose:
            logger.info('ESPRESSO data seems to be split already, doing nothing')
        return

    from .timeseries import RV

    before = self.time < ESPRESSO_technical_intervention
    after = self.time >= ESPRESSO_technical_intervention
    new_instruments = []


    for inst, mask in zip(['ESPRESSO18', 'ESPRESSO19'], [before, after]):
        if not mask.any():
            continue

        _s = RV.from_arrays(self.star, self.time[mask], self.vrad[mask], self.svrad[mask],
                            inst=inst)
        for q in self._quantities:
            setattr(_s, q, getattr(self, q)[mask])
        setattr(self, inst, _s)
        _s._quantities = self._quantities
        _s.mask = self.mask[mask]
        new_instruments.append(inst)

    delattr(self, 'ESPRESSO')
    self.instruments = new_instruments
    self._build_arrays()

    if self.verbose:
        logger.info(f'divided ESPRESSO into {self.instruments}')
    

def divide_HARPS(self):
    """ Split HARPS data into separate sub HARPS03 and HARPS15 subsets """
    if self._check_instrument('HARPS', strict=False) is None:
        return
    if 'HARPS03' in self.instruments and 'HARPS15' in self.instruments:
        if self.verbose:
            logger.info('HARPS data seems to be split already, doing nothing')
        return

    from .timeseries import RV

    new_instruments = []
    before = self.time < HARPS_technical_intervention
    if before.any():
        new_instruments += ['HARPS03']

    after = self.time >= HARPS_technical_intervention
    if after.any():
        new_instruments += ['HARPS15']

    for inst, mask in zip(new_instruments, [before, after]):
        _s = RV.from_arrays(self.star, self.time[mask], self.vrad[mask], self.svrad[mask],
                            inst=inst)
        for q in self._quantities:
            setattr(_s, q, getattr(self, q)[mask])
        setattr(self, inst, _s)
        _s._quantities = self._quantities
        _s.mask = self.mask[mask]

    delattr(self, 'HARPS')
    self.instruments = new_instruments
    self._build_arrays()

    if self.verbose:
        logger.info(f'divided HARPS into {self.instruments}')
    

def check(self, instrument):
    instruments = self._check_instrument(instrument)
    if instruments is None:
        if self.verbose:
            logger.error(f"HARPS_fiber_commissioning: no data from {instrument}")
        return None
    return instruments


# HARPS commissioning
def HARPS_commissioning(self, mask=True, plot=True):
    """ Identify and optionally mask points during HARPS commissioning (HARPS).

    Args:
        mask (bool, optional):
            Whether to mask out the points.
        plot (bool, optional):
            Whether to plot the masked points.
    """
    if check(self, 'HARPS') is None:
        return

    affected = self.time < HARPS_start
    total_affected = affected.sum()

    if self.verbose:
        n = total_affected
        logger.info(f"there {'are'[:n^1]}{'is'[n^1:]} {n} frame{'s'[:n^1]} "
                     "during HARPS commissioning")

    if mask:
        self.mask[affected] = False
        self._propagate_mask_changes()

        if plot:
            self.plot(show_masked=True)

    return affected


# HARPS fiber commissioning
def HARPS_fiber_commissioning(self, mask=True, plot=True):
    """ Identify and optionally mask points affected by fiber commissioning (HARPS).

    Args:
        mask (bool, optional):
            Whether to mask out the points.
        plot (bool, optional):
            Whether to plot the masked points.
    """
    if check(self, 'HARPS') is None:
        return

    affected = (self.time >= HARPS_technical_intervention_range[0]) & (self.time <= HARPS_technical_intervention_range[1])
    total_affected = affected.sum()

    if self.verbose:
        n = total_affected
        logger.info(f"there {'are'[:n^1]}{'is'[n^1:]} {n} frame{'s'[:n^1]} "
                     "during the HARPS fiber commissioning period")

    if mask:
        self.mask[affected] = False
        self._propagate_mask_changes()

        if plot:
            self.plot(show_masked=True)

    return affected


# ESPRESSO ADC issues
def ADC_issues(self, mask=True, plot=True, check_headers=False):
    """ Identify and optionally mask points affected by ADC issues (ESPRESSO).

    Args:
        mask (bool, optional):
            Whether to mask out the points.
        plot (bool, optional):
            Whether to plot the masked points.
        check_headers (bool, optional):
            Whether to (double-)check the headers for missing/zero keywords.
    """
    instruments = self._check_instrument('ESPRESSO')
    
    if instruments is None:
        if self.verbose:
            logger.error(f"ADC_issues: no data from ESPRESSO")
        return

    affected_file_roots = ESPRESSO_ADC_issues()
    file_roots = [os.path.basename(f).replace('.fits', '') for f in self.raw_file]
    intersect = np.in1d(file_roots, affected_file_roots)

    if check_headers:
        from .headers import get_headers
        H = get_headers(self, check_lesta=False, check_exo2=False, instrument='ESPRE')
        badACD2 = np.array([h['*ADC2 RA'][0] for h in H]) == 0
        badACD2 |= np.array([h['*ADC2 SENS1'][0] for h in H]) == 0
        badACD2 |= np.array([h['*ADC2 TEMP'][0] for h in H]) == 0
        intersect = np.logical_or(intersect, badACD2)

    total_affected = intersect.sum()

    if self.verbose:
        n = total_affected
        logger.info(f"there {'are'[:n^1]}{'is'[n^1:]} {n} frame{'s'[:n^1]} "
                     "affected by ADC issues")

    if mask:
        self.mask[intersect] = False
        self._propagate_mask_changes()

        if plot:
            self.plot(show_masked=True)

    return intersect

# ESPRESSO cryostat issues
def blue_cryostat_issues(self, mask=True, plot=True):
    """ Identify and mask points affected by blue cryostat issues (ESPRESSO).

    Args:
        mask (bool, optional): Whether to mask out the points.
        plot (bool, optional): Whether to plot the masked points.
    """
    instruments = self._check_instrument('ESPRESSO')
    
    if instruments is None:
        if self.verbose:
            logger.error(f"blue_cryostat_issues: no data from ESPRESSO")
        return

    affected_file_roots = ESPRESSO_cryostat_issues()
    file_roots = [os.path.basename(f).replace('.fits', '') for f in self.raw_file]
    intersect = np.in1d(file_roots, affected_file_roots)

    total_affected = intersect.sum()

    if self.verbose:
        n = total_affected
        logger.info(f"there {'are'[:n^1]}{'is'[n^1:]} {n} frame{'s'[:n^1]} "
                     "affected by blue cryostat issues")

    if mask:
        self.mask[intersect] = False
        self._propagate_mask_changes()

        if plot:
            self.plot(show_masked=True)

    return intersect


def qc_scired_issues(self, plot=False, **kwargs):
    """ Identify and mask points with failed SCIRED QC

    Args:
        plot (bool, optional): Whether to plot the masked points.
    """
    from .headers import get_headers

    instruments = self._check_instrument('ESPRESSO')
    
    if instruments is None:
        if self.verbose:
            logger.error(f"no data from ESPRESSO")
            logger.info(f'available: {self.instruments}')
        return

    H = kwargs.get('H', None)
    if H is None:
        H = get_headers(self, check_lesta=False, check_exo2=False, instrument='ESPRE')
    if len(H) == 0:
        if self.verbose:
            logger.warning('this function requires access to headers, but none found')
            logger.warning('trying to download')
        self.download_ccf()
        H = get_headers(self, check_lesta=False, check_exo2=False, instrument='ESPRE')
    
    scired_check = np.array([h['HIERARCH ESO QC SCIRED CHECK'] for h in H])
    affected = scired_check == 0
    n = affected.sum()

    if self.verbose:
        logger.info(f"there {'are'[:n^1]}{'is'[n^1:]} {n} frame{'s'[:n^1]} "
                     "where QC SCIRED CHECK is 0")

    if n == 0:
        return

    self.mask[affected] = False
    self._propagate_mask_changes()

    if plot:
        self.plot(show_masked=True)

    return affected


def known_issues(self, mask=True, plot=False, **kwargs):
    """ Identify and optionally mask known instrumental issues.

    Args:
        mask (bool, optional): Whether to mask out the points.
        plot (bool, optional): Whether to plot the masked points.
    """
    try:
        adc = ADC_issues(self, mask, plot, **kwargs)
    except IndexError:
        logger.error('are the data binned? cannot proceed to mask these points...')

    try:
        cryostat = blue_cryostat_issues(self, mask, plot)
    except IndexError:
        logger.error('are the data binned? cannot proceed to mask these points...')

    try:
        harps_comm = HARPS_commissioning(self, mask, plot)
    except IndexError:
        logger.error('are the data binned? cannot proceed to mask these points...')

    try:
        harps_fibers = HARPS_fiber_commissioning(self, mask, plot)
    except IndexError:
        logger.error('are the data binned? cannot proceed to mask these points...')

    # if None in (adc, cryostat, harps_comm, harps_fibers):
    #     return

    try:
        # return adc | cryostat
        return np.logical_or.reduce((adc, cryostat, harps_comm, harps_fibers))
    except UnboundLocalError:
        return
