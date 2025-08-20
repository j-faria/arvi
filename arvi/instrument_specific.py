import os, sys
import numpy as np

from .setup_logger import setup_logger
from .utils import ESPRESSO_ADC_issues, ESPRESSO_cryostat_issues


# HARPS started operations on October 1st, 2003
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


# ESPRESSO started operations on October 1st, 2018
# see Pepe et al. (2021, A&A 645, A96)
ESPRESSO_start = 58392

# ESPRESSO fiber link upgrade (1 July 2019)
ESPRESSO_technical_intervention = 58665


def divide_ESPRESSO(self):
    """ Split ESPRESSO data into separate sub ESP18 and ESP19 subsets """
    logger = setup_logger()
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
    logger = setup_logger()
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
    return self._check_instrument(instrument)

# HARPS commissioning
def HARPS_commissioning(self, mask=True, plot=True):
    """ Identify and optionally mask points during HARPS commissioning.

    Args:
        mask (bool, optional):
            Whether to mask out the points.
        plot (bool, optional):
            Whether to plot the masked points.
    """
    logger = setup_logger()
    if check(self, 'HARPS') is None:
        return

    affected = np.logical_and(
        self.instrument_array == 'HARPS03', 
        self.time < HARPS_start
    )
    total_affected = affected.sum()

    if self.verbose:
        n, i = total_affected, int(total_affected != 1)
        logger.info(f"there {['is', 'are'][i]} {n} frame{['', 's'][i]} "
                    "during HARPS commissioning")

    if mask:
        self.mask[affected] = False
        self._propagate_mask_changes(_remove_instrument=False)

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
    logger = setup_logger()
    if check(self, 'HARPS') is None:
        return

    affected = np.logical_and(
        self.time >= HARPS_technical_intervention_range[0],
        self.time <= HARPS_technical_intervention_range[1]
    )
    affected = np.logical_and(
        affected,
        np.char.find(self.instrument_array, 'HARPS') == 0
    )
    total_affected = affected.sum()

    if self.verbose:
        n, i = total_affected, int(total_affected != 1)
        logger.info(f"there {['is', 'are'][i]} {n} frame{['', 's'][i]} "
                    "during the HARPS fiber commissioning period")

    if mask:
        self.mask[affected] = False
        self._propagate_mask_changes(_remove_instrument=False)

        if plot:
            self.plot(show_masked=True)

    return affected


# ESPRESSO commissioning
def ESPRESSO_commissioning(self, mask=True, plot=True):
    """ Identify and optionally mask points during ESPRESSO commissioning.

    Args:
        mask (bool, optional):
            Whether to mask out the points.
        plot (bool, optional):
            Whether to plot the masked points.
    """
    logger = setup_logger()
    if check(self, 'ESPRESSO') is None:
        return

    affected = np.logical_and(
        self.instrument_array == 'ESPRESSO18',
        self.time < ESPRESSO_start
    )
    total_affected = affected.sum()

    if self.verbose:
        n, i = total_affected, int(total_affected != 1)
        logger.info(f"there {['is', 'are'][i]} {n} frame{['', 's'][i]} "
                    "during ESPRESSO commissioning")

    if mask:
        self.mask[affected] = False
        self._propagate_mask_changes(_remove_instrument=False)

        if plot and total_affected > 0:
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
    logger = setup_logger()
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
        n, i = total_affected, int(total_affected != 1)
        logger.info(f"there {['is', 'are'][i]} {n} frame{['', 's'][i]} "
                    "affected by ADC issues")

    if mask:
        self.mask[intersect] = False
        self._propagate_mask_changes(_remove_instrument=False)

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
    logger = setup_logger()
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
        n, i = total_affected, int(total_affected != 1)
        logger.info(f"there {['is', 'are'][i]} {n} frame{['', 's'][i]} "
                    "affected by blue cryostat issues")

    if mask:
        self.mask[intersect] = False
        self._propagate_mask_changes(_remove_instrument=False)

        if plot:
            self.plot(show_masked=True)

    return intersect


def qc_scired_issues(self, plot=False, **kwargs):
    """ Identify and mask points with failed SCIRED QC

    Args:
        plot (bool, optional): Whether to plot the masked points.
    """
    logger = setup_logger()
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
        i = int(n != 1)
        logger.info(f"there {['is', 'are'][i]} {n} frame{['', 's'][i]} "
                    "where QC SCIRED CHECK is 0")

    if n == 0:
        return

    self.mask[affected] = False
    self._propagate_mask_changes(_remove_instrument=False)

    if plot:
        self.plot(show_masked=True)

    return affected


class ISSUES:
    def known_issues(self, mask=True, plot=False, **kwargs):
        """ Identify and optionally mask known instrumental issues.

        Args:
            mask (bool, optional): Whether to mask out the points.
            plot (bool, optional): Whether to plot the masked points.
        """
        logger = setup_logger()

        functions = (
            ESPRESSO_commissioning,
            ADC_issues,
            blue_cryostat_issues,
            HARPS_commissioning,
            HARPS_fiber_commissioning
        )
        results = []

        for fun in functions:
            try:
                results.append(fun(self, mask, plot, **kwargs))
            except IndexError:
                logger.error('are the data binned? cannot proceed to mask these points...')
        
        results = list(filter(lambda x: x is not None, results))
        self._propagate_mask_changes()

        try:
            return np.logical_or.reduce(results)
        except UnboundLocalError:
            return
