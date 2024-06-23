import os, sys
import numpy as np

from .setup_logger import logger

# ESPRESSO ADC issues
from .utils import ESPRESSO_ADC_issues

def ADC_issues(self, plot=True, check_headers=False):
    """ Identify and mask points affected by ADC issues (ESPRESSO).

    Args:
        plot (bool, optional):
            Whether to plot the masked points.
        check_headers (bool, optional):
            Whether to (double-)check the headers for missing/zero keywords.
    """
    instruments = self._check_instrument('ESPRESSO')
    
    if len(instruments) < 1:
        if self.verbose:
            logger.error(f"no data from ESPRESSO")
            logger.info(f'available: {self.instruments}')
    

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

    self.mask[intersect] = False
    self._propagate_mask_changes()

    if plot:
        self.plot(show_masked=True)

    return intersect

# ESPRESSO cryostat issues
from .utils import ESPRESSO_cryostat_issues

def blue_cryostat_issues(self, plot=True):
    """ Identify and mask points affected by blue cryostat issues (ESPRESSO).

    Args:
        plot (bool, optional): Whether to plot the masked points.
    """
    instruments = self._check_instrument('ESPRESSO')
    
    if len(instruments) < 1:
        if self.verbose:
            logger.error(f"no data from ESPRESSO")
            logger.info(f'available: {self.instruments}')

    affected_file_roots = ESPRESSO_cryostat_issues()
    file_roots = [os.path.basename(f).replace('.fits', '') for f in self.raw_file]
    intersect = np.in1d(file_roots, affected_file_roots)

    total_affected = intersect.sum()

    if self.verbose:
        n = total_affected
        logger.info(f"there {'are'[:n^1]}{'is'[n^1:]} {n} frame{'s'[:n^1]} "
                     "affected by blue cryostat issues")

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


def known_issues(self, plot=False, **kwargs):
    """ Identify and mask known instrumental issues (ADC and blue cryostat for ESPRESSO)

    Args:
        plot (bool, optional): Whether to plot the masked points.
    """
    try:
        adc = ADC_issues(self, plot, **kwargs)
    except IndexError as e:
        # logger.error(e)
        logger.error('are the data binned? cannot proceed to mask these points...')

    try:
        cryostat = blue_cryostat_issues(self, plot)
    except IndexError as e:
        # logger.error(e)
        logger.error('are the data binned? cannot proceed to mask these points...')

    try:
        return adc | cryostat
    except UnboundLocalError:
        return
