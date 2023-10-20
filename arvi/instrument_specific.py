import os, sys
import numpy as np
import matplotlib.pyplot as plt

from .setup_logger import logger

# ESPRESSO ADC issues
from .utils import ESPRESSO_ADC_issues

def ADC_issues(self, plot=True, check_keywords=True):
    instruments = self._check_instrument('ESPRESSO')
    
    if len(instruments) < 1:
        if self.verbose:
            logger.error(f"no data from ESPRESSO")
            logger.info(f'available: {self.instruments}')
    

    affected_file_roots = ESPRESSO_ADC_issues()
    file_roots = [os.path.basename(f).replace('.fits', '') for f in self.raw_file]
    intersect = np.in1d(file_roots, affected_file_roots)

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


def known_issues(self, plot=False, **kwargs):
    adc = ADC_issues(self, plot, **kwargs)
    cryostat = blue_cryostat_issues(self, plot)
    return adc | cryostat
