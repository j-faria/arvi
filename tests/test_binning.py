import pytest
import numpy as np
from arvi.binning import bin_ccf_mask


def test_bin_ccf_mask_basic():
    time = np.array([1.1, 1.2, 2.1, 2.2, 2.3])
    ccf_mask = np.array(['A', 'A', 'B', 'B', 'B'])
    expected = np.array(['A', 'B'])
    result = bin_ccf_mask(time, ccf_mask)
    assert np.array_equal(result, expected)


def test_bin_ccf_mask_nonunique():
    time = np.array([1.1, 1.2, 1.3, 2.1, 2.2])
    ccf_mask = np.array(['A', 'B', 'A', 'B', 'B'])
    expected = np.array(['nan', 'B'])
    result = bin_ccf_mask(time, ccf_mask)
    assert np.array_equal(result, expected)


def test_bin_ccf_mask_empty():
    time = np.array([])
    ccf_mask = np.array([])
    result = bin_ccf_mask(time, ccf_mask)
    assert result.size == 0
