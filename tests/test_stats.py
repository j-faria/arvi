import pytest
import numpy as np
from arvi.stats import wmean


def test_wmean():
    a = np.array([1, 2, 3])
    e = np.array([0.1, 0.2, 0.3])
    expected = (a/e**2).sum() / (1/e**2).sum()
    result = wmean(a, e)
    assert result == pytest.approx(expected)


def test_wmean_zero_weights():
    a = np.array([1, 2, 3])
    e = np.array([0, 0, 0])
    with pytest.raises(ZeroDivisionError):
        wmean(a, e)


def test_wmean_invalid_inputs():
    # Incompatible shapes
    a = np.array([1, 2, 3])
    e = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        wmean(a, e)

    # Negative uncertainties
    a = np.array([1, 2, 3])
    e = np.array([-0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        wmean(a, e)
