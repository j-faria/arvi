import pytest

def test_import():
    from arvi.simbad_wrapper import simbad

def test_star():
    from arvi.simbad_wrapper import simbad
    s = simbad('HD69830')
    assert hasattr(s, 'ra')
    assert hasattr(s, 'dec')
    assert hasattr(s, 'coords')
    assert hasattr(s, 'plx')
    assert hasattr(s, 'sp_type')
    assert hasattr(s, 'B')
    assert hasattr(s, 'V')
    assert hasattr(s, 'ids')
    assert s.V == 5.95
    assert s.sp_type == 'G8:V'


