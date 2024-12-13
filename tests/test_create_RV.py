import pytest
import os
from numpy import isnan

@pytest.fixture
def change_test_dir(request):
    os.chdir(request.fspath.dirname)
    yield
    os.chdir(request.config.invocation_params.dir)

def test_from_rdb(change_test_dir):
    from arvi import RV
    s = RV.from_rdb('./HD10700-Bcor_ESPRESSO18.rdb', verbose=False)
    assert s.star == 'HD10700_Bcor'
    assert s.instruments == ['ESPRESSO18']
    assert s.N == 1
    assert s.fwhm.shape == (1,)
    assert (s.fwhm == 0).all()
    assert (s.bispan == 0).all()
    assert isnan(s.rhk).all()

