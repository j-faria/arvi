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


def test_list_instruments():
    from arvi import RV, config
    config.request_as_public = True
    _ = RV('HD28185', instrument='CORALIE')
    _ = RV('HD28185', instrument=['CORALIE'])
    _ = RV('HD28185', instrument=['CORALIE', 'HRS'])


def test_remove_instruments():
    from numpy import isin
    from arvi import RV, config
    config.request_as_public = True

    s = RV('51Peg', verbose=False)
    first_two = tuple(s.instruments[:2])
    s.remove_condition(isin(s.instrument_array, first_two))

    url = 'https://github.com/j-faria/arvi/issues/19'
    msg = f'did not remove instruments correctly, see {url}'
    assert first_two[0] not in s.instruments, msg
    assert first_two[1] not in s.instruments, msg
