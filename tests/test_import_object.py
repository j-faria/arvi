import pytest
import os

def test_import():
    import arvi
    from arvi import RV


def test_import_object():
    from arvi import _51Peg
    from arvi import HD10180

def test_import_object_public():
    from arvi import config

    config.request_as_public = True
    
    from arvi import _51Peg
    from arvi import HD10180

    config.request_as_public = False


def test_fancy_import_off():
    from arvi import config

    from arvi import HD10180

    config.fancy_import = False

    with pytest.raises(ImportError):
        from arvi import HD69830

