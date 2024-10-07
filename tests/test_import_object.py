import pytest
import os

def test_import():
    import arvi
    from arvi import RV


def test_import_object():
    from arvi import _51Peg

    from arvi import HD10180
    # if os.path.exists(os.path.expanduser('~/.dacerc')):
    # else:
    #     with pytest.raises(ImportError):
    #         from arvi import HD10180

