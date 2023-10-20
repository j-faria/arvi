import pytest
import os

def test_import():
    import arvi
    from arvi import RV


def test_import_object():
    from arvi import _51Peg

    if os.path.exists(os.path.expanduser('~/.dacerc')):
        from arvi import HD10180
    else:
        with pytest.raises(ValueError):
            from arvi import HD10180

