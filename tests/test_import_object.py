import pytest

def test_import():
    import arvi
    from arvi import RV


def test_import_object():
    from arvi import _51Peg

    with pytest.raises(ValueError):
        from arvi import HD10180

