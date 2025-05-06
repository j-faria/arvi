# import pytest

def test_config_keys():
    from arvi import config
    assert hasattr(config, 'return_self'), 'config.return_self not found'
    assert hasattr(config, 'adjust_means_gls'), 'config.adjust_means_gls not found'
    assert hasattr(config, 'check_internet'), 'config.check_internet not found'
    assert hasattr(config, 'request_as_public'), 'config.request_as_public not found'
    assert hasattr(config, 'fancy_import'), 'config.fancy_import not found'
    assert hasattr(config, 'debug'), 'config.debug not found'


def test_config_set():
    from arvi import config
    config.return_self = True
    assert config.return_self

    config.new_option = False
    assert not config.new_option

    config.new_option_str = 'hello'
    assert config.new_option_str == 'hello'
