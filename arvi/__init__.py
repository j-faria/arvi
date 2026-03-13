__all__ = ['RV', 'config', 'simbad', 'gaia']

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("arvi")
except PackageNotFoundError:
    # package is not installed
    pass

from .config import config
from .simbad_wrapper import simbad
from .gaia_wrapper import gaia

# from .timeseries import RV
from .dace_wrapper import load_spectroscopy
query_database = load_spectroscopy().query_database


def __getattr__(name: str):
    if name == 'RV':
        from .timeseries import RV
        return RV

    if not config.fancy_import:
        raise AttributeError
    
    if name.startswith('__'):
        return
    if name in (
        '_ipython_canary_method_should_not_exist_',
        '_ipython_display_',
        '_repr_mimebundle_',
        # '__custom_documentations__',
        # '__wrapped__',
        # '__dataframe__'
    ):
        return

    try:
        from .timeseries import RV
        globals()[name] = RV(name)
        return globals()[name]
    except ValueError as e:
        raise AttributeError(e)

