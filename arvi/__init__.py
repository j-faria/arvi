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

from .timeseries import RV

def __getattr__(name: str):
    if not config.fancy_import:
        raise AttributeError
    
    if name in (
        '_ipython_canary_method_should_not_exist_',
        '_ipython_display_',
        '_repr_mimebundle_',
        '__wrapped__'
    ):
        return

    try:
        globals()[name] = RV(name)
        return globals()[name]
    except ValueError as e:
        raise AttributeError(e)

