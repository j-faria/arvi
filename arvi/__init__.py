__all__ = ['RV']

from .timeseries import RV

_ran_once = False

def __getattr__(name: str):
    if name in (
        '_ipython_canary_method_should_not_exist_',
        '_repr_mimebundle_',
        '__wrapped__'
    ):
        return

    global _ran_once  # can't do it any other way :(
    if _ran_once:
        return RV(name)
    else:
        _ran_once = True
