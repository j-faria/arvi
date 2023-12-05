__all__ = ['RV']

from .timeseries import RV

_ran_once = False

def __getattr__(name: str):
    global _ran_once  # can't do it any other way :(
    if _ran_once:
        return RV(name)
    else:
        _ran_once = True
